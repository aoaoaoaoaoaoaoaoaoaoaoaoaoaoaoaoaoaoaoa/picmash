use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Instant,
};

use anyhow::{Context, bail};
use directories::ProjectDirs;
use parking_lot::{Mutex, RwLock};
use rand::{Rng, rng, seq::SliceRandom};
use serde::Serialize;
use time::Duration;
use tokio::sync::Notify;
use tracing::{info, warn};

use crate::{
    asset_domain::{AssetDomainLabel, AssetDomainOracle, AssetDomainPrediction, AssetDomainStatus},
    config::{AppConfig, ImportPolicy, SourceConfig},
    face::{DetectedFace, align_face_for_display, align_face_for_embedding},
    facemash::{FaceOracle, FacePrediction, pool_embeddings, rate_face_win},
    maintenance::{
        ClaimedMaintenanceJob, MaintenanceJobKind, MaintenanceJobSpec, MaintenancePriority,
    },
    model::{
        ARENA_RECENT_REPEAT_EXCLUDE, ArenaCard, ArenaHandle, ArenaLocalCard, ArenaPair,
        ArenaRemoteCard, ArenaView, AssetDomainView, AssetId, AssetQualitySummary, AssetRecord,
        BoardEntry, ClusterSatellite, CorpusId, DuplicateCluster, ExploreEntry, ExploreMapMode,
        ExploreNeighbor, ExplorePanels, ExploreSelection, ExploreTriad, ExploreView,
        ExternalArenaStatus, ExternalEventKind, FaceId, FaceIdentityId, LATENT_DIM, MAP_DIM,
        PosteriorSummary, ProjectionModel, RemoteCandidate, RemoteItemId, SIMILARITY_DIM,
        SessionEmbeddingHead, SessionId, SessionRecord, SimilarityChoice, SimilarityModel,
        certainty, dot, learned_reduce_points, prepare_raw_layout_space, sample_softmax_index,
        session_utility, sigmoid, subtract, weighted_choice_index,
    },
    onnx::OnnxEngine,
    quality::{
        HierarchicalAssetPosterior, HierarchicalSessionPosterior, LEGACY_EXACT_OFFSET_EPSILON,
        LEGACY_HEART_GLOBAL_BOOST, LEGACY_HEART_SESSION_BOOST, LEGACY_L2_FRONTIER,
        LEGACY_L2_SESSION_HEAD, LEGACY_LR_DUEL_ALPHA, LEGACY_LR_DUEL_COORD, LEGACY_LR_DUEL_HEAD,
        LEGACY_LR_DUEL_MOOD, LEGACY_LR_PROJECTION, LEGACY_PROJECTION_WEIGHT_DECAY,
        LegacyUnaryFeedback, PerturbativeAssetPosterior, PerturbativeHyperParamsV3,
        PerturbativeSessionPosterior, QualityFormalVersion, legacy_batter_asset,
        legacy_projection_prior, legacy_shove_mood,
    },
    sources::{SourceHarvest, SourceScanner},
    store::{FaceIdentityRecord, Store},
    vptree::VpTree,
};

mod arena;
mod explore;
mod external;
mod facemash;
mod gate;
mod identities;
mod lifecycle;
mod maintenance;
mod orchestration;
mod ready_frontier;
mod runtime;
mod state;
mod support;
#[cfg(test)]
mod tests;

use self::ready_frontier::{
    REMOTE_SOURCE_IDLE_SCAN_GRACE, ReadyTargetProfile, SourceReadyFrontier,
};
use self::runtime::surviving_local_anchor;
use self::state::{
    ActiveArena, ConfigReloadState, DuplicateFrontier, ExploreLayoutCache, ExploreVectorCache,
    FacemashPairKey, LockExhaustionPolicy, ScoredRemoteCandidate, SessionField, SimilarityField,
};
use self::support::{
    choose_local_pair, choose_local_pair_against_anchor, hierarchical_asset_cache,
    hierarchical_canonical_mean, hierarchical_face_summary, hierarchical_semantic_summary,
    hierarchical_session_cache, hierarchical_session_utility_variance, hierarchical_total_summary,
    hierarchical_vibe_summary, normalize_embedding, perturbative_remote_total_summary,
    perturbative_semantic_summary, perturbative_total_summary, perturbative_vibe_summary,
    projection_state, raw_distance_sq, redirect_target_for_pair, remote_baseline_summary,
    remote_total_summary, remote_vibe_summary, sanitize_score, squared_similarity_gap,
    visible_assets,
};
pub use self::{
    facemash::{
        FaceFrameOverlay, FacemashFaceView, FacemashLocalAssetView, FacemashPairView,
        FacemashStatus,
    },
    identities::{
        IdentityReviewCandidateView, IdentityReviewRowView, IdentityReviewStatus,
        IdentityReviewView,
    },
    runtime::{RedirectTarget, RuntimePhase, RuntimeSnapshot, RuntimeState, SharedRuntimeState},
    state::{BoardView, StartupSummary},
};

const SESSION_SNAP_WINDOW: Duration = Duration::minutes(10);
const FACE_CROP_CACHE_VERSION: &str = "v6";
const FACEMASH_PAIR_ENTROPY_TEMPERATURE: f32 = 120.0;
const FACEMASH_MODEL_TOPNESS_FLOOR: f32 = 0.3;
const FACEMASH_FRONTIER_TOPNESS_WEIGHT: f32 = 420.0;
const FACEMASH_FRONTIER_POSTERIOR_SIGMA_WEIGHT: f32 = 0.85;
const FACEMASH_FRONTIER_MODEL_SIGMA_WEIGHT: f32 = 0.1;
const FACEMASH_FRONTIER_SCARCITY_WEIGHT: f32 = 180.0;
const FACEMASH_COVERAGE_TOPNESS_WEIGHT: f32 = 260.0;
const FACEMASH_COVERAGE_POSTERIOR_SIGMA_WEIGHT: f32 = 1.2;
const FACEMASH_COVERAGE_MODEL_SIGMA_WEIGHT: f32 = 0.12;
const FACEMASH_COVERAGE_SCARCITY_WEIGHT: f32 = 220.0;
const FACEMASH_PAIR_POSTERIOR_SIGMA_WEIGHT: f32 = 1.0;
const FACEMASH_PAIR_MODEL_SIGMA_WEIGHT: f32 = 0.15;
const FACEMASH_PAIR_SCARCITY_WEIGHT: f32 = 48.0;
const FACEMASH_CANDIDATE_LIMIT: usize = 4096;
const FACEMASH_FRONTIER_SHORTLIST: usize = 32;
const FACEMASH_COVERAGE_SHORTLIST: usize = 24;
const FACEMASH_RECENT_PAIR_EXCLUDE: usize = 48;
const FACEMASH_RECENT_FACE_EXCLUDE: usize = 16;
const CONFIG_RELOAD_PULSE: Duration = Duration::seconds(2);
const IDENTITY_HANDLE_MAC_LEN: usize = 16;
const QUALITY_MODEL_REFRESH_DEBOUNCE_SECONDS: u64 = 4;

const L2_OFFSET: f32 = 0.012;
const LR_SIMILARITY: f32 = 0.06;
const SIMILARITY_BETA: f32 = 1.35;
const SIMILARITY_WEIGHT_DECAY: f32 = 0.0025;
const EXPLORE_RECENT_EXCLUDE: usize = 18;
const EXPLORE_TRIAD_ANCHORS: usize = 24;
const EXPLORE_TRIAD_NEIGHBORS: usize = 8;
const EXPLORE_SELECTION_NEIGHBORS: usize = 14;
const EXPLORE_TRIAD_TOP_K: usize = 12;
const EXTERNAL_RECENT_EXCLUDE: usize = 14;
const EXTERNAL_STREAM_RECENT_EXCLUDE: usize = 8;
const EXTERNAL_STREAM_HARD_EXCLUDE: usize = 16;
const EXTERNAL_SOURCE_RECENT_EXCLUDE_CAP: usize = 6;
const EXTERNAL_FACE_BACKFILL_BATCH: usize = 32;
const ARENA_EXPLORE_SIGMA_WEIGHT: f32 = 0.95;
const ARENA_EXPLORE_FRONTIER_WEIGHT: f32 = 1.2;
const ARENA_EXPLORE_SCARCITY_WEIGHT: f32 = 0.8;
const ARENA_EXPLOIT_TEMPERATURE: f32 = 0.42;
const ARENA_EXPLORE_TEMPERATURE: f32 = 1.18;
const ARENA_EXPLOIT_UNIFORM_MIX: f32 = 0.03;
const ARENA_EXPLORE_UNIFORM_MIX: f32 = 0.1;
const ARENA_PAIR_CLOSENESS_WEIGHT: f32 = 1.35;
const ARENA_PAIR_FRONTIER_WEIGHT: f32 = 0.55;
const ARENA_PAIR_SCARCITY_WEIGHT: f32 = 0.75;
const ARENA_PAIR_STYLE_WEIGHT: f32 = 1.0;
const ARENA_REMOTE_PAIR_BALANCE_WEIGHT: f32 = 1.4;
const ARENA_REMOTE_ACCEPT_WEIGHT: f32 = 1.1;
const ARENA_REMOTE_ITEM_RECENCY_WEIGHT: f32 = 0.65;
const ARENA_REMOTE_SOURCE_WEIGHT: f32 = 0.35;
const ARENA_REMOTE_SOURCE_RECENCY_WEIGHT: f32 = 0.75;
const ARENA_REMOTE_STREAM_SIZE_WEIGHT: f32 = 0.35;
const ARENA_REMOTE_STREAM_FRESHNESS_WEIGHT: f32 = 0.55;
const ARENA_REMOTE_STREAM_RECENCY_WEIGHT: f32 = 0.55;
const EXTERNAL_SCAN_PULSE: Duration = Duration::seconds(20);

fn arena_sampling_temperature(explore: f32) -> f32 {
    let explore = explore.clamp(0.0, 1.0);
    ARENA_EXPLOIT_TEMPERATURE + explore * (ARENA_EXPLORE_TEMPERATURE - ARENA_EXPLOIT_TEMPERATURE)
}

fn arena_uniform_mix(explore: f32) -> f32 {
    let explore = explore.clamp(0.0, 1.0);
    ARENA_EXPLOIT_UNIFORM_MIX + explore * (ARENA_EXPLORE_UNIFORM_MIX - ARENA_EXPLOIT_UNIFORM_MIX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryFeedback {
    Less,
    More,
}

impl UnaryFeedback {
    fn from_nudge(direction: i32) -> anyhow::Result<Self> {
        match direction.signum() {
            -1 => Ok(Self::Less),
            1 => Ok(Self::More),
            _ => bail!("unary nudge direction must be ±1"),
        }
    }

    const fn into_legacy(self) -> LegacyUnaryFeedback {
        match self {
            Self::Less => LegacyUnaryFeedback::Less,
            Self::More => LegacyUnaryFeedback::More,
        }
    }
}

pub struct AppState {
    store: Mutex<Store>,
    db_write_gate: gate::DbWriteGate,
    db_path: PathBuf,
    config: RwLock<AppConfig>,
    config_path: PathBuf,
    config_reload: Mutex<ConfigReloadState>,
    embedder: OnnxEngine,
    source_scanner: SourceScanner,
    active: ActiveArena,
    root_path: PathBuf,
    cache_root: std::path::PathBuf,
    explore_layouts: RwLock<HashMap<ExploreMapMode, ExploreLayoutCache>>,
    explore_vectors: RwLock<Option<ExploreVectorCache>>,
    asset_domain_oracle: RwLock<Option<AssetDomainOracle>>,
    duplicate_frontier: RwLock<Option<DuplicateFrontier>>,
    face_oracle: RwLock<Option<FaceOracle>>,
    maintenance_notify: Notify,
    recent_facemash_pairs: Mutex<VecDeque<FacemashPairKey>>,
    recent_facemash_identities: Mutex<VecDeque<FaceIdentityId>>,
    recent_facemash_faces: Mutex<VecDeque<FaceId>>,
    identity_handle_key: [u8; blake3::KEY_LEN],
}

pub type SharedAppState = Arc<AppState>;
