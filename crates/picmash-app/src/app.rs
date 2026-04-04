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

mod explore;
mod external;
mod facemash;
mod gate;
mod identities;
mod lifecycle;
mod maintenance;
mod ready_frontier;
mod support;
#[cfg(test)]
mod tests;

use self::ready_frontier::{
    REMOTE_SOURCE_IDLE_SCAN_GRACE, ReadyTargetProfile, SourceReadyFrontier,
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

#[derive(Debug, Clone, Copy)]
pub struct StartupSummary {
    pub corpus_id: CorpusId,
    pub session_id: SessionId,
    pub visible_assets: usize,
    pub embedded_assets: usize,
}

#[derive(Debug, Clone)]
pub struct BoardView {
    pub entries: Vec<BoardEntry>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveArena {
    corpus_id: CorpusId,
    session_id: SessionId,
}

#[derive(Debug, Clone)]
struct ExploreLayoutCache {
    asset_ids: Vec<AssetId>,
    plots: Vec<[f32; MAP_DIM]>,
}

#[derive(Debug, Clone)]
struct ExploreVectorCache {
    asset_ids: Vec<AssetId>,
    embeddings: HashMap<AssetId, Vec<f32>>,
    raw_vectors: HashMap<AssetId, Vec<f32>>,
    learned_corpus: Vec<Vec<f32>>,
    latents: HashMap<AssetId, [f32; SIMILARITY_DIM]>,
    model: SimilarityModel,
}

impl ExploreVectorCache {
    fn refresh_learned_geometry(&mut self) {
        self.latents = self
            .asset_ids
            .iter()
            .filter_map(|asset_id| {
                self.embeddings
                    .get(asset_id)
                    .map(|embedding| (asset_id, embedding))
            })
            .map(|(asset_id, embedding)| {
                (
                    asset_id.clone(),
                    self.model.project_asset(asset_id, embedding),
                )
            })
            .filter(|(_, latent)| latent.iter().all(|value| value.is_finite()))
            .collect::<HashMap<_, _>>();
        self.learned_corpus = self
            .asset_ids
            .iter()
            .filter_map(|asset_id| {
                self.latents
                    .get(asset_id)
                    .map(|latent: &[f32; SIMILARITY_DIM]| latent.to_vec())
            })
            .collect::<Vec<_>>();
    }

    fn embedding(&self, asset_id: &AssetId) -> Option<&[f32]> {
        self.embeddings.get(asset_id).map(Vec::as_slice)
    }

    fn learned_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        Some(
            self.model
                .distance_sq(lhs, self.embedding(lhs)?, rhs, self.embedding(rhs)?),
        )
    }

    fn nearest_neighbor_ids(
        &self,
        mode: ExploreMapMode,
        asset_id: &AssetId,
        limit: usize,
        include_self: bool,
    ) -> Vec<(AssetId, f32)> {
        let mut neighbors = self
            .asset_ids
            .iter()
            .filter_map(|candidate_id| {
                let distance = match mode {
                    ExploreMapMode::Raw => {
                        raw_distance_sq(&self.raw_vectors, asset_id, candidate_id)?
                    }
                    ExploreMapMode::Learned => self.learned_distance_sq(asset_id, candidate_id)?,
                };
                if !include_self && candidate_id == asset_id {
                    return None;
                }
                Some((candidate_id.clone(), distance))
            })
            .collect::<Vec<_>>();
        neighbors.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.0.cmp(&rhs.0.0)));
        if include_self {
            neighbors
                .retain(|(candidate_id, distance)| candidate_id != asset_id || *distance <= 1e-6);
        }
        neighbors.truncate(limit);
        neighbors
    }

    fn raw_layout_corpus(&self) -> Vec<Vec<f32>> {
        let raw_corpus = self
            .asset_ids
            .iter()
            .filter_map(|asset_id| self.raw_vectors.get(asset_id).cloned())
            .collect::<Vec<_>>();
        let prepared = prepare_raw_layout_space(&raw_corpus);
        if prepared.len() == raw_corpus.len() {
            prepared
        } else {
            raw_corpus
        }
    }
}

#[derive(Debug, Clone)]
struct ScoredRemoteCandidate {
    candidate: crate::model::RemoteCandidate,
    utility: f32,
    quality: AssetQualitySummary,
    selection_score: f32,
}

#[derive(Debug, Clone)]
struct SessionField {
    session: SessionRecord,
    quality_model: QualityFormalVersion,
    exact_offsets: HashMap<AssetId, f32>,
    hearted_assets: HashSet<AssetId>,
    subsource_lock: Option<crate::model::SessionSubsourceLock>,
    embeddings: HashMap<AssetId, Vec<f32>>,
    embedding_head: Option<SessionEmbeddingHead>,
    hierarchical_assets: HashMap<AssetId, HierarchicalAssetPosterior>,
    dominant_faces: HashMap<AssetId, FaceIdentityRecord>,
    hierarchical_session: Option<HierarchicalSessionPosterior>,
    perturbative_assets: HashMap<AssetId, PerturbativeAssetPosterior>,
    perturbative_session: Option<PerturbativeSessionPosterior>,
    perturbative_hyper: PerturbativeHyperParamsV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockExhaustionPolicy {
    ClearAndRetry,
    PreserveAndStop,
}

impl SessionField {
    fn exact_offset(&self, asset_id: &AssetId) -> f32 {
        self.exact_offsets.get(asset_id).copied().unwrap_or(0.0)
    }

    fn hearted(&self, asset_id: &AssetId) -> bool {
        self.hearted_assets.contains(asset_id)
    }

    fn heart_bias(&self, asset: &AssetRecord) -> f32 {
        asset.heart_count as f32 * LEGACY_HEART_GLOBAL_BOOST
            + if asset.is_hearted {
                LEGACY_HEART_SESSION_BOOST
            } else {
                0.0
            }
    }

    fn embedding(&self, asset_id: &AssetId) -> Option<&[f32]> {
        self.embeddings.get(asset_id).map(Vec::as_slice)
    }

    fn residual_score_for_embedding(&self, embedding: &[f32]) -> f32 {
        self.embedding_head
            .as_ref()
            .map_or(0.0, |head| head.score(embedding))
    }

    fn residual_score(&self, asset_id: &AssetId) -> f32 {
        match (&self.embedding_head, self.embedding(asset_id)) {
            (Some(head), Some(embedding)) => head.score(embedding),
            _ => 0.0,
        }
    }

    fn utility(&self, asset: &AssetRecord) -> f32 {
        if let (
            QualityFormalVersion::HierarchicalPerturbativeV3,
            Some(asset_quality),
            Some(session_quality),
        ) = (
            self.quality_model,
            self.perturbative_assets.get(&asset.id),
            self.perturbative_session,
        ) {
            return perturbative_total_summary(
                asset_quality,
                &session_quality,
                self.perturbative_hyper,
                self.dominant_faces
                    .get(&asset.id)
                    .map(|identity| PosteriorSummary {
                        mean: identity.beauty.mean,
                        sigma: identity.beauty.sigma,
                    }),
            )
            .mean
                + self.exact_offset(&asset.id)
                + self.heart_bias(asset);
        }
        if let (
            QualityFormalVersion::HierarchicalGaussianV1,
            Some(asset_quality),
            Some(session_quality),
        ) = (
            self.quality_model,
            self.hierarchical_assets.get(&asset.id),
            self.hierarchical_session,
        ) {
            return asset_quality.canonical_mean
                + dot(
                    &asset_quality.mood_loading_mean,
                    &session_quality.semantic_mood_mean,
                )
                + asset_quality
                    .technical_mean
                    .map(|_| {
                        asset_quality
                            .vibe_mean
                            .iter()
                            .zip(session_quality.vibe_mean.iter())
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum::<f32>()
                    })
                    .unwrap_or_default()
                + self.residual_score(&asset.id)
                + self.exact_offset(&asset.id)
                + self.heart_bias(asset);
        }
        session_utility(
            asset,
            &self.session,
            self.residual_score(&asset.id),
            self.exact_offset(&asset.id),
            self.heart_bias(asset),
        )
    }

    fn focus(&self, asset: &AssetRecord) -> f32 {
        self.utility(asset)
            - match self.quality_model {
                QualityFormalVersion::HierarchicalPerturbativeV2
                | QualityFormalVersion::HierarchicalPerturbativeV3 => self
                    .perturbative_session
                    .map_or(self.session.frontier, |session| session.threshold_mean),
                _ => self
                    .hierarchical_session
                    .map_or(self.session.frontier, |session| session.frontier_mean),
            }
    }

    fn quality_posterior(&self, asset: &AssetRecord) -> PosteriorSummary {
        self.quality_summary(asset).map_or(
            PosteriorSummary {
                mean: self.utility(asset),
                sigma: crate::quality::legacy_cache_variance(asset.compare_count).sqrt(),
            },
            |summary| summary.asset,
        )
    }

    fn arena_anchor_score(&self, asset: &AssetRecord, explore: f32) -> f32 {
        let posterior = self.quality_posterior(asset);
        let frontier_pull = sigmoid(self.focus(asset));
        let scarcity = 1.0 - certainty(asset.compare_count);
        posterior.mean
            + explore
                * (posterior.sigma * ARENA_EXPLORE_SIGMA_WEIGHT
                    + frontier_pull * ARENA_EXPLORE_FRONTIER_WEIGHT
                    + scarcity * ARENA_EXPLORE_SCARCITY_WEIGHT)
    }

    fn arena_opponent_score(
        &self,
        anchor: &AssetRecord,
        candidate: &AssetRecord,
        explore: f32,
    ) -> f32 {
        let posterior = self.quality_posterior(candidate);
        let anchor_focus = self.focus(anchor);
        let candidate_focus = self.focus(candidate);
        let closeness = 1.0 / (1.0 + (anchor_focus - candidate_focus).abs());
        let frontier_pressure = 1.0 / (1.0 + candidate_focus.abs());
        let scarcity =
            (1.0 - certainty(anchor.compare_count)) + (1.0 - certainty(candidate.compare_count));
        let style_gap = dot(
            &subtract(&anchor.coords, &candidate.coords),
            &self.session.mood,
        )
        .abs();
        let style_similarity = 1.0 / (1.0 + style_gap);
        posterior.mean
            + explore
                * (posterior.sigma * ARENA_EXPLORE_SIGMA_WEIGHT
                    + closeness * ARENA_PAIR_CLOSENESS_WEIGHT
                    + frontier_pressure * ARENA_PAIR_FRONTIER_WEIGHT
                    + scarcity * ARENA_PAIR_SCARCITY_WEIGHT
                    + style_similarity * ARENA_PAIR_STYLE_WEIGHT)
    }

    fn sampling_pull(&self, asset: &AssetRecord) -> f32 {
        let uncertainty = self
            .quality_summary(asset)
            .map(|summary| summary.asset.sigma / (1.0 + summary.asset.sigma))
            .unwrap_or_else(|| 1.0 - certainty(asset.compare_count));
        let frontier_pull = sigmoid(self.focus(asset));
        let weighted = 0.18 + uncertainty * 1.15 + frontier_pull * 0.95;
        sanitize_score(weighted)
    }

    fn quality_summary(&self, asset: &AssetRecord) -> Option<AssetQualitySummary> {
        match self.quality_model {
            QualityFormalVersion::LegacyIndependentV1 => Some(AssetQualitySummary {
                asset: PosteriorSummary {
                    mean: asset.alpha,
                    sigma: crate::quality::legacy_cache_variance(asset.compare_count).sqrt(),
                },
                baseline: PosteriorSummary {
                    mean: asset.alpha,
                    sigma: crate::quality::legacy_cache_variance(asset.compare_count).sqrt(),
                },
                semantic: None,
                vibe: None,
                technical: None,
                face: self
                    .dominant_faces
                    .get(&asset.id)
                    .map(|identity| PosteriorSummary {
                        mean: identity.beauty.mean,
                        sigma: identity.beauty.sigma,
                    }),
            }),
            QualityFormalVersion::HierarchicalGaussianV1 => {
                let asset_quality = self.hierarchical_assets.get(&asset.id)?;
                let session_quality = self.hierarchical_session?;
                let face = hierarchical_face_summary(self.dominant_faces.get(&asset.id));
                Some(AssetQualitySummary {
                    asset: hierarchical_total_summary(asset_quality, &session_quality, face),
                    baseline: PosteriorSummary {
                        mean: asset_quality.baseline_mean,
                        sigma: asset_quality.baseline_variance.max(0.0).sqrt(),
                    },
                    semantic: Some(hierarchical_semantic_summary(
                        asset_quality,
                        &session_quality,
                    )),
                    vibe: hierarchical_vibe_summary(asset_quality, &session_quality),
                    technical: asset_quality.technical_mean.map(|mean| PosteriorSummary {
                        mean,
                        sigma: asset_quality
                            .technical_variance
                            .unwrap_or_default()
                            .max(0.0)
                            .sqrt(),
                    }),
                    face,
                })
            }
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => {
                let asset_quality = self.perturbative_assets.get(&asset.id)?;
                let session_quality = self.perturbative_session?;
                let face = hierarchical_face_summary(self.dominant_faces.get(&asset.id));
                Some(AssetQualitySummary {
                    asset: perturbative_total_summary(
                        asset_quality,
                        &session_quality,
                        self.perturbative_hyper,
                        face,
                    ),
                    baseline: PosteriorSummary {
                        mean: asset_quality.baseline_mean,
                        sigma: asset_quality.baseline_variance.max(0.0).sqrt(),
                    },
                    semantic: Some(perturbative_semantic_summary(
                        asset_quality,
                        &session_quality,
                    )),
                    vibe: perturbative_vibe_summary(
                        asset_quality,
                        &session_quality,
                        self.perturbative_hyper,
                    ),
                    technical: asset_quality.technical_mean.map(|mean| PosteriorSummary {
                        mean,
                        sigma: asset_quality
                            .technical_variance
                            .unwrap_or_default()
                            .max(0.0)
                            .sqrt(),
                    }),
                    face,
                })
            }
        }
    }

    fn board_entry(&self, asset: AssetRecord) -> BoardEntry {
        let exact_offset = self.exact_offset(&asset.id);
        let residual_score = self.residual_score(&asset.id);
        let session_utility = self.utility(&asset);
        let quality = self.quality_summary_or_fallback(&asset);
        BoardEntry {
            global_score: quality.asset.mean,
            certainty: certainty(asset.compare_count),
            hearted: self.hearted(&asset.id),
            session_offset: exact_offset,
            residual_score,
            session_focus: self.focus(&asset),
            session_utility,
            sampling_pull: self.sampling_pull(&asset),
            quality,
            asset,
        }
    }
}

#[derive(Debug, Clone)]
struct SimilarityField {
    points: Vec<ExploreEntry>,
    raw_vectors: HashMap<AssetId, Vec<f32>>,
    latents: HashMap<AssetId, [f32; SIMILARITY_DIM]>,
    point_index: HashMap<AssetId, usize>,
}

impl SimilarityField {
    fn entry(&self, asset_id: &AssetId) -> Option<ExploreEntry> {
        self.point_index
            .get(asset_id)
            .and_then(|index| self.points.get(*index))
            .cloned()
    }

    fn latent(&self, asset_id: &AssetId) -> Option<&[f32; SIMILARITY_DIM]> {
        self.latents.get(asset_id)
    }

    fn learned_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        Some(squared_similarity_gap(self.latent(lhs)?, self.latent(rhs)?))
    }

    fn raw_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        raw_distance_sq(&self.raw_vectors, lhs, rhs)
    }

    fn distance_sq(&self, mode: ExploreMapMode, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        match mode {
            ExploreMapMode::Raw => self.raw_distance_sq(lhs, rhs),
            ExploreMapMode::Learned => self.learned_distance_sq(lhs, rhs),
        }
    }

    fn nearest_neighbors(
        &self,
        mode: ExploreMapMode,
        asset_id: &AssetId,
        limit: usize,
        include_self: bool,
    ) -> Vec<ExploreNeighbor> {
        let Some(focus) = self.entry(asset_id) else {
            return Vec::new();
        };
        let mut neighbors = self
            .points
            .iter()
            .filter_map(|entry| {
                let distance = self.distance_sq(mode, asset_id, &entry.asset.id)?;
                if !include_self && entry.asset.id == *asset_id {
                    return None;
                }
                Some(ExploreNeighbor {
                    entry: entry.clone(),
                    distance,
                })
            })
            .collect::<Vec<_>>();
        neighbors.sort_by(|lhs, rhs| {
            lhs.distance
                .total_cmp(&rhs.distance)
                .then_with(|| lhs.entry.asset.id.0.cmp(&rhs.entry.asset.id.0))
        });
        if include_self {
            neighbors.retain(|neighbor| {
                neighbor.entry.asset.id != focus.asset.id || neighbor.distance <= 1e-6
            });
        }
        neighbors.truncate(limit);
        neighbors
    }

    fn selection(
        &self,
        mode: ExploreMapMode,
        focus_id: Option<&AssetId>,
        triad: Option<&ExploreTriad>,
    ) -> Option<ExploreSelection> {
        let focus_id = focus_id
            .cloned()
            .or_else(|| triad.map(|current| current.a.asset.id.clone()))?;
        let focus = self.entry(&focus_id)?;
        Some(ExploreSelection {
            focus,
            neighbors: self.nearest_neighbors(mode, &focus_id, EXPLORE_SELECTION_NEIGHBORS, false),
        })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FacemashPairKey {
    left: FaceIdentityId,
    right: FaceIdentityId,
}

impl FacemashPairKey {
    fn forge(left: FaceIdentityId, right: FaceIdentityId) -> Self {
        if left.0 <= right.0 {
            Self { left, right }
        } else {
            Self {
                left: right,
                right: left,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DuplicateFrontier {
    tree: VpTree<RemoteItemId>,
}

#[derive(Debug, Clone)]
struct ConfigReloadState {
    live_digest: String,
    warned_rejected_digest: Option<String>,
}

impl ConfigReloadState {
    fn forge(live_digest: String) -> Self {
        Self {
            live_digest,
            warned_rejected_digest: None,
        }
    }

    fn is_live_digest(&self, digest: &str) -> bool {
        self.live_digest == digest
    }

    fn should_warn_rejected(&self, digest: &str) -> bool {
        self.warned_rejected_digest.as_deref() != Some(digest)
    }

    fn note_rejected(&mut self, digest: String) {
        self.warned_rejected_digest = Some(digest);
    }

    fn note_applied(&mut self, digest: String) {
        self.live_digest = digest;
        self.warned_rejected_digest = None;
    }
}

impl AppState {
    pub fn home_target(&self) -> anyhow::Result<RedirectTarget> {
        self.arena_target()
    }

    pub fn config_reload_pulse(&self) -> Duration {
        CONFIG_RELOAD_PULSE
    }

    pub fn external_scan_pulse(&self) -> Duration {
        EXTERNAL_SCAN_PULSE
    }

    pub fn quality_refresh_is_inline(&self) -> anyhow::Result<bool> {
        let store = self.store.lock();
        Ok(matches!(
            store.active_quality_model()?.formal_version,
            QualityFormalVersion::LegacyIndependentV1
                | QualityFormalVersion::HierarchicalGaussianV1
        ))
    }

    pub fn reload_config_if_changed(&self) -> anyhow::Result<()> {
        let raw = match fs::read_to_string(&self.config_path) {
            Ok(raw) => raw,
            Err(error) => {
                warn!(
                    path = %self.config_path.display(),
                    error = %format!("{error:#}"),
                    "live config reload skipped"
                );
                return Ok(());
            }
        };
        let digest = blake3::hash(raw.as_bytes()).to_hex().to_string();
        {
            let reload = self.config_reload.lock();
            if reload.is_live_digest(&digest) {
                return Ok(());
            }
        }
        let (mut candidate, _) = match AppConfig::parse(&raw, &self.config_path) {
            Ok(parsed) => parsed,
            Err(error) => {
                let mut reload = self.config_reload.lock();
                if reload.should_warn_rejected(&digest) {
                    warn!(
                        path = %self.config_path.display(),
                        error = %format!("{error:#}"),
                        "ignoring invalid live config update"
                    );
                    reload.note_rejected(digest);
                }
                return Ok(());
            }
        };

        let current_runtime = self.config.read().runtime.clone();
        if candidate.runtime.bind_addr != current_runtime.bind_addr
            || candidate.runtime.corpus_root != current_runtime.corpus_root
        {
            warn!(
                path = %self.config_path.display(),
                "ignoring live changes to runtime config; restart required for bind_addr/corpus_root"
            );
            candidate.runtime = current_runtime;
        }

        *self.config.write() = candidate;
        self.config_reload.lock().note_applied(digest);
        info!(path = %self.config_path.display(), "reloaded live config");
        Ok(())
    }

    fn persist_live_config(&self, config: &AppConfig) -> anyhow::Result<()> {
        let digest = config.write(&self.config_path)?;
        self.config_reload.lock().note_applied(digest);
        Ok(())
    }

    pub fn external_status(&self) -> anyhow::Result<ExternalArenaStatus> {
        let config = self.config.read().clone();
        let sources = config
            .sources
            .iter()
            .map(|source| crate::model::ExternalSourceOption {
                source_key: source.source_key(),
                label: source.display_name(),
                weight: source.weight,
            })
            .collect::<Vec<_>>();
        let store = self.store.lock();
        let (active_streams, blocked_streams, cached_items) =
            config
                .sources
                .iter()
                .try_fold((0usize, 0usize, 0usize), |counts, source| {
                    let (active, blocked, cached) =
                        store.external_source_counts(&source.source_key())?;
                    Ok::<_, anyhow::Error>((
                        counts.0 + active,
                        counts.1 + blocked,
                        counts.2 + cached,
                    ))
                })?;
        Ok(ExternalArenaStatus {
            sources,
            external_probability: (config.external_probability() * 100.0).round() as u8,
            arena_explore_percent: (config.arena_explore() * 100.0).round() as u8,
            active_streams,
            blocked_streams,
            cached_items,
            dedup_radius_percent: (config.dedup_radius() * 100.0).round() as u8,
        })
    }

    pub fn set_external_probability_percent(&self, percent: u8) -> anyhow::Result<()> {
        let probability = f32::from(percent) / 100.0;
        let snapshot = {
            let mut config = self.config.write();
            config.shove_external_probability(probability);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(
            external_probability = probability,
            "updated external sampling probability"
        );
        Ok(())
    }

    pub fn arena_explore(&self) -> f32 {
        self.config.read().arena_explore()
    }

    pub fn set_arena_explore_percent(&self, percent: u8) -> anyhow::Result<()> {
        let explore = f32::from(percent) / 100.0;
        let snapshot = {
            let mut config = self.config.write();
            config.shove_arena_explore(explore);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(
            arena_explore = explore,
            "updated arena exploration pressure"
        );
        Ok(())
    }

    pub fn refresh_external_sources_if_due(&self, force: bool) -> anyhow::Result<()> {
        let mut due_sources = Vec::new();
        let mut due_local_sources = Vec::new();
        let store = self.store.lock();
        for source in self.configured_sources() {
            let source_key = source.source_key();
            let due = force
                || store.external_scan_due(
                    &source_key,
                    Duration::seconds(source.scan_interval_seconds as i64),
                )?;
            if !due {
                continue;
            }
            if !force && source.local_directory().is_some() {
                due_local_sources.push(source_key);
                continue;
            }
            if !force && self.remote_source_scan_can_rest(&store, &source)? {
                info!(source = %source_key, "remote source warm buffer saturated; skipping refresh");
                continue;
            }
            due_sources.push(source);
        }
        drop(store);
        for source_key in due_local_sources {
            self.schedule_local_directory_refresh(&source_key);
        }
        if due_sources.is_empty() {
            return Ok(());
        }

        let scanner = self.source_scanner.clone();
        let harvests = thread::scope(|scope| {
            #[allow(clippy::needless_collect)]
            let handles = due_sources
                .into_iter()
                .map(|source| {
                    let scanner = scanner.clone();
                    let source_key = source.source_key();
                    info!(source = %source_key, force, "refreshing external source");
                    scope.spawn(move || {
                        let result = scanner.harvest(&source);
                        (source, result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("external source worker panicked"))
                })
                .map(|handle| handle.map_err(anyhow::Error::from))
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let mut harvests = harvests;
        harvests.sort_by(|(lhs_source, lhs_harvest), (rhs_source, rhs_harvest)| {
            let lhs_local = lhs_source.local_directory().is_some();
            let rhs_local = rhs_source.local_directory().is_some();
            rhs_local
                .cmp(&lhs_local)
                .then_with(|| {
                    let lhs_streams = lhs_harvest
                        .as_ref()
                        .ok()
                        .map(|harvest| harvest.streams.len())
                        .unwrap_or(usize::MAX);
                    let rhs_streams = rhs_harvest
                        .as_ref()
                        .ok()
                        .map(|harvest| harvest.streams.len())
                        .unwrap_or(usize::MAX);
                    lhs_streams.cmp(&rhs_streams)
                })
                .then_with(|| lhs_source.source_key().cmp(&rhs_source.source_key()))
        });

        for (source, harvest) in harvests {
            let source_key = source.source_key();
            if let Err(error) =
                harvest.and_then(|harvest| self.devour_external_harvest(&source, &harvest))
            {
                let message = format!("{error:#}");
                warn!(source = %source_key, error = %message, "external source scan failed");
                self.with_db_write_gate(|| {
                    self.store.lock().external_scan_fault(&source_key, &message)
                })?;
            }
        }
        Ok(())
    }

    pub fn close(&self) -> anyhow::Result<()> {
        self.with_locked_store_write(|store| store.close_session(self.active.session_id))
    }

    pub(super) fn source_config_for_key(&self, source_key: &str) -> Option<SourceConfig> {
        self.config
            .read()
            .sources
            .iter()
            .find(|source| source.source_key() == source_key)
            .cloned()
    }

    fn configured_sources(&self) -> Vec<SourceConfig> {
        self.config.read().sources.clone()
    }

    fn asset_domain_oracle(&self, store: &Store) -> anyhow::Result<AssetDomainOracle> {
        if let Some(oracle) = self.asset_domain_oracle.read().as_ref().cloned() {
            return Ok(oracle);
        }
        let oracle = AssetDomainOracle::train(
            &store.asset_domain_training_rows(self.active.corpus_id, self.embedder.model_name())?,
        );
        *self.asset_domain_oracle.write() = Some(oracle.clone());
        Ok(oracle)
    }

    fn retrain_asset_domain_oracle(&self, store: &Store) -> anyhow::Result<AssetDomainOracle> {
        let oracle = AssetDomainOracle::train(
            &store.asset_domain_training_rows(self.active.corpus_id, self.embedder.model_name())?,
        );
        *self.asset_domain_oracle.write() = Some(oracle.clone());
        Ok(oracle)
    }

    fn asset_domain_overlay(
        &self,
        store: &Store,
        asset_ids: &[AssetId],
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> anyhow::Result<(AssetDomainStatus, HashMap<AssetId, AssetDomainView>)> {
        let manual = store.asset_domain_labels(asset_ids)?;
        let oracle = self.asset_domain_oracle(store)?;
        let status = oracle.status();
        let views = asset_ids
            .iter()
            .map(|asset_id| {
                let predicted = embeddings
                    .get(asset_id)
                    .and_then(|embedding| oracle.predict(embedding));
                (
                    asset_id.clone(),
                    AssetDomainView {
                        manual: manual.get(asset_id).copied(),
                        predicted,
                    },
                )
            })
            .collect();
        Ok((status, views))
    }

    fn asset_domain_view(
        &self,
        store: &Store,
        asset_id: &AssetId,
        embedding: Option<&[f32]>,
    ) -> anyhow::Result<AssetDomainView> {
        let manual = store
            .asset_domain_labels(std::slice::from_ref(asset_id))?
            .get(asset_id)
            .copied();
        let predicted = embedding.and_then(|embedding| {
            self.asset_domain_oracle(store)
                .ok()
                .and_then(|oracle| oracle.predict(embedding))
        });
        Ok(AssetDomainView { manual, predicted })
    }

    fn pairing_asset_domain_labels(
        &self,
        store: &Store,
        asset_ids: &[AssetId],
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> anyhow::Result<Option<HashMap<AssetId, AssetDomainLabel>>> {
        let (status, views) = self.asset_domain_overlay(store, asset_ids, embeddings)?;
        if !status.ready() {
            return Ok(None);
        }
        Ok(Some(
            asset_ids
                .iter()
                .filter_map(|asset_id| {
                    views.get(asset_id).copied().and_then(|view| {
                        view.manual
                            .or_else(|| view.predicted.map(AssetDomainPrediction::label))
                            .map(|label| (asset_id.clone(), label))
                    })
                })
                .collect(),
        ))
    }

    pub fn rescan(&self) -> anyhow::Result<StartupSummary> {
        let mut store = Store::open_hot(&self.db_path)?;
        store.ingest_corpus(&self.root_path, self.active.corpus_id, &self.embedder)?;
        self.with_locked_store_write(|store| store.touch_session(self.active.session_id))?;
        self.purge_explore_vectors();
        self.purge_all_explore_layouts();
        self.refresh_external_sources_if_due(true)?;
        self.startup_summary()
    }

    pub fn arena_target(&self) -> anyhow::Result<RedirectTarget> {
        self.redirect_target_for_next_pair()
    }

    pub fn arena_prefetch_target(&self) -> anyhow::Result<RedirectTarget> {
        redirect_target_for_pair(self.choose_next_pair(LockExhaustionPolicy::PreserveAndStop)?)
    }

    pub fn arena_prefetch_target_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
    ) -> anyhow::Result<RedirectTarget> {
        redirect_target_for_pair(self.choose_next_pair_preserving_local_anchor(
            local_anchor,
            LockExhaustionPolicy::PreserveAndStop,
        )?)
    }

    pub fn arena_empty(&self) -> anyhow::Result<ArenaView> {
        let store = self.store.lock();
        let _field = self.session_field(&store)?;
        Ok(ArenaView {
            pair: None,
            cluster: None,
        })
    }

    pub fn arena_pair(
        &self,
        left: &ArenaHandle,
        right: &ArenaHandle,
    ) -> anyhow::Result<Option<ArenaView>> {
        let store = self.store.lock();
        let field = self.session_field(&store)?;
        let Some(left) = self.load_arena_card(&store, &field, left)? else {
            return Ok(None);
        };
        let Some(right) = self.load_arena_card(&store, &field, right)? else {
            return Ok(None);
        };
        if left.handle() == right.handle() {
            return Ok(None);
        }
        let cluster = match &right {
            ArenaCard::Remote(remote_card) => {
                self.cluster_around_remote(&store, remote_card.item.id)?
            }
            ArenaCard::Local(_) => None,
        };

        Ok(Some(ArenaView {
            pair: Some(ArenaPair { left, right }),
            cluster,
        }))
    }

    pub fn board(&self) -> anyhow::Result<BoardView> {
        let store = self.store.lock();
        let field = self.session_field(&store)?;
        let mut entries = visible_assets(&store, self.active.corpus_id)?
            .into_iter()
            .map(|asset| field.board_entry(asset))
            .collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| {
            rhs.sampling_pull
                .total_cmp(&lhs.sampling_pull)
                .then_with(|| rhs.session_focus.total_cmp(&lhs.session_focus))
                .then_with(|| rhs.global_score.total_cmp(&lhs.global_score))
                .then_with(|| lhs.asset.id.0.cmp(&rhs.asset.id.0))
        });
        Ok(BoardView { entries })
    }

    pub fn rotate_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        self.with_locked_store_write(|store| {
            store.rotate_asset(asset_id, direction)?;
            store.touch_session(self.active.session_id)
        })
    }

    pub fn rotate_arena_handle(&self, handle: &ArenaHandle, direction: i32) -> anyhow::Result<()> {
        self.with_locked_store_write(|store| {
            match handle {
                ArenaHandle::Local(asset_id) => {
                    if store
                        .corpus_asset(self.active.corpus_id, asset_id)?
                        .is_none()
                    {
                        return Ok(());
                    }
                    store.rotate_asset(asset_id, direction)?;
                }
                ArenaHandle::Remote(item_id) => {
                    if store.remote_item(*item_id)?.is_none() {
                        return Ok(());
                    }
                    store.rotate_external_item(*item_id, direction)?;
                }
            }
            store.touch_session(self.active.session_id)
        })
    }

    pub fn hide_asset(&self, asset_id: &AssetId, hidden: bool) -> anyhow::Result<RedirectTarget> {
        let present = self.with_locked_store_write(|store| {
            let present = store
                .corpus_asset(self.active.corpus_id, asset_id)?
                .is_some();
            if !present {
                return Ok(false);
            }
            store.set_hidden(self.active.corpus_id, asset_id, hidden)?;
            self.purge_explore_vectors();
            self.purge_all_explore_layouts();
            store.touch_session(self.active.session_id)?;
            Ok(true)
        })?;
        if !present {
            return redirect_target_for_pair(None);
        }
        self.redirect_target_for_next_pair()
    }

    pub fn hide_arena_handle(
        &self,
        handle: &ArenaHandle,
        hidden: bool,
        cluster_ids: &[RemoteItemId],
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Local(asset_id) => self.hide_asset(asset_id, hidden),
            ArenaHandle::Remote(item_id) => {
                let local_anchor = surviving_local_anchor(handle, pair_left, pair_right).cloned();
                self.with_fresh_store_write(|store| {
                    if hidden {
                        store.reject_external_item(
                            self.active.session_id,
                            self.active.corpus_id,
                            *item_id,
                            local_anchor.as_ref(),
                            ExternalEventKind::Rejected,
                        )?;
                        for satellite_id in cluster_ids {
                            store.reject_external_item(
                                self.active.session_id,
                                self.active.corpus_id,
                                *satellite_id,
                                local_anchor.as_ref(),
                                ExternalEventKind::Rejected,
                            )?;
                        }
                        self.purge_duplicate_frontier();
                    }
                    store.touch_session(self.active.session_id)?;
                    Ok(())
                })?;
                self.redirect_target_preserving_local_anchor(local_anchor.as_ref())
            }
        }
    }

    pub fn veto_external_thread_for_handle(
        &self,
        handle: &ArenaHandle,
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        let ArenaHandle::Remote(item_id) = handle else {
            return self.arena_target();
        };
        let item = {
            let store = Store::open_hot(&self.db_path)?;
            store
                .remote_item(*item_id)?
                .with_context(|| format!("missing remote item {}", item_id.0))?
        };
        let local_anchor = surviving_local_anchor(handle, pair_left, pair_right).cloned();
        self.with_fresh_store_write(|store| {
            store.block_external_stream(
                self.active.session_id,
                self.active.corpus_id,
                *item_id,
                local_anchor.as_ref(),
            )?;
            self.purge_duplicate_frontier();
            info!(
                source = %item.source_key,
                thread_no = item.thread_no,
                title = %item.stream_title,
                "blocked external stream"
            );
            store.touch_session(self.active.session_id)?;
            Ok(())
        })?;
        self.redirect_target_preserving_local_anchor(local_anchor.as_ref())
    }

    pub fn set_external_subsource_lock_for_handle(
        &self,
        handle: &ArenaHandle,
        active: bool,
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Remote(item_id) => {
                self.set_external_subsource_lock(*item_id, active)?;
                Ok(RedirectTarget::ArenaPair {
                    left: pair_left.clone(),
                    right: pair_right.clone(),
                })
            }
            ArenaHandle::Local(_) => Ok(RedirectTarget::ArenaPair {
                left: pair_left.clone(),
                right: pair_right.clone(),
            }),
        }
    }

    pub fn hide_asset_from_board(&self, asset_id: &AssetId) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        self.with_locked_store_write(|store| {
            store.set_hidden(self.active.corpus_id, asset_id, true)?;
            self.purge_explore_vectors();
            self.purge_all_explore_layouts();
            store.touch_session(self.active.session_id)
        })
    }

    pub fn set_asset_domain_label(
        &self,
        asset_id: &AssetId,
        label: AssetDomainLabel,
    ) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        let status = self.with_locked_store_write(|store| {
            store.set_asset_domain_label(asset_id, label)?;
            let oracle = self.retrain_asset_domain_oracle(store)?;
            store.touch_session(self.active.session_id)?;
            Ok(oracle.status())
        })?;
        info!(
            asset_id = %asset_id.0,
            label = label.as_str(),
            real_labels = status.real_labels,
            anime_labels = status.anime_labels,
            trained = status.trained,
            "updated asset domain label"
        );
        Ok(())
    }

    pub fn nudge_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        self.apply_unary_feedback(asset_id, UnaryFeedback::from_nudge(direction)?)
    }

    pub fn set_heart_asset(&self, asset_id: &AssetId, active: bool) -> anyhow::Result<()> {
        self.with_locked_store_write(|store| {
            if !active {
                return Ok(());
            }
            let mut field = self.session_field(store)?;
            let Some(mut asset) = store
                .corpus_assets(self.active.corpus_id)?
                .into_iter()
                .find(|candidate| candidate.id == *asset_id)
            else {
                return Ok(());
            };

            if asset.is_hearted {
                return Ok(());
            }

            asset.is_hearted = true;
            asset.heart_count = asset.heart_count.saturating_add(1);
            field.session.hearts = field.session.hearts.saturating_add(1);
            field.hearted_assets.insert(asset_id.clone());

            store.persist_heart_step(&field.session, &asset, asset_id, active)?;
            Ok(())
        })
    }

    pub fn set_heart_arena_handle(
        &self,
        handle: &ArenaHandle,
        active: bool,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Local(asset_id) => {
                self.set_heart_asset(asset_id, active)?;
                Ok(RedirectTarget::ArenaRoot)
            }
            ArenaHandle::Remote(item_id) => {
                if active {
                    self.enshrine_remote_item(*item_id, true)?;
                }
                self.arena_target()
            }
        }
    }

    fn apply_unary_feedback(
        &self,
        asset_id: &AssetId,
        feedback: UnaryFeedback,
    ) -> anyhow::Result<()> {
        self.with_locked_store_write(|store| {
            let mut field = self.session_field(store)?;
            let Some(mut asset) = store
                .corpus_assets(self.active.corpus_id)?
                .into_iter()
                .find(|candidate| candidate.id == *asset_id)
            else {
                return Ok(());
            };

            let exact_offset = field.exact_offset(asset_id);
            let utility_before = field.utility(&asset);
            let frontier_before = match field.quality_model {
                QualityFormalVersion::HierarchicalPerturbativeV2
                | QualityFormalVersion::HierarchicalPerturbativeV3 => field
                    .perturbative_session
                    .map_or(field.session.frontier, |session| session.threshold_mean),
                _ => field
                    .hierarchical_session
                    .map_or(field.session.frontier, |session| session.frontier_mean),
            };
            let legacy_feedback = feedback.into_legacy();
            let tuning = legacy_feedback.tuning(utility_before, frontier_before);

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalPerturbativeV2
                    | QualityFormalVersion::HierarchicalPerturbativeV3
            ) {
                let next_offset =
                    exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);
                field.session.nudges += 1;
                store.persist_nudge_step(
                    &field.session,
                    &asset,
                    asset_id,
                    legacy_feedback.direction(),
                    utility_before,
                    frontier_before,
                    tuning.signal,
                    next_offset,
                    None,
                    None,
                )?;
                return Ok(());
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalGaussianV1
            ) {
                let mut session_quality = field.hierarchical_session.with_context(|| {
                    format!(
                        "missing hierarchical session posterior {}",
                        field.session.id.0
                    )
                })?;
                let next_offset =
                    exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);
                session_quality.frontier_mean += tuning.frontier_rate
                    * (-tuning.signal - LEGACY_L2_FRONTIER * session_quality.frontier_mean);
                field.session.frontier = session_quality.frontier_mean;
                field.session.nudges += 1;

                let embedding = field.embedding(asset_id).map(ToOwned::to_owned);
                let mut embedding_head = field.embedding_head.clone().or_else(|| {
                    embedding.as_ref().map(|vector| {
                        SessionEmbeddingHead::zero(
                            self.embedder.model_name().to_owned(),
                            vector.len(),
                        )
                    })
                });
                if let (Some(head), Some(vector)) = (&mut embedding_head, embedding.as_deref()) {
                    head.unary_step(
                        vector,
                        tuning.signal,
                        tuning.head_rate,
                        LEGACY_L2_SESSION_HEAD,
                    );
                }

                let asset_cache = field
                    .hierarchical_assets
                    .get(asset_id)
                    .copied()
                    .with_context(|| {
                        format!("missing hierarchical posterior for {}", asset_id.0)
                    })?;
                store.persist_hierarchical_nudge_step(
                    &field.session,
                    &asset,
                    asset_id,
                    legacy_feedback.direction(),
                    utility_before,
                    frontier_before,
                    tuning.signal,
                    next_offset,
                    embedding_head.as_ref(),
                    &hierarchical_asset_cache(
                        &asset_cache,
                        hierarchical_face_summary(field.dominant_faces.get(asset_id)),
                    ),
                    &hierarchical_session_cache(&session_quality),
                )?;
                if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
                    field.exact_offsets.remove(asset_id);
                } else {
                    field.exact_offsets.insert(asset_id.clone(), next_offset);
                }
                field.embedding_head = embedding_head;
                field.hierarchical_session = Some(session_quality);
                return Ok(());
            }

            let embedding = field.embedding(asset_id).map(ToOwned::to_owned);
            let mut projection =
                projection_state(store, self.embedder.model_name(), embedding.as_deref())?;
            let prior = legacy_projection_prior(projection.as_ref(), embedding.as_deref());
            let asset_before = asset.clone();

            legacy_batter_asset(
                &mut asset.alpha,
                &mut asset.coords,
                &field.session.mood,
                &prior,
                tuning.signal,
                tuning.alpha_rate,
                tuning.coord_rate,
            );
            legacy_shove_mood(
                &mut field.session.mood,
                &asset_before.coords,
                tuning.signal,
                tuning.mood_rate,
            );
            field.session.frontier += tuning.frontier_rate
                * (-tuning.signal - LEGACY_L2_FRONTIER * field.session.frontier);

            let next_offset =
                exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);

            let mut embedding_head = field.embedding_head.clone().or_else(|| {
                embedding.as_ref().map(|vector| {
                    SessionEmbeddingHead::zero(self.embedder.model_name().to_owned(), vector.len())
                })
            });
            if let (Some(head), Some(vector)) = (&mut embedding_head, embedding.as_deref()) {
                head.unary_step(
                    vector,
                    tuning.signal,
                    tuning.head_rate,
                    LEGACY_L2_SESSION_HEAD,
                );
            }
            if let (Some(model), Some(vector)) = (&mut projection, embedding.as_deref()) {
                model.gradient_step(
                    vector,
                    &asset.coords,
                    tuning.projection_rate,
                    LEGACY_PROJECTION_WEIGHT_DECAY,
                );
            }

            field.session.nudges += 1;
            store.persist_nudge_step(
                &field.session,
                &asset,
                asset_id,
                legacy_feedback.direction(),
                utility_before,
                frontier_before,
                tuning.signal,
                next_offset,
                projection.as_ref(),
                embedding_head.as_ref(),
            )?;

            if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
                field.exact_offsets.remove(asset_id);
            } else {
                field.exact_offsets.insert(asset_id.clone(), next_offset);
            }
            field.embedding_head = embedding_head;
            Ok(())
        })
    }

    pub fn image_asset(&self, asset_id: &AssetId) -> anyhow::Result<AssetRecord> {
        self.store
            .lock()
            .corpus_asset(self.active.corpus_id, asset_id)?
            .with_context(|| {
                format!(
                    "asset {} not found in corpus {}",
                    asset_id.0, self.active.corpus_id.0
                )
            })
    }

    pub fn maybe_image_asset(&self, asset_id: &AssetId) -> anyhow::Result<Option<AssetRecord>> {
        self.store
            .lock()
            .corpus_asset(self.active.corpus_id, asset_id)
    }

    pub fn arena_handle_is_live(&self, handle: &ArenaHandle) -> anyhow::Result<bool> {
        let store = self.store.lock();
        Ok(match handle {
            ArenaHandle::Local(asset_id) => store
                .corpus_asset(self.active.corpus_id, asset_id)?
                .filter(|asset| !asset.hidden && asset.path.exists())
                .is_some(),
            ArenaHandle::Remote(item_id) => store
                .remote_item(*item_id)?
                .is_some_and(|item| item.path.exists()),
        })
    }

    pub fn maybe_remote_item(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<crate::model::RemoteItemRecord>> {
        self.store.lock().remote_item(item_id)
    }

    pub fn hearted_assets(&self) -> anyhow::Result<HashSet<AssetId>> {
        self.store.lock().hearted_assets()
    }

    pub fn vote(
        &self,
        left: &ArenaHandle,
        right: &ArenaHandle,
        winner: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match (left, right) {
            (ArenaHandle::Local(left_id), ArenaHandle::Local(right_id)) => {
                if let ArenaHandle::Local(winner_id) = winner {
                    return self.vote_local(left_id, right_id, winner_id);
                }
                bail!("remote winner is impossible in a local duel");
            }
            (ArenaHandle::Local(local_id), ArenaHandle::Remote(remote_id))
            | (ArenaHandle::Remote(remote_id), ArenaHandle::Local(local_id)) => {
                self.vote_remote_duel(local_id, *remote_id, winner)
            }
            (ArenaHandle::Remote(_), ArenaHandle::Remote(_)) => {
                bail!("remote-vs-remote arena duels are forbidden");
            }
        }
    }

    fn vote_local(
        &self,
        left_id: &AssetId,
        right_id: &AssetId,
        winner_id: &AssetId,
    ) -> anyhow::Result<RedirectTarget> {
        self.with_locked_store_write(|store| {
            let corpus_id = self.active.corpus_id;
            let mut field = self.session_field(store)?;
            let assets = store.corpus_assets(corpus_id)?;
            let mut left = assets
                .iter()
                .find(|asset| asset.id == *left_id)
                .cloned()
                .with_context(|| format!("missing left asset {}", left_id.0))?;
            let mut right = assets
                .iter()
                .find(|asset| asset.id == *right_id)
                .cloned()
                .with_context(|| format!("missing right asset {}", right_id.0))?;

            if left.id == right.id {
                bail!("cannot compare an asset against itself");
            }
            if winner_id != &left.id && winner_id != &right.id {
                bail!("winner is not one of the compared assets");
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalPerturbativeV2
                    | QualityFormalVersion::HierarchicalPerturbativeV3
            ) {
                let left_before = left.clone();
                let right_before = right.clone();
                let left_utility = field.utility(&left);
                let right_utility = field.utility(&right);
                let left_won = winner_id == &left.id;
                left.compare_count += 1;
                right.compare_count += 1;
                if left_won {
                    left.win_count += 1;
                } else {
                    right.win_count += 1;
                }
                field.session.comparisons += 1;
                store.persist_duel_step(
                    &field.session,
                    &left_before,
                    &right_before,
                    &left,
                    &right,
                    winner_id,
                    left_utility,
                    right_utility,
                    None,
                    None,
                )?;
                return Ok(());
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalGaussianV1
            ) {
                let mut left_quality = field
                    .hierarchical_assets
                    .get(&left.id)
                    .copied()
                    .with_context(|| format!("missing hierarchical posterior for {}", left.id.0))?;
                let mut right_quality = field
                    .hierarchical_assets
                    .get(&right.id)
                    .copied()
                    .with_context(|| {
                        format!("missing hierarchical posterior for {}", right.id.0)
                    })?;
                let mut session_quality = field.hierarchical_session.with_context(|| {
                    format!(
                        "missing hierarchical session posterior {}",
                        field.session.id.0
                    )
                })?;
                let left_face = hierarchical_face_summary(field.dominant_faces.get(&left.id));
                let right_face = hierarchical_face_summary(field.dominant_faces.get(&right.id));
                let left_before = left.clone();
                let right_before = right.clone();
                let left_utility = field.utility(&left);
                let right_utility = field.utility(&right);
                let left_won = winner_id == &left.id;
                let outcome = if left_won { 1.0 } else { -1.0 };
                let delta_mean = left_utility - right_utility;
                let delta_variance = hierarchical_session_utility_variance(
                    &left_quality,
                    &session_quality,
                    left_face,
                ) + hierarchical_session_utility_variance(
                    &right_quality,
                    &session_quality,
                    right_face,
                );
                let moments = crate::quality::gaussian_duel_moment_match(
                    delta_mean,
                    delta_variance,
                    outcome,
                    crate::quality::HIERARCHICAL_DUEL_BETA,
                )
                .context("moment-matching hierarchical duel update")?;
                crate::quality::diagonal_adf_update(
                    &mut left_quality.baseline_mean,
                    &mut left_quality.baseline_variance,
                    1.0,
                    outcome,
                    moments,
                );
                crate::quality::diagonal_adf_update(
                    &mut right_quality.baseline_mean,
                    &mut right_quality.baseline_variance,
                    -1.0,
                    outcome,
                    moments,
                );
                for axis in 0..LATENT_DIM {
                    crate::quality::diagonal_adf_update(
                        &mut left_quality.mood_loading_mean[axis],
                        &mut left_quality.mood_loading_variance[axis],
                        session_quality.semantic_mood_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut right_quality.mood_loading_mean[axis],
                        &mut right_quality.mood_loading_variance[axis],
                        -session_quality.semantic_mood_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut session_quality.semantic_mood_mean[axis],
                        &mut session_quality.semantic_mood_variance[axis],
                        left_quality.mood_loading_mean[axis]
                            - right_quality.mood_loading_mean[axis],
                        outcome,
                        moments,
                    );
                }
                if let (Some(mean), Some(variance)) = (
                    &mut left_quality.technical_mean,
                    &mut left_quality.technical_variance,
                ) {
                    crate::quality::diagonal_adf_update(
                        mean,
                        variance,
                        crate::quality::HIERARCHICAL_TECH_WEIGHT,
                        outcome,
                        moments,
                    );
                }
                if let (Some(mean), Some(variance)) = (
                    &mut right_quality.technical_mean,
                    &mut right_quality.technical_variance,
                ) {
                    crate::quality::diagonal_adf_update(
                        mean,
                        variance,
                        -crate::quality::HIERARCHICAL_TECH_WEIGHT,
                        outcome,
                        moments,
                    );
                }
                for axis in 0..crate::quality_features::VIBE_DESCRIPTOR_DIM {
                    crate::quality::diagonal_adf_update(
                        &mut left_quality.vibe_mean[axis],
                        &mut left_quality.vibe_variance[axis],
                        session_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut right_quality.vibe_mean[axis],
                        &mut right_quality.vibe_variance[axis],
                        -session_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut session_quality.vibe_mean[axis],
                        &mut session_quality.vibe_variance[axis],
                        left_quality.vibe_mean[axis] - right_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                }
                let left_face_id = field
                    .dominant_faces
                    .get(&left.id)
                    .map(|identity| identity.id);
                let right_face_id = field
                    .dominant_faces
                    .get(&right.id)
                    .map(|identity| identity.id);
                let mut touched_subjects = BTreeSet::new();
                match (left_face_id, right_face_id) {
                    (Some(left_id), Some(right_id)) if left_id != right_id => {
                        if let Some(identity) = field.dominant_faces.get(&left.id) {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                crate::quality::hierarchical_face_backflow_coeff(),
                                outcome,
                                moments,
                            );
                            for entry in field
                                .dominant_faces
                                .values_mut()
                                .filter(|entry| entry.id == left_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(left_id);
                        }
                        if let Some(identity) = field.dominant_faces.get(&right.id) {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                -crate::quality::hierarchical_face_backflow_coeff(),
                                outcome,
                                moments,
                            );
                            for entry in field
                                .dominant_faces
                                .values_mut()
                                .filter(|entry| entry.id == right_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(right_id);
                        }
                    }
                    (Some(identity_id), None) | (None, Some(identity_id)) => {
                        let coefficient = if left_face_id == Some(identity_id) {
                            crate::quality::hierarchical_face_backflow_coeff()
                        } else {
                            -crate::quality::hierarchical_face_backflow_coeff()
                        };
                        if let Some(identity) = field
                            .dominant_faces
                            .values()
                            .find(|entry| entry.id == identity_id)
                            .cloned()
                        {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                coefficient,
                                outcome,
                                moments,
                            );
                            for entry in field
                                .dominant_faces
                                .values_mut()
                                .filter(|entry| entry.id == identity_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(identity_id);
                        }
                    }
                    _ => {}
                }
                if !touched_subjects.is_empty() {
                    let snapshot = touched_subjects
                        .into_iter()
                        .filter_map(|identity_id| {
                            field
                                .dominant_faces
                                .values()
                                .find(|entry| entry.id == identity_id)
                                .map(|entry| (entry.id, entry.beauty, entry.duel_count))
                        })
                        .collect::<Vec<_>>();
                    store.save_identity_beauty_snapshot(&snapshot)?;
                }
                let left_face = hierarchical_face_summary(field.dominant_faces.get(&left.id));
                let right_face = hierarchical_face_summary(field.dominant_faces.get(&right.id));

                left.compare_count += 1;
                right.compare_count += 1;
                if left_won {
                    left.win_count += 1;
                } else {
                    right.win_count += 1;
                }
                left.alpha = hierarchical_canonical_mean(&left_quality, left_face);
                right.alpha = hierarchical_canonical_mean(&right_quality, right_face);
                left.coords = left_quality.mood_loading_mean;
                right.coords = right_quality.mood_loading_mean;
                field.session.comparisons += 1;
                field.session.mood = session_quality.semantic_mood_mean;
                field.session.frontier = session_quality.frontier_mean;

                let left_embedding = field.embedding(&left.id).map(ToOwned::to_owned);
                let right_embedding = field.embedding(&right.id).map(ToOwned::to_owned);
                let mut embedding_head = field.embedding_head.clone().or_else(|| {
                    left_embedding
                        .as_ref()
                        .or(right_embedding.as_ref())
                        .map(|vector| {
                            SessionEmbeddingHead::zero(
                                self.embedder.model_name().to_owned(),
                                vector.len(),
                            )
                        })
                });
                let logistic_err = if left_won { 1.0 } else { 0.0 } - sigmoid(delta_mean);
                if let (Some(head), Some(lhs), Some(rhs)) = (
                    &mut embedding_head,
                    left_embedding.as_deref(),
                    right_embedding.as_deref(),
                ) {
                    head.contrast_step(
                        lhs,
                        rhs,
                        logistic_err,
                        LEGACY_LR_DUEL_HEAD,
                        LEGACY_L2_SESSION_HEAD,
                    );
                }

                store.persist_hierarchical_duel_step(
                    &field.session,
                    &left_before,
                    &right_before,
                    &left,
                    &right,
                    winner_id,
                    left_utility,
                    right_utility,
                    embedding_head.as_ref(),
                    &hierarchical_asset_cache(&left_quality, left_face),
                    &hierarchical_asset_cache(&right_quality, right_face),
                    &hierarchical_session_cache(&session_quality),
                )?;
                field.embedding_head = embedding_head;
                field
                    .hierarchical_assets
                    .insert(left.id.clone(), left_quality);
                field
                    .hierarchical_assets
                    .insert(right.id.clone(), right_quality);
                field.hierarchical_session = Some(session_quality);
                return Ok(());
            }

            let left_before = left.clone();
            let right_before = right.clone();
            let left_utility = field.utility(&left);
            let right_utility = field.utility(&right);
            let left_won = winner_id == &left.id;
            let y = if left_won { 1.0 } else { 0.0 };
            let err = y - sigmoid(left_utility - right_utility);
            let coord_gap = subtract(&left.coords, &right.coords);

            let left_embedding = field.embedding(&left.id).map(ToOwned::to_owned);
            let right_embedding = field.embedding(&right.id).map(ToOwned::to_owned);
            let mut projection = projection_state(
                store,
                self.embedder.model_name(),
                left_embedding.as_deref().or(right_embedding.as_deref()),
            )?;
            let left_prior =
                legacy_projection_prior(projection.as_ref(), left_embedding.as_deref());
            let right_prior =
                legacy_projection_prior(projection.as_ref(), right_embedding.as_deref());

            legacy_batter_asset(
                &mut left.alpha,
                &mut left.coords,
                &field.session.mood,
                &left_prior,
                err,
                LEGACY_LR_DUEL_ALPHA,
                LEGACY_LR_DUEL_COORD,
            );
            legacy_batter_asset(
                &mut right.alpha,
                &mut right.coords,
                &field.session.mood,
                &right_prior,
                -err,
                LEGACY_LR_DUEL_ALPHA,
                LEGACY_LR_DUEL_COORD,
            );
            for (mood, gap) in field.session.mood.iter_mut().zip(coord_gap.iter().copied()) {
                *mood += LEGACY_LR_DUEL_MOOD * (err * gap - crate::quality::LEGACY_L2_MOOD * *mood);
            }

            left.compare_count += 1;
            right.compare_count += 1;
            if left_won {
                left.win_count += 1;
            } else {
                right.win_count += 1;
            }
            field.session.comparisons += 1;

            let mut embedding_head = field.embedding_head.clone().or_else(|| {
                left_embedding
                    .as_ref()
                    .or(right_embedding.as_ref())
                    .map(|vector| {
                        SessionEmbeddingHead::zero(
                            self.embedder.model_name().to_owned(),
                            vector.len(),
                        )
                    })
            });
            if let (Some(head), Some(lhs), Some(rhs)) = (
                &mut embedding_head,
                left_embedding.as_deref(),
                right_embedding.as_deref(),
            ) {
                head.contrast_step(lhs, rhs, err, LEGACY_LR_DUEL_HEAD, LEGACY_L2_SESSION_HEAD);
            }
            if let Some(model) = &mut projection {
                if let Some(embedding) = &left_embedding {
                    model.gradient_step(
                        embedding,
                        &left.coords,
                        LEGACY_LR_PROJECTION,
                        LEGACY_PROJECTION_WEIGHT_DECAY,
                    );
                }
                if let Some(embedding) = &right_embedding {
                    model.gradient_step(
                        embedding,
                        &right.coords,
                        LEGACY_LR_PROJECTION,
                        LEGACY_PROJECTION_WEIGHT_DECAY,
                    );
                }
            }

            store.persist_duel_step(
                &field.session,
                &left_before,
                &right_before,
                &left,
                &right,
                winner_id,
                left_utility,
                right_utility,
                projection.as_ref(),
                embedding_head.as_ref(),
            )?;
            field.embedding_head = embedding_head;
            Ok(())
        })?;
        self.redirect_target_for_next_pair()
    }

    fn session_with_store(&self, store: &Store) -> anyhow::Result<SessionRecord> {
        store.session(self.active.session_id)
    }

    fn choose_next_pair(
        &self,
        lock_exhaustion: LockExhaustionPolicy,
    ) -> anyhow::Result<Option<ArenaPair>> {
        for _ in 0..2 {
            let store = self.store.lock();
            let field = self.session_field(&store)?;
            let assets = visible_assets(&store, self.active.corpus_id)?;
            if field.subsource_lock.is_some() {
                let pair =
                    self.choose_pair_in_locked_subsource_with_store(&store, &field, &assets, None)?;
                drop(store);
                if pair.is_some() {
                    return Ok(pair);
                }
                if lock_exhaustion == LockExhaustionPolicy::PreserveAndStop {
                    return Ok(None);
                }
                self.clear_external_subsource_lock()?;
                continue;
            }
            return self.choose_pair_with_store(&store, &field, &assets);
        }
        Ok(None)
    }

    fn choose_next_pair_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
        lock_exhaustion: LockExhaustionPolicy,
    ) -> anyhow::Result<Option<ArenaPair>> {
        for _ in 0..2 {
            let store = self.store.lock();
            let field = self.session_field(&store)?;
            if field.subsource_lock.is_some() {
                let assets = visible_assets(&store, self.active.corpus_id)?;
                let pair = self.choose_pair_in_locked_subsource_with_store(
                    &store,
                    &field,
                    &assets,
                    local_anchor,
                )?;
                drop(store);
                if pair.is_some() {
                    return Ok(pair);
                }
                if lock_exhaustion == LockExhaustionPolicy::PreserveAndStop {
                    return Ok(None);
                }
                self.clear_external_subsource_lock()?;
                continue;
            }
            return self.choose_pair_preserving_local_anchor_with_store(
                &store,
                &field,
                local_anchor,
            );
        }
        Ok(None)
    }

    fn redirect_target_for_next_pair(&self) -> anyhow::Result<RedirectTarget> {
        let pair = self.choose_next_pair(LockExhaustionPolicy::ClearAndRetry)?;
        if let Some(pair_ref) = pair.as_ref() {
            self.note_remote_pair_selected(pair_ref)?;
        }
        redirect_target_for_pair(pair)
    }

    fn redirect_target_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
    ) -> anyhow::Result<RedirectTarget> {
        let pair = self.choose_next_pair_preserving_local_anchor(
            local_anchor,
            LockExhaustionPolicy::ClearAndRetry,
        )?;
        if let Some(pair_ref) = pair.as_ref() {
            self.note_remote_pair_selected(pair_ref)?;
        }
        redirect_target_for_pair(pair)
    }

    fn session_field(&self, store: &Store) -> anyhow::Result<SessionField> {
        let quality_model = store.active_quality_model()?.formal_version;
        let hierarchical_assets = match quality_model {
            QualityFormalVersion::LegacyIndependentV1 => HashMap::new(),
            QualityFormalVersion::HierarchicalGaussianV1 => store
                .corpus_asset_quality_caches(self.active.corpus_id, quality_model)?
                .into_iter()
                .filter_map(|(asset_id, cache)| {
                    HierarchicalAssetPosterior::decode(&cache.payload)
                        .map(|payload| (asset_id, payload))
                })
                .collect(),
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => HashMap::new(),
        };
        let hierarchical_session = match quality_model {
            QualityFormalVersion::LegacyIndependentV1 => None,
            QualityFormalVersion::HierarchicalGaussianV1 => store
                .session_quality_cache(self.active.session_id, quality_model)?
                .and_then(|cache| HierarchicalSessionPosterior::decode(&cache.payload)),
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => None,
        };
        let perturbative_assets = match quality_model {
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => store
                .corpus_asset_quality_caches(self.active.corpus_id, quality_model)?
                .into_iter()
                .filter_map(|(asset_id, cache)| {
                    PerturbativeAssetPosterior::decode(&cache.payload)
                        .map(|payload| (asset_id, payload))
                })
                .collect(),
            _ => HashMap::new(),
        };
        let perturbative_session = match quality_model {
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => store
                .session_quality_cache(self.active.session_id, quality_model)?
                .and_then(|cache| PerturbativeSessionPosterior::decode(&cache.payload)),
            _ => None,
        };
        let perturbative_hyper = store
            .load_perturbative_hyper_params(&store.active_quality_model()?)?
            .unwrap_or_default();
        Ok(SessionField {
            session: self.session_with_store(store)?,
            quality_model,
            exact_offsets: store.session_asset_offsets(self.active.session_id)?,
            hearted_assets: store.hearted_assets()?,
            subsource_lock: store.session_subsource_lock(self.active.session_id)?,
            embeddings: store
                .corpus_embeddings(self.active.corpus_id, self.embedder.model_name())?,
            embedding_head: store
                .session_embedding_head(self.active.session_id, self.embedder.model_name())?,
            hierarchical_assets,
            dominant_faces: store.dominant_local_face_identities(self.active.corpus_id)?,
            hierarchical_session,
            perturbative_assets,
            perturbative_session,
            perturbative_hyper,
        })
    }

    fn purge_duplicate_frontier(&self) {
        self.duplicate_frontier.write().take();
    }

    fn ensure_duplicate_frontier(&self, store: &Store) -> anyhow::Result<()> {
        if self.duplicate_frontier.read().is_some() {
            return Ok(());
        }
        let points = store.all_frontier_embeddings(self.embedder.model_name())?;
        let frontier = DuplicateFrontier {
            tree: VpTree::forge(points),
        };
        *self.duplicate_frontier.write() = Some(frontier);
        Ok(())
    }

    fn dedup_radius(&self) -> f32 {
        self.config.read().dedup_radius()
    }

    pub fn set_dedup_radius(&self, radius: f32) -> anyhow::Result<()> {
        let snapshot = {
            let mut config = self.config.write();
            config.shove_dedup_radius(radius);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)
    }

    pub fn startup_summary(&self) -> anyhow::Result<StartupSummary> {
        let store = self.store.lock();
        let visible_assets = visible_assets(&store, self.active.corpus_id)?.len();
        let embedded_assets = store
            .corpus_embeddings(self.active.corpus_id, self.embedder.model_name())?
            .len();
        Ok(StartupSummary {
            corpus_id: self.active.corpus_id,
            session_id: self.active.session_id,
            visible_assets,
            embedded_assets,
        })
    }
}

#[derive(Debug, Clone)]
pub enum RedirectTarget {
    ArenaRoot,
    ArenaPair {
        left: ArenaHandle,
        right: ArenaHandle,
    },
    FacemashRoot,
    FacemashPair {
        left_face_id: FaceId,
        right_face_id: FaceId,
    },
    ExploreRoot {
        map_mode: ExploreMapMode,
    },
    ExploreTriad {
        asset_a: AssetId,
        asset_b: AssetId,
        asset_c: AssetId,
        focus_id: Option<AssetId>,
        map_mode: ExploreMapMode,
    },
}

impl RedirectTarget {
    #[must_use]
    pub fn href(&self) -> String {
        match self {
            Self::ArenaRoot => "/arena".to_owned(),
            Self::ArenaPair { left, right } => format!("/arena/{}/{}", left.slug(), right.slug()),
            Self::FacemashRoot => "/facemash".to_owned(),
            Self::FacemashPair {
                left_face_id,
                right_face_id,
            } => format!("/facemash/{}/{}", left_face_id.0, right_face_id.0),
            Self::ExploreRoot { map_mode } => format!("/explore?mode={}", map_mode.as_str()),
            Self::ExploreTriad {
                asset_a,
                asset_b,
                asset_c,
                focus_id,
                map_mode,
            } => {
                let mut query = format!(
                    "mode={}&triad={},{},{}",
                    map_mode.as_str(),
                    asset_a.0,
                    asset_b.0,
                    asset_c.0
                );
                if let Some(focus_id) = focus_id {
                    query.push_str("&focus=");
                    query.push_str(&focus_id.0);
                }
                format!("/explore?{query}")
            }
        }
    }
}

fn surviving_local_anchor<'a>(
    rejected: &ArenaHandle,
    pair_left: &'a ArenaHandle,
    pair_right: &'a ArenaHandle,
) -> Option<&'a AssetId> {
    match (rejected, pair_left, pair_right) {
        (ArenaHandle::Remote(_), ArenaHandle::Local(asset_id), ArenaHandle::Remote(_))
        | (ArenaHandle::Remote(_), ArenaHandle::Remote(_), ArenaHandle::Local(asset_id)) => {
            Some(asset_id)
        }
        _ => None,
    }
}

pub type SharedAppState = Arc<AppState>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimePhase {
    Loading,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSnapshot {
    pub phase: RuntimePhase,
    pub message: Option<String>,
}

enum RuntimeStateInner {
    Loading,
    Ready(SharedAppState),
    Failed(String),
}

pub struct RuntimeState {
    inner: RwLock<RuntimeStateInner>,
}

impl RuntimeState {
    #[must_use]
    pub fn loading() -> Self {
        Self {
            inner: RwLock::new(RuntimeStateInner::Loading),
        }
    }

    pub fn install_ready(&self, state: SharedAppState) {
        *self.inner.write() = RuntimeStateInner::Ready(state);
    }

    pub fn install_failed(&self, message: String) {
        *self.inner.write() = RuntimeStateInner::Failed(message);
    }

    #[must_use]
    pub fn ready_app(&self) -> Option<SharedAppState> {
        match &*self.inner.read() {
            RuntimeStateInner::Ready(state) => Some(state.clone()),
            RuntimeStateInner::Loading | RuntimeStateInner::Failed(_) => None,
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> RuntimeSnapshot {
        match &*self.inner.read() {
            RuntimeStateInner::Loading => RuntimeSnapshot {
                phase: RuntimePhase::Loading,
                message: None,
            },
            RuntimeStateInner::Ready(_) => RuntimeSnapshot {
                phase: RuntimePhase::Ready,
                message: None,
            },
            RuntimeStateInner::Failed(message) => RuntimeSnapshot {
                phase: RuntimePhase::Failed,
                message: Some(message.clone()),
            },
        }
    }

    pub fn close_ready(&self) -> anyhow::Result<()> {
        let ready = match &*self.inner.read() {
            RuntimeStateInner::Ready(state) => Some(state.clone()),
            RuntimeStateInner::Loading | RuntimeStateInner::Failed(_) => None,
        };
        if let Some(state) = ready {
            state.close()?;
        }
        Ok(())
    }
}

pub type SharedRuntimeState = Arc<RuntimeState>;
