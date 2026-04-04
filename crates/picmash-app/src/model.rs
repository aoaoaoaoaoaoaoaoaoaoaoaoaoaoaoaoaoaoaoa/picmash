mod ids;
mod layout;
mod projection;
mod records;
mod scoring;
mod similarity;
mod views;

pub const LATENT_DIM: usize = 3;
pub const SIMILARITY_DIM: usize = 5;
pub const MAP_DIM: usize = 2;
pub const ARENA_RECENT_REPEAT_EXCLUDE: usize = 100;
pub const ORDINAL_BOOTSTRAP_TRIADS: usize = 24;

pub use self::{
    ids::{ArenaHandle, AssetId, CorpusId, FaceId, FaceIdentityId, RemoteItemId, SessionId},
    layout::{
        learned_reduce_points, pca_reduce_points, prepare_raw_layout_space, umap_reduce_points,
    },
    projection::{EmbeddingRecord, ProjectionModel, SessionEmbeddingHead},
    records::{
        AssetRecord, ExternalEventKind, RemoteCandidate, RemoteItemRecord, SessionRecord,
        SessionSubsourceLock,
    },
    scoring::{
        canonical_utility, certainty, dot, sample_softmax_index, session_focus, session_utility,
        sigmoid, subtract, weighted_choice_index,
    },
    similarity::{
        LinearSimilarityModel, OrdinalSimilarityModel, SimilarityChoice, SimilarityModel,
        SimilarityObservation,
    },
    views::{
        ArenaCard, ArenaLocalCard, ArenaPair, ArenaRemoteCard, ArenaView, AssetDomainView,
        AssetQualitySummary, BoardEntry, ClusterSatellite, DuplicateCluster, ExploreEntry,
        ExploreMapMode, ExploreNeighbor, ExplorePanels, ExploreSelection, ExploreTriad,
        ExploreView, ExternalArenaStatus, ExternalSourceOption, PosteriorSummary,
    },
};
