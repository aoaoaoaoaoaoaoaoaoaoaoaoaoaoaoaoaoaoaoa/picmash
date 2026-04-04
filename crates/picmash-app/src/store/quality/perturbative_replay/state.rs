use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct PerturbativeCenterSeed {
    pub(super) semantic: [f32; crate::model::LATENT_DIM],
    pub(super) vibe: [f32; crate::quality_features::VIBE_DESCRIPTOR_DIM],
    pub(super) threshold: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VibeStandardization {
    pub(super) mean: [f32; crate::quality_features::VIBE_DESCRIPTOR_DIM],
    pub(super) scale: [f32; crate::quality_features::VIBE_DESCRIPTOR_DIM],
}

#[derive(Debug, Clone)]
pub(super) struct PerturbativeReplayAssetState {
    pub(super) asset: AssetRecord,
    pub(super) domain_label: AssetDomainLabel,
    pub(super) baseline_mean: f32,
    pub(super) baseline_variance: f32,
    pub(super) perturbation_basis: [f32; crate::quality::PERTURBATIVE_DIM],
    pub(super) technical_mean: Option<f32>,
    pub(super) technical_variance: Option<f32>,
    pub(super) face: Option<SubjectBeautyAnchor>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PerturbativeReplayExternalState {
    pub(super) domain_label: AssetDomainLabel,
    pub(super) baseline_mean: f32,
    pub(super) baseline_variance: f32,
    pub(super) perturbation_basis: [f32; crate::quality::PERTURBATIVE_DIM],
    pub(super) technical_mean: Option<f32>,
    pub(super) technical_variance: Option<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct PerturbativeReplaySessionState {
    pub(super) session: SessionRecord,
    pub(super) perturbation_weight_mean: [f32; crate::quality::PERTURBATIVE_DIM],
    pub(super) perturbation_weight_variance: [f32; crate::quality::PERTURBATIVE_DIM],
    pub(super) importance_raw_mean: f32,
    pub(super) importance_raw_variance: f32,
    pub(super) threshold_mean: f32,
    pub(super) threshold_variance: f32,
    pub(super) exact_offsets: HashMap<AssetId, f32>,
    pub(super) hearted_assets: HashSet<AssetId>,
}

#[derive(Debug, Clone)]
pub(super) struct PerturbativeReplayState {
    pub(super) assets: HashMap<AssetId, PerturbativeReplayAssetState>,
    pub(super) external_items: HashMap<RemoteItemId, PerturbativeReplayExternalState>,
    pub(super) subjects: HashMap<FaceIdentityId, HierarchicalSubjectState>,
    pub(super) sessions: HashMap<SessionId, PerturbativeReplaySessionState>,
    pub(super) comparison_events: usize,
    pub(super) nudge_events: usize,
    pub(super) heart_events: usize,
    pub(super) external_events: usize,
    pub(super) max_comparison_id: i64,
    pub(super) max_nudge_id: i64,
    pub(super) max_heart_id: i64,
    pub(super) max_external_id: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedPerturbativeReplay {
    pub(super) frontier: ReplayFrontier,
    pub(super) model: crate::quality::QualityModelRecord,
    pub(super) state: PerturbativeReplayState,
    pub(super) subject_snapshot: Vec<(FaceIdentityId, FaceBeauty, u32)>,
    pub(super) asset_cache_payloads: HashMap<AssetId, String>,
    pub(super) session_cache_payloads: HashMap<SessionId, String>,
    pub(super) subject_cache_payloads: HashMap<FaceIdentityId, String>,
    pub(super) external_cache_payloads: HashMap<RemoteItemId, String>,
    pub(super) technical_head_payload: String,
    pub(super) hyper_payload: String,
    pub(super) stats: crate::quality::QualityReplayStats,
}
