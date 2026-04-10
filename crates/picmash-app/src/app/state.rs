use super::*;

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
pub(super) struct ActiveArena {
    pub(super) corpus_id: CorpusId,
    pub(super) session_id: SessionId,
}

#[derive(Debug, Clone)]
pub(super) struct ExploreLayoutCache {
    pub(super) asset_ids: Vec<AssetId>,
    pub(super) plots: Vec<[f32; MAP_DIM]>,
}

#[derive(Debug, Clone)]
pub(super) struct ExploreVectorCache {
    pub(super) asset_ids: Vec<AssetId>,
    pub(super) embeddings: HashMap<AssetId, Vec<f32>>,
    pub(super) raw_vectors: HashMap<AssetId, Vec<f32>>,
    pub(super) learned_corpus: Vec<Vec<f32>>,
    pub(super) latents: HashMap<AssetId, [f32; SIMILARITY_DIM]>,
    pub(super) model: SimilarityModel,
}

impl ExploreVectorCache {
    pub(super) fn refresh_learned_geometry(&mut self) {
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

    pub(super) fn embedding(&self, asset_id: &AssetId) -> Option<&[f32]> {
        self.embeddings.get(asset_id).map(Vec::as_slice)
    }

    pub(super) fn learned_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        Some(
            self.model
                .distance_sq(lhs, self.embedding(lhs)?, rhs, self.embedding(rhs)?),
        )
    }

    pub(super) fn nearest_neighbor_ids(
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

    pub(super) fn raw_layout_corpus(&self) -> Vec<Vec<f32>> {
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
pub(super) struct ScoredRemoteCandidate {
    pub(super) candidate: crate::model::RemoteCandidate,
    pub(super) utility: f32,
    pub(super) quality: AssetQualitySummary,
    pub(super) selection_score: f32,
}

#[derive(Debug, Clone)]
pub(super) struct SessionField {
    pub(super) session: SessionRecord,
    pub(super) quality_model: QualityFormalVersion,
    pub(super) exact_offsets: HashMap<AssetId, f32>,
    pub(super) hearted_assets: HashSet<AssetId>,
    pub(super) subsource_lock: Option<crate::model::SessionSubsourceLock>,
    pub(super) embeddings: Arc<HashMap<AssetId, Vec<f32>>>,
    pub(super) embedding_head: Option<SessionEmbeddingHead>,
    pub(super) hierarchical_assets: Arc<HashMap<AssetId, HierarchicalAssetPosterior>>,
    pub(super) dominant_faces: Arc<HashMap<AssetId, FaceIdentityRecord>>,
    pub(super) hierarchical_session: Option<HierarchicalSessionPosterior>,
    pub(super) perturbative_assets: Arc<HashMap<AssetId, PerturbativeAssetPosterior>>,
    pub(super) perturbative_session: Option<PerturbativeSessionPosterior>,
    pub(super) perturbative_hyper: PerturbativeHyperParamsV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LockExhaustionPolicy {
    ClearAndRetry,
    PreserveAndStop,
}

impl SessionField {
    pub(super) fn exact_offset(&self, asset_id: &AssetId) -> f32 {
        self.exact_offsets.get(asset_id).copied().unwrap_or(0.0)
    }

    pub(super) fn hearted(&self, asset_id: &AssetId) -> bool {
        self.hearted_assets.contains(asset_id)
    }

    pub(super) fn heart_bias(&self, asset: &AssetRecord) -> f32 {
        asset.heart_count as f32 * LEGACY_HEART_GLOBAL_BOOST
            + if asset.is_hearted {
                LEGACY_HEART_SESSION_BOOST
            } else {
                0.0
            }
    }

    pub(super) fn embedding(&self, asset_id: &AssetId) -> Option<&[f32]> {
        self.embeddings.get(asset_id).map(Vec::as_slice)
    }

    pub(super) fn residual_score_for_embedding(&self, embedding: &[f32]) -> f32 {
        self.embedding_head
            .as_ref()
            .map_or(0.0, |head| head.score(embedding))
    }

    pub(super) fn residual_score(&self, asset_id: &AssetId) -> f32 {
        match (&self.embedding_head, self.embedding(asset_id)) {
            (Some(head), Some(embedding)) => head.score(embedding),
            _ => 0.0,
        }
    }

    pub(super) fn utility(&self, asset: &AssetRecord) -> f32 {
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

    pub(super) fn focus(&self, asset: &AssetRecord) -> f32 {
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

    pub(super) fn quality_posterior(&self, asset: &AssetRecord) -> PosteriorSummary {
        self.quality_summary(asset).map_or(
            PosteriorSummary {
                mean: self.utility(asset),
                sigma: crate::quality::legacy_cache_variance(asset.compare_count).sqrt(),
            },
            |summary| summary.asset,
        )
    }

    pub(super) fn arena_anchor_score(&self, asset: &AssetRecord, explore: f32) -> f32 {
        let posterior = self.quality_posterior(asset);
        let frontier_pull = sigmoid(self.focus(asset));
        let scarcity = 1.0 - certainty(asset.compare_count);
        posterior.mean
            + explore
                * (posterior.sigma * ARENA_EXPLORE_SIGMA_WEIGHT
                    + frontier_pull * ARENA_EXPLORE_FRONTIER_WEIGHT
                    + scarcity * ARENA_EXPLORE_SCARCITY_WEIGHT)
    }

    pub(super) fn arena_opponent_score(
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

    pub(super) fn sampling_pull(&self, asset: &AssetRecord) -> f32 {
        let uncertainty = self
            .quality_summary(asset)
            .map(|summary| summary.asset.sigma / (1.0 + summary.asset.sigma))
            .unwrap_or_else(|| 1.0 - certainty(asset.compare_count));
        let frontier_pull = sigmoid(self.focus(asset));
        let weighted = 0.18 + uncertainty * 1.15 + frontier_pull * 0.95;
        sanitize_score(weighted)
    }

    pub(super) fn quality_summary(&self, asset: &AssetRecord) -> Option<AssetQualitySummary> {
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

    pub(super) fn board_entry(&self, asset: AssetRecord) -> BoardEntry {
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
pub(super) struct SimilarityField {
    pub(super) points: Vec<ExploreEntry>,
    pub(super) raw_vectors: HashMap<AssetId, Vec<f32>>,
    pub(super) latents: HashMap<AssetId, [f32; SIMILARITY_DIM]>,
    pub(super) point_index: HashMap<AssetId, usize>,
}

impl SimilarityField {
    pub(super) fn entry(&self, asset_id: &AssetId) -> Option<ExploreEntry> {
        self.point_index
            .get(asset_id)
            .and_then(|index| self.points.get(*index))
            .cloned()
    }

    pub(super) fn latent(&self, asset_id: &AssetId) -> Option<&[f32; SIMILARITY_DIM]> {
        self.latents.get(asset_id)
    }

    pub(super) fn learned_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        Some(squared_similarity_gap(self.latent(lhs)?, self.latent(rhs)?))
    }

    pub(super) fn raw_distance_sq(&self, lhs: &AssetId, rhs: &AssetId) -> Option<f32> {
        raw_distance_sq(&self.raw_vectors, lhs, rhs)
    }

    pub(super) fn distance_sq(
        &self,
        mode: ExploreMapMode,
        lhs: &AssetId,
        rhs: &AssetId,
    ) -> Option<f32> {
        match mode {
            ExploreMapMode::Raw => self.raw_distance_sq(lhs, rhs),
            ExploreMapMode::Learned => self.learned_distance_sq(lhs, rhs),
        }
    }

    pub(super) fn nearest_neighbors(
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

    pub(super) fn selection(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct FacemashPairKey {
    left: FaceIdentityId,
    right: FaceIdentityId,
}

impl FacemashPairKey {
    pub(super) fn forge(left: FaceIdentityId, right: FaceIdentityId) -> Self {
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
pub(super) struct DuplicateFrontier {
    pub(super) tree: VpTree<RemoteItemId>,
}

#[derive(Debug, Clone)]
pub(super) struct ConfigReloadState {
    live_digest: String,
    warned_rejected_digest: Option<String>,
}

impl ConfigReloadState {
    pub(super) fn forge(live_digest: String) -> Self {
        Self {
            live_digest,
            warned_rejected_digest: None,
        }
    }

    pub(super) fn is_live_digest(&self, digest: &str) -> bool {
        self.live_digest == digest
    }

    pub(super) fn should_warn_rejected(&self, digest: &str) -> bool {
        self.warned_rejected_digest.as_deref() != Some(digest)
    }

    pub(super) fn note_rejected(&mut self, digest: String) {
        self.warned_rejected_digest = Some(digest);
    }

    pub(super) fn note_applied(&mut self, digest: String) {
        self.live_digest = digest;
        self.warned_rejected_digest = None;
    }
}
