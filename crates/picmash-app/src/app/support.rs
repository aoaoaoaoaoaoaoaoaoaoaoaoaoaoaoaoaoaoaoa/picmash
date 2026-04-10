use super::*;
use crate::identity::VisualKey;
use std::collections::HashSet;

impl SessionField {
    pub(super) fn fallback_quality_summary(&self, asset: &AssetRecord) -> AssetQualitySummary {
        let sigma = crate::quality::legacy_cache_variance(asset.compare_count).sqrt();
        AssetQualitySummary {
            asset: PosteriorSummary {
                mean: self.utility(asset),
                sigma,
            },
            baseline: PosteriorSummary {
                mean: asset.alpha,
                sigma,
            },
            semantic: None,
            vibe: None,
            technical: None,
            face: hierarchical_face_summary(self.dominant_faces.get(&asset.id)),
        }
    }

    pub(super) fn quality_summary_or_fallback(&self, asset: &AssetRecord) -> AssetQualitySummary {
        self.quality_summary(asset)
            .unwrap_or_else(|| self.fallback_quality_summary(asset))
    }
}

pub(super) fn hierarchical_face_summary(
    identity: Option<&FaceIdentityRecord>,
) -> Option<PosteriorSummary> {
    identity.map(|identity| PosteriorSummary {
        mean: identity.beauty.mean,
        sigma: identity.beauty.sigma,
    })
}

pub(super) fn hierarchical_canonical_mean(
    asset: &HierarchicalAssetPosterior,
    face: Option<PosteriorSummary>,
) -> f32 {
    asset.baseline_mean
        + face.map_or(0.0, |face| {
            crate::quality::HIERARCHICAL_FACE_WEIGHT
                * crate::quality::hierarchical_face_latent_mean(crate::facemash::FaceBeauty::forge(
                    face.mean, face.sigma,
                ))
        })
        + asset.technical_mean.map_or(0.0, |technical| {
            crate::quality::HIERARCHICAL_TECH_WEIGHT * technical
        })
}

pub(super) fn hierarchical_canonical_variance(
    asset: &HierarchicalAssetPosterior,
    face: Option<PosteriorSummary>,
) -> f32 {
    asset.baseline_variance
        + face.map_or(0.0, |face| {
            crate::quality::HIERARCHICAL_FACE_WEIGHT.powi(2)
                * crate::quality::hierarchical_face_latent_variance(
                    crate::facemash::FaceBeauty::forge(face.mean, face.sigma),
                )
        })
        + asset.technical_variance.map_or(0.0, |variance| {
            crate::quality::HIERARCHICAL_TECH_WEIGHT.powi(2) * variance
        })
}

pub(super) fn hierarchical_session_utility_variance(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
    face: Option<PosteriorSummary>,
) -> f32 {
    hierarchical_canonical_variance(asset, face)
        + hierarchical_semantic_variance(asset, session)
        + hierarchical_vibe_variance(asset, session).unwrap_or_default()
}

pub(super) fn hierarchical_semantic_mean(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> f32 {
    dot(&asset.mood_loading_mean, &session.semantic_mood_mean)
}

pub(super) fn hierarchical_semantic_summary(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> PosteriorSummary {
    PosteriorSummary {
        mean: hierarchical_semantic_mean(asset, session),
        sigma: hierarchical_semantic_variance(asset, session)
            .max(0.0)
            .sqrt(),
    }
}

pub(super) fn hierarchical_semantic_variance(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> f32 {
    asset
        .mood_loading_variance
        .iter()
        .zip(session.semantic_mood_mean.iter())
        .map(|(variance, mood)| variance * mood * mood)
        .sum::<f32>()
        + session
            .semantic_mood_variance
            .iter()
            .zip(asset.mood_loading_mean.iter())
            .map(|(variance, loading)| variance * loading * loading)
            .sum::<f32>()
}

pub(super) fn hierarchical_vibe_mean(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> Option<f32> {
    asset.technical_mean.map(|_| {
        asset
            .vibe_mean
            .iter()
            .zip(session.vibe_mean.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>()
    })
}

pub(super) fn hierarchical_vibe_variance(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> Option<f32> {
    asset.technical_mean.map(|_| {
        asset
            .vibe_variance
            .iter()
            .zip(session.vibe_mean.iter())
            .map(|(variance, taste)| variance * taste * taste)
            .sum::<f32>()
            + session
                .vibe_variance
                .iter()
                .zip(asset.vibe_mean.iter())
                .map(|(variance, vibe)| variance * vibe * vibe)
                .sum::<f32>()
    })
}

pub(super) fn hierarchical_vibe_summary(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> Option<PosteriorSummary> {
    hierarchical_vibe_mean(asset, session).map(|mean| PosteriorSummary {
        mean,
        sigma: hierarchical_vibe_variance(asset, session)
            .unwrap_or_default()
            .max(0.0)
            .sqrt(),
    })
}

pub(super) fn hierarchical_total_summary(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
    face: Option<PosteriorSummary>,
) -> PosteriorSummary {
    PosteriorSummary {
        mean: hierarchical_canonical_mean(asset, face)
            + hierarchical_semantic_mean(asset, session)
            + hierarchical_vibe_mean(asset, session).unwrap_or_default(),
        sigma: hierarchical_session_utility_variance(asset, session, face)
            .max(0.0)
            .sqrt(),
    }
}

pub(super) fn remote_baseline_summary() -> PosteriorSummary {
    PosteriorSummary {
        mean: 0.0,
        sigma: crate::quality::HIERARCHICAL_BASELINE_PRIOR_VARIANCE.sqrt(),
    }
}

pub(super) fn remote_vibe_summary(
    descriptor: Option<&[f32; crate::quality_features::VIBE_DESCRIPTOR_DIM]>,
    session: &HierarchicalSessionPosterior,
) -> Option<PosteriorSummary> {
    descriptor.map(|descriptor| {
        let mean = descriptor
            .iter()
            .zip(session.vibe_mean.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        let variance = descriptor
            .iter()
            .zip(session.vibe_variance.iter())
            .map(|(value, variance)| value * value * variance)
            .sum::<f32>();
        PosteriorSummary {
            mean,
            sigma: variance.max(0.0).sqrt(),
        }
    })
}

pub(super) fn remote_total_summary(
    semantic: PosteriorSummary,
    baseline: PosteriorSummary,
    technical: Option<PosteriorSummary>,
    face: Option<PosteriorSummary>,
    vibe: Option<PosteriorSummary>,
) -> PosteriorSummary {
    let face_term = face.map_or((0.0, 0.0), |face| {
        let beauty = crate::facemash::FaceBeauty::forge(face.mean, face.sigma);
        (
            crate::quality::HIERARCHICAL_FACE_WEIGHT
                * crate::quality::hierarchical_face_latent_mean(beauty),
            crate::quality::HIERARCHICAL_FACE_WEIGHT.powi(2)
                * crate::quality::hierarchical_face_latent_variance(beauty),
        )
    });
    let technical_term = technical.map_or((0.0, 0.0), |technical| {
        (
            crate::quality::HIERARCHICAL_TECH_WEIGHT * technical.mean,
            crate::quality::HIERARCHICAL_TECH_WEIGHT.powi(2) * technical.sigma.powi(2),
        )
    });
    PosteriorSummary {
        mean: baseline.mean
            + semantic.mean
            + technical_term.0
            + face_term.0
            + vibe.map_or(0.0, |vibe| vibe.mean),
        sigma: (baseline.sigma.powi(2)
            + semantic.sigma.powi(2)
            + technical_term.1
            + face_term.1
            + vibe.map_or(0.0, |vibe| vibe.sigma.powi(2)))
        .max(0.0)
        .sqrt(),
    }
}

pub(super) fn perturbative_remote_total_summary(
    semantic: PosteriorSummary,
    baseline: PosteriorSummary,
    technical: Option<PosteriorSummary>,
    face: Option<PosteriorSummary>,
    vibe: Option<PosteriorSummary>,
    domain_label: crate::asset_domain::AssetDomainLabel,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> PosteriorSummary {
    let face_term = face.map_or((0.0, 0.0), |face| {
        let beauty = crate::facemash::FaceBeauty::forge(face.mean, face.sigma);
        (
            hyper.face_weight(domain_label) * crate::quality::hierarchical_face_latent_mean(beauty),
            hyper.face_weight(domain_label).powi(2)
                * crate::quality::hierarchical_face_latent_variance(beauty),
        )
    });
    let technical_term = technical.map_or((0.0, 0.0), |technical| {
        (
            hyper.technical_weight(domain_label) * technical.mean,
            hyper.technical_weight(domain_label).powi(2) * technical.sigma.powi(2),
        )
    });
    PosteriorSummary {
        mean: baseline.mean
            + semantic.mean
            + technical_term.0
            + face_term.0
            + vibe.map_or(0.0, |vibe| vibe.mean),
        sigma: (baseline.sigma.powi(2)
            + semantic.sigma.powi(2)
            + technical_term.1
            + face_term.1
            + vibe.map_or(0.0, |vibe| vibe.sigma.powi(2)))
        .max(0.0)
        .sqrt(),
    }
}

pub(super) fn hierarchical_asset_cache(
    asset: &HierarchicalAssetPosterior,
    face: Option<PosteriorSummary>,
) -> crate::quality::HierarchicalAssetQualityCacheV1 {
    crate::quality::HierarchicalAssetQualityCacheV1 {
        baseline_mean: asset.baseline_mean,
        baseline_variance: asset.baseline_variance,
        canonical_mean: hierarchical_canonical_mean(asset, face),
        canonical_variance: hierarchical_canonical_variance(asset, face),
        mood_loading_mean: asset.mood_loading_mean.to_vec(),
        mood_loading_variance: asset.mood_loading_variance.to_vec(),
        technical_mean: asset.technical_mean,
        technical_variance: asset.technical_variance,
        vibe_mean: asset.vibe_mean.to_vec(),
        vibe_variance: asset.vibe_variance.to_vec(),
    }
}

pub(super) fn hierarchical_session_cache(
    session: &HierarchicalSessionPosterior,
) -> crate::quality::HierarchicalSessionQualityCacheV1 {
    crate::quality::HierarchicalSessionQualityCacheV1 {
        semantic_mood_mean: session.semantic_mood_mean.to_vec(),
        semantic_mood_variance: session.semantic_mood_variance.to_vec(),
        vibe_mean: session.vibe_mean.to_vec(),
        vibe_variance: session.vibe_variance.to_vec(),
        frontier_mean: session.frontier_mean,
        frontier_variance: session.frontier_variance,
    }
}

pub(super) fn perturbative_semantic_mean(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
) -> f32 {
    let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
    importance
        * asset
            .semantic_basis()
            .iter()
            .zip(session.semantic_weight_mean().iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>()
}

pub(super) fn perturbative_semantic_variance(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
) -> f32 {
    let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
    let importance_slope =
        crate::quality::perturbative_importance_slope(session.importance_raw_mean);
    let centered = asset
        .semantic_basis()
        .iter()
        .zip(session.semantic_weight_mean().iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    let weight = asset
        .semantic_basis()
        .iter()
        .zip(session.semantic_weight_variance().iter())
        .map(|(value, variance)| value * value * variance)
        .sum::<f32>();
    (importance.powi(2) * weight
        + importance_slope.powi(2) * centered.powi(2) * session.importance_raw_variance)
        .max(0.0)
}

pub(super) fn perturbative_semantic_summary(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
) -> PosteriorSummary {
    PosteriorSummary {
        mean: perturbative_semantic_mean(asset, session),
        sigma: perturbative_semantic_variance(asset, session).sqrt(),
    }
}

pub(super) fn perturbative_canonical_mean(
    asset: &PerturbativeAssetPosterior,
    face: Option<PosteriorSummary>,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    asset.baseline_mean
        + face.map_or(0.0, |face| {
            hyper.face_weight(asset.domain_label)
                * crate::quality::hierarchical_face_latent_mean(crate::facemash::FaceBeauty::forge(
                    face.mean, face.sigma,
                ))
        })
        + asset.technical_mean.map_or(0.0, |technical| {
            hyper.technical_weight(asset.domain_label) * technical
        })
}

pub(super) fn perturbative_canonical_variance(
    asset: &PerturbativeAssetPosterior,
    face: Option<PosteriorSummary>,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    asset.baseline_variance
        + face.map_or(0.0, |face| {
            hyper.face_weight(asset.domain_label).powi(2)
                * crate::quality::hierarchical_face_latent_variance(
                    crate::facemash::FaceBeauty::forge(face.mean, face.sigma),
                )
        })
        + asset.technical_variance.map_or(0.0, |variance| {
            hyper.technical_weight(asset.domain_label).powi(2) * variance
        })
}

pub(super) fn perturbative_vibe_mean(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> Option<f32> {
    asset.technical_mean.map(|_| {
        let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
        importance
            * hyper.vibe_scale(asset.domain_label)
            * asset
                .vibe_basis()
                .iter()
                .zip(session.vibe_weight_mean().iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>()
    })
}

pub(super) fn perturbative_vibe_variance(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> Option<f32> {
    asset.technical_mean.map(|_| {
        let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
        let importance_slope =
            crate::quality::perturbative_importance_slope(session.importance_raw_mean);
        let vibe_scale = hyper.vibe_scale(asset.domain_label);
        let centered = asset
            .vibe_basis()
            .iter()
            .zip(session.vibe_weight_mean().iter())
            .map(|(lhs, rhs)| vibe_scale * lhs * rhs)
            .sum::<f32>();
        let weight = asset
            .vibe_basis()
            .iter()
            .zip(session.vibe_weight_variance().iter())
            .map(|(value, variance)| vibe_scale.powi(2) * value * value * variance)
            .sum::<f32>();
        (importance.powi(2) * weight
            + importance_slope.powi(2) * centered.powi(2) * session.importance_raw_variance)
            .max(0.0)
    })
}

pub(super) fn perturbative_vibe_summary(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> Option<PosteriorSummary> {
    perturbative_vibe_mean(asset, session, hyper).map(|mean| PosteriorSummary {
        mean,
        sigma: perturbative_vibe_variance(asset, session, hyper)
            .unwrap_or_default()
            .sqrt(),
    })
}

pub(super) fn perturbative_session_utility_variance(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
    hyper: crate::quality::PerturbativeHyperParamsV3,
    face: Option<PosteriorSummary>,
) -> f32 {
    perturbative_canonical_variance(asset, face, hyper)
        + crate::quality::perturbative_projection_variance(
            &asset.perturbation_basis,
            session,
            hyper,
        )
}

pub(super) fn perturbative_total_summary(
    asset: &PerturbativeAssetPosterior,
    session: &PerturbativeSessionPosterior,
    hyper: crate::quality::PerturbativeHyperParamsV3,
    face: Option<PosteriorSummary>,
) -> PosteriorSummary {
    PosteriorSummary {
        mean: perturbative_canonical_mean(asset, face, hyper)
            + crate::quality::perturbative_projection_mean(
                &asset.perturbation_basis,
                session,
                hyper,
            ),
        sigma: perturbative_session_utility_variance(asset, session, hyper, face)
            .max(0.0)
            .sqrt(),
    }
}

pub(super) fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm = embedding
        .iter()
        .map(|axis| axis * axis)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    embedding.iter().map(|axis| axis / norm).collect()
}

pub(super) fn choose_local_pair(
    assets: &[AssetRecord],
    field: &SessionField,
    store: &Store,
    explore: f32,
    excluded_visual_keys: &HashSet<VisualKey>,
) -> anyhow::Result<Option<ArenaPair>> {
    if assets.len() < 2 {
        return Ok(None);
    }

    let recent = (assets.len() > ARENA_RECENT_REPEAT_EXCLUDE)
        .then(|| store.recent_arena_asset_ids(field.session.id, ARENA_RECENT_REPEAT_EXCLUDE))
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .collect::<HashSet<_>>();

    let usable = assets
        .iter()
        .filter(|asset| !recent.contains(&asset.id))
        .filter(|asset| !asset_visual_key_excluded(asset, excluded_visual_keys))
        .cloned()
        .collect::<Vec<_>>();
    if usable.len() < 2 {
        return Ok(None);
    }
    let pool = usable;
    let mut rng = rng();
    let temperature = arena_sampling_temperature(explore);
    let uniform_mix = arena_uniform_mix(explore);
    let anchor_scores = pool
        .iter()
        .map(|asset| field.arena_anchor_score(asset, explore))
        .collect::<Vec<_>>();
    let Some(anchor_index) =
        sample_softmax_index(&mut rng, &anchor_scores, temperature, uniform_mix)
    else {
        return Ok(None);
    };
    let anchor = pool[anchor_index].clone();
    let anchor_utility = field.utility(&anchor);
    let opponent_pool = pool
        .iter()
        .enumerate()
        .filter(|(index, candidate)| *index != anchor_index && candidate.id != anchor.id)
        .map(|(_, asset)| asset.clone())
        .collect::<Vec<_>>();
    let opponent_scores = opponent_pool
        .iter()
        .map(|candidate| field.arena_opponent_score(&anchor, candidate, explore))
        .collect::<Vec<_>>();
    let Some(opponent_index) =
        sample_softmax_index(&mut rng, &opponent_scores, temperature, uniform_mix)
    else {
        return Ok(None);
    };
    let opponent = opponent_pool[opponent_index].clone();

    Ok(Some(ArenaPair {
        left: ArenaCard::Local(ArenaLocalCard {
            utility: anchor_utility,
            hearted: field.hearted(&anchor.id),
            quality: field.quality_summary_or_fallback(&anchor),
            domain: AssetDomainView::default(),
            asset: anchor,
        }),
        right: ArenaCard::Local(ArenaLocalCard {
            utility: field.utility(&opponent),
            hearted: field.hearted(&opponent.id),
            quality: field.quality_summary_or_fallback(&opponent),
            domain: AssetDomainView::default(),
            asset: opponent,
        }),
    }))
}

pub(super) fn choose_local_pair_against_anchor(
    anchor: &AssetRecord,
    assets: &[AssetRecord],
    field: &SessionField,
    store: &Store,
    explore: f32,
    excluded_visual_keys: &HashSet<VisualKey>,
) -> anyhow::Result<Option<ArenaPair>> {
    if assets.len() < 2 {
        return Ok(None);
    }

    let recent = (assets.len() > ARENA_RECENT_REPEAT_EXCLUDE)
        .then(|| store.recent_arena_asset_ids(field.session.id, ARENA_RECENT_REPEAT_EXCLUDE))
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .collect::<HashSet<_>>();
    let anchor_utility = field.utility(anchor);

    let usable = assets
        .iter()
        .filter(|asset| asset.id != anchor.id && !recent.contains(&asset.id))
        .filter(|asset| !asset_visual_key_excluded(asset, excluded_visual_keys))
        .cloned()
        .collect::<Vec<_>>();
    let pool = usable;
    if pool.is_empty() {
        return Ok(None);
    }
    let mut rng = rng();
    let opponent_scores = pool
        .iter()
        .map(|candidate| field.arena_opponent_score(anchor, candidate, explore))
        .collect::<Vec<_>>();
    let Some(opponent_index) = sample_softmax_index(
        &mut rng,
        &opponent_scores,
        arena_sampling_temperature(explore),
        arena_uniform_mix(explore),
    ) else {
        return Ok(None);
    };
    let opponent = pool[opponent_index].clone();

    Ok(Some(ArenaPair {
        left: ArenaCard::Local(ArenaLocalCard {
            utility: anchor_utility,
            hearted: field.hearted(&anchor.id),
            quality: field.quality_summary_or_fallback(anchor),
            domain: AssetDomainView::default(),
            asset: anchor.clone(),
        }),
        right: ArenaCard::Local(ArenaLocalCard {
            utility: field.utility(&opponent),
            hearted: field.hearted(&opponent.id),
            quality: field.quality_summary_or_fallback(&opponent),
            domain: AssetDomainView::default(),
            asset: opponent,
        }),
    }))
}

pub(super) fn projection_state(
    store: &Store,
    model_name: &str,
    embedding: Option<&[f32]>,
) -> anyhow::Result<Option<ProjectionModel>> {
    Ok(match (store.projection_model(model_name)?, embedding) {
        (Some(model), _) => Some(model),
        (None, Some(vector)) => Some(ProjectionModel::zero(model_name.to_owned(), vector.len())),
        (None, None) => None,
    })
}

pub(super) fn sanitize_score(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub(super) fn raw_distance_sq(
    raw_vectors: &HashMap<AssetId, Vec<f32>>,
    lhs: &AssetId,
    rhs: &AssetId,
) -> Option<f32> {
    let left = raw_vectors.get(lhs)?;
    let right = raw_vectors.get(rhs)?;
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(left_axis, right_axis)| {
                let delta = left_axis - right_axis;
                delta * delta
            })
            .sum(),
    )
}

pub(super) fn squared_similarity_gap(
    lhs: &[f32; SIMILARITY_DIM],
    rhs: &[f32; SIMILARITY_DIM],
) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

pub(super) fn redirect_target_for_pair(pair: Option<ArenaPair>) -> anyhow::Result<RedirectTarget> {
    Ok(match pair {
        Some(pair) => RedirectTarget::ArenaPair {
            left: pair.left.handle(),
            right: pair.right.handle(),
        },
        None => RedirectTarget::ArenaRoot,
    })
}

pub(super) fn visible_assets(
    store: &Store,
    corpus_id: CorpusId,
) -> anyhow::Result<Vec<AssetRecord>> {
    let mut assets = store.corpus_assets(corpus_id)?;
    assets.retain(|asset| !asset.hidden && asset.path.exists());
    Ok(assets)
}

pub(super) fn asset_visual_key_excluded(
    asset: &AssetRecord,
    excluded_visual_keys: &HashSet<VisualKey>,
) -> bool {
    asset
        .visual_key
        .as_ref()
        .is_some_and(|visual_key| excluded_visual_keys.contains(visual_key))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        path::PathBuf,
    };

    use super::*;

    #[test]
    fn perturbative_quality_summary_falls_back_without_cached_posteriors() {
        let field = SessionField {
            session: SessionRecord {
                id: SessionId(1),
                corpus_id: CorpusId(1),
                mood: [0.0; LATENT_DIM],
                frontier: 0.0,
                comparisons: 0,
                nudges: 0,
                hearts: 0,
            },
            quality_model: QualityFormalVersion::HierarchicalPerturbativeV3,
            exact_offsets: HashMap::new(),
            hearted_assets: HashSet::new(),
            subsource_lock: None,
            embeddings: HashMap::new().into(),
            embedding_head: None,
            hierarchical_assets: HashMap::new().into(),
            dominant_faces: HashMap::new().into(),
            hierarchical_session: None,
            perturbative_assets: HashMap::new().into(),
            perturbative_session: None,
            perturbative_hyper: PerturbativeHyperParamsV3::default(),
        };
        let asset = AssetRecord {
            id: AssetId("asset_test".to_owned()),
            path: PathBuf::from("/tmp/asset_test.png"),
            visual_key: None,
            width: 1024,
            height: 1024,
            alpha: 1.25,
            coords: [0.0; LATENT_DIM],
            rotation_quarters: 0,
            compare_count: 0,
            win_count: 0,
            heart_count: 0,
            is_hearted: false,
            hidden: false,
        };

        let quality = field.quality_summary_or_fallback(&asset);
        assert_eq!(quality.baseline.mean, 1.25);
        assert!(quality.technical.is_none());
        assert!(quality.face.is_none());
        assert!(quality.asset.sigma.is_finite());
    }
}
