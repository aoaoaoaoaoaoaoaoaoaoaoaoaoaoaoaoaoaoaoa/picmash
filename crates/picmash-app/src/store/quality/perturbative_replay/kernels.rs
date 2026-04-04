use super::*;

impl ExternalUnaryFeedback {
    pub(super) fn from_direction(direction: f32) -> anyhow::Result<Self> {
        match direction.signum() as i32 {
            -1 => Ok(Self::Reject),
            1 => Ok(Self::Accept),
            _ => bail!("perturbative unary direction must be ±1"),
        }
    }
}

pub(super) fn fit_vibe_standardization(
    local: &HashMap<AssetId, crate::quality_features::AssetQualityFeatures>,
    external: &HashMap<RemoteItemId, crate::quality_features::AssetQualityFeatures>,
    local_gate: &HashMap<AssetId, bool>,
    external_gate: &HashMap<RemoteItemId, bool>,
) -> VibeStandardization {
    let mut mean = [0.0; crate::quality_features::VIBE_DESCRIPTOR_DIM];
    let mut count = 0usize;
    for (asset_id, feature) in local {
        if !local_gate.get(asset_id).copied().unwrap_or(false) {
            continue;
        }
        for (slot, value) in mean.iter_mut().zip(feature.vibe.iter().copied()) {
            *slot += value;
        }
        count += 1;
    }
    for (item_id, feature) in external {
        if !external_gate.get(item_id).copied().unwrap_or(false) {
            continue;
        }
        for (slot, value) in mean.iter_mut().zip(feature.vibe.iter().copied()) {
            *slot += value;
        }
        count += 1;
    }
    if count > 0 {
        for slot in &mut mean {
            *slot /= count as f32;
        }
    }
    let mut variance = [0.0; crate::quality_features::VIBE_DESCRIPTOR_DIM];
    for (asset_id, feature) in local {
        if !local_gate.get(asset_id).copied().unwrap_or(false) {
            continue;
        }
        for ((slot, value), mu) in variance
            .iter_mut()
            .zip(feature.vibe.iter())
            .zip(mean.iter())
        {
            *slot += (*value - *mu).powi(2);
        }
    }
    for (item_id, feature) in external {
        if !external_gate.get(item_id).copied().unwrap_or(false) {
            continue;
        }
        for ((slot, value), mu) in variance
            .iter_mut()
            .zip(feature.vibe.iter())
            .zip(mean.iter())
        {
            *slot += (*value - *mu).powi(2);
        }
    }
    let mut scale = [1.0; crate::quality_features::VIBE_DESCRIPTOR_DIM];
    for (slot, var) in scale.iter_mut().zip(variance.iter()) {
        *slot = (var / count.max(1) as f32).sqrt().max(1e-3);
    }
    VibeStandardization { mean, scale }
}

pub(super) fn standardized_vibe(
    feature: &crate::quality_features::AssetQualityFeatures,
    gate_3d: bool,
    standardization: &VibeStandardization,
    _hyper: crate::quality::PerturbativeHyperParamsV3,
) -> [f32; crate::quality_features::VIBE_DESCRIPTOR_DIM] {
    if !gate_3d {
        return [0.0; crate::quality_features::VIBE_DESCRIPTOR_DIM];
    }
    let mut standardized = [0.0; crate::quality_features::VIBE_DESCRIPTOR_DIM];
    for axis in 0..crate::quality_features::VIBE_DESCRIPTOR_DIM {
        standardized[axis] =
            (feature.vibe[axis] - standardization.mean[axis]) / standardization.scale[axis];
    }
    standardized
}

pub(super) fn perturbation_basis(
    semantic: Option<[f32; crate::model::LATENT_DIM]>,
    vibe: [f32; crate::quality_features::VIBE_DESCRIPTOR_DIM],
) -> [f32; crate::quality::PERTURBATIVE_DIM] {
    let mut basis = [0.0; crate::quality::PERTURBATIVE_DIM];
    if let Some(semantic) = semantic {
        basis[..crate::model::LATENT_DIM].copy_from_slice(&semantic);
    }
    basis[crate::model::LATENT_DIM..].copy_from_slice(&vibe);
    basis
}

pub(super) fn perturbative_asset_canonical_mean(
    asset: &PerturbativeReplayAssetState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    asset.baseline_mean
        + asset.face.map_or(0.0, |face| {
            hyper.face_weight(asset.domain_label)
                * crate::quality::hierarchical_face_latent_mean(FaceBeauty::forge(
                    face.mean,
                    face.variance
                        .max(crate::quality::HIERARCHICAL_MIN_VARIANCE)
                        .sqrt(),
                ))
        })
        + asset.technical_mean.map_or(0.0, |technical| {
            hyper.technical_weight(asset.domain_label) * technical
        })
}

pub(super) fn perturbative_asset_canonical_variance(
    asset: &PerturbativeReplayAssetState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    asset.baseline_variance
        + asset.face.map_or(0.0, |face| {
            hyper.face_weight(asset.domain_label).powi(2)
                * crate::quality::hierarchical_face_latent_variance(FaceBeauty::forge(
                    face.mean,
                    face.variance
                        .max(crate::quality::HIERARCHICAL_MIN_VARIANCE)
                        .sqrt(),
                ))
        })
        + asset.technical_variance.map_or(0.0, |variance| {
            hyper.technical_weight(asset.domain_label).powi(2) * variance
        })
}

pub(super) fn perturbative_external_canonical_mean(
    item: &PerturbativeReplayExternalState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    item.baseline_mean
        + item.technical_mean.map_or(0.0, |technical| {
            hyper.technical_weight(item.domain_label) * technical
        })
}

pub(super) fn perturbative_external_canonical_variance(
    item: &PerturbativeReplayExternalState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    item.baseline_variance
        + item.technical_variance.map_or(0.0, |variance| {
            hyper.technical_weight(item.domain_label).powi(2) * variance
        })
}

pub(super) fn perturbative_session_utility_mean(
    asset: &PerturbativeReplayAssetState,
    session: &PerturbativeReplaySessionState,
    asset_id: &AssetId,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    let exact_offset = session
        .exact_offsets
        .get(asset_id)
        .copied()
        .unwrap_or_default();
    perturbative_asset_canonical_mean(asset, hyper)
        + crate::quality::perturbative_projection_mean(
            &asset.perturbation_basis,
            &perturbative_session_cache(session),
            hyper,
        )
        + exact_offset
        + crate::quality::legacy_heart_bias(&asset.asset, asset.asset.is_hearted)
}

pub(super) fn perturbative_session_utility_variance(
    asset: &PerturbativeReplayAssetState,
    session: &PerturbativeReplaySessionState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    perturbative_asset_canonical_variance(asset, hyper)
        + crate::quality::perturbative_projection_variance(
            &asset.perturbation_basis,
            &perturbative_session_cache(session),
            hyper,
        )
}

pub(super) fn perturbative_external_session_utility_mean(
    item: &PerturbativeReplayExternalState,
    session: &PerturbativeReplaySessionState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    perturbative_external_canonical_mean(item, hyper)
        + crate::quality::perturbative_projection_mean(
            &item.perturbation_basis,
            &perturbative_session_cache(session),
            hyper,
        )
}

pub(super) fn perturbative_external_session_utility_variance(
    item: &PerturbativeReplayExternalState,
    session: &PerturbativeReplaySessionState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    perturbative_external_canonical_variance(item, hyper)
        + crate::quality::perturbative_projection_variance(
            &item.perturbation_basis,
            &perturbative_session_cache(session),
            hyper,
        )
}

pub(super) fn perturbative_session_cache(
    session: &PerturbativeReplaySessionState,
) -> crate::quality::PerturbativeSessionPosterior {
    crate::quality::PerturbativeSessionPosterior {
        perturbation_weight_mean: session.perturbation_weight_mean,
        perturbation_weight_variance: session.perturbation_weight_variance,
        importance_raw_mean: session.importance_raw_mean,
        importance_raw_variance: session.importance_raw_variance,
        threshold_mean: session.threshold_mean,
        threshold_variance: session.threshold_variance,
    }
}

pub(super) fn apply_perturbative_session_projection_update(
    session: &mut PerturbativeReplaySessionState,
    basis: &[f32; crate::quality::PERTURBATIVE_DIM],
    outcome: f32,
    moments: crate::quality::GaussianMomentMatch,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) {
    let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
    for axis in 0..crate::quality::PERTURBATIVE_DIM {
        let scaled_basis = if axis < crate::model::LATENT_DIM {
            basis[axis]
        } else {
            hyper.vibe_scale_real * basis[axis]
        };
        crate::quality::diagonal_adf_update(
            &mut session.perturbation_weight_mean[axis],
            &mut session.perturbation_weight_variance[axis],
            importance * scaled_basis,
            outcome,
            moments,
        );
    }
    let centered = crate::quality::perturbative_projection_mean(
        basis,
        &perturbative_session_cache(session),
        hyper,
    ) / importance.max(1e-6);
    let importance_slope =
        crate::quality::perturbative_importance_slope(session.importance_raw_mean);
    crate::quality::diagonal_adf_update(
        &mut session.importance_raw_mean,
        &mut session.importance_raw_variance,
        importance_slope * centered,
        outcome,
        moments,
    );
}

pub(super) fn subtract_basis(
    lhs: &[f32; crate::quality::PERTURBATIVE_DIM],
    rhs: &[f32; crate::quality::PERTURBATIVE_DIM],
) -> [f32; crate::quality::PERTURBATIVE_DIM] {
    let mut out = [0.0; crate::quality::PERTURBATIVE_DIM];
    for axis in 0..crate::quality::PERTURBATIVE_DIM {
        out[axis] = lhs[axis] - rhs[axis];
    }
    out
}

pub(super) fn refresh_perturbative_asset_face_anchor(
    asset: &mut PerturbativeReplayAssetState,
    subjects: &HashMap<FaceIdentityId, HierarchicalSubjectState>,
) {
    asset.face = asset
        .face
        .map(|face| super::face_anchor_for_subject(face, subjects));
}

pub(super) fn sync_perturbative_asset_record(
    asset: &mut PerturbativeReplayAssetState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) {
    asset.asset.alpha = perturbative_asset_canonical_mean(asset, hyper);
    asset
        .asset
        .coords
        .copy_from_slice(&asset.perturbation_basis[..crate::model::LATENT_DIM]);
}

pub(super) fn sync_perturbative_session_record(session: &mut PerturbativeReplaySessionState) {
    let importance = crate::quality::perturbative_importance(session.importance_raw_mean);
    for axis in 0..crate::model::LATENT_DIM {
        session.session.mood[axis] = importance * session.perturbation_weight_mean[axis];
    }
    session.session.frontier = session.threshold_mean;
}

pub(super) fn gauge_fix_perturbative_technical(
    state: &mut PerturbativeReplayState,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) {
    let (sum, count) = state
        .assets
        .values()
        .filter_map(|asset| asset.technical_mean)
        .chain(
            state
                .external_items
                .values()
                .filter_map(|item| item.technical_mean),
        )
        .fold((0.0f32, 0usize), |(sum, count), mean| {
            (sum + mean, count + 1)
        });
    if count == 0 {
        return;
    }
    let center = sum / count as f32;
    if center.abs() <= 1e-6 {
        return;
    }
    let absorbed = hyper.technical_weight(AssetDomainLabel::Real) * center;
    for asset in state.assets.values_mut() {
        if let Some(technical) = asset.technical_mean.as_mut() {
            *technical -= center;
            asset.baseline_mean += absorbed;
            sync_perturbative_asset_record(asset, hyper);
        }
    }
    for item in state.external_items.values_mut() {
        if let Some(technical) = item.technical_mean.as_mut() {
            *technical -= center;
            item.baseline_mean += absorbed;
        }
    }
}

#[derive(Clone, Copy)]
enum PerturbativeHyperAxis {
    FaceReal,
    FaceAnime,
    TechReal,
    VibeReal,
}

impl PerturbativeHyperAxis {
    const ALL: [Self; 4] = [
        Self::FaceReal,
        Self::FaceAnime,
        Self::TechReal,
        Self::VibeReal,
    ];

    fn value(self, hyper: crate::quality::PerturbativeHyperParamsV3) -> f32 {
        match self {
            Self::FaceReal => hyper.face_weight_real,
            Self::FaceAnime => hyper.face_weight_anime,
            Self::TechReal => hyper.technical_weight_real,
            Self::VibeReal => hyper.vibe_scale_real,
        }
    }

    fn overwrite(
        self,
        hyper: crate::quality::PerturbativeHyperParamsV3,
        value: f32,
    ) -> crate::quality::PerturbativeHyperParamsV3 {
        let value = match self {
            Self::FaceReal | Self::FaceAnime => value.clamp(0.2, 2.4),
            Self::TechReal => value.clamp(0.1, 2.2),
            Self::VibeReal => value.clamp(0.05, 1.6),
        };
        match self {
            Self::FaceReal => crate::quality::PerturbativeHyperParamsV3 {
                face_weight_real: value,
                ..hyper
            },
            Self::FaceAnime => crate::quality::PerturbativeHyperParamsV3 {
                face_weight_anime: value,
                ..hyper
            },
            Self::TechReal => crate::quality::PerturbativeHyperParamsV3 {
                technical_weight_real: value,
                ..hyper
            },
            Self::VibeReal => crate::quality::PerturbativeHyperParamsV3 {
                vibe_scale_real: value,
                ..hyper
            },
        }
    }

    fn prior_sigma(self) -> f32 {
        match self {
            Self::FaceReal | Self::FaceAnime => 0.45,
            Self::TechReal => 0.35,
            Self::VibeReal => 0.4,
        }
    }
}

pub(super) fn fit_perturbative_hyper_params(
    state: &PerturbativeReplayState,
    replay: &ReplayEventStream,
    seed: crate::quality::PerturbativeHyperParamsV3,
) -> crate::quality::PerturbativeHyperParamsV3 {
    let mut current = seed;
    let mut best_score = perturbative_hyper_objective(state, replay, current);
    const MULTIPLIERS: [f32; 5] = [0.6, 0.8, 1.0, 1.25, 1.6];
    for _ in 0..3 {
        let mut improved = false;
        for axis in PerturbativeHyperAxis::ALL {
            let base = axis.value(current);
            let mut local_best = current;
            let mut local_best_score = best_score;
            for multiplier in MULTIPLIERS {
                let candidate = axis.overwrite(current, base * multiplier);
                let score = perturbative_hyper_objective(state, replay, candidate);
                if score > local_best_score + 1e-4 {
                    local_best = candidate;
                    local_best_score = score;
                    improved = true;
                }
            }
            current = local_best;
            best_score = local_best_score;
        }
        if !improved {
            break;
        }
    }
    current
}

fn perturbative_hyper_objective(
    state: &PerturbativeReplayState,
    replay: &ReplayEventStream,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> f32 {
    let data_score = replay
        .events
        .iter()
        .filter_map(|event| perturbative_event_log_likelihood(state, event, hyper))
        .sum::<f32>();
    data_score + perturbative_hyper_log_prior(hyper)
}

fn perturbative_hyper_log_prior(hyper: crate::quality::PerturbativeHyperParamsV3) -> f32 {
    let default = crate::quality::PerturbativeHyperParamsV3::default();
    PerturbativeHyperAxis::ALL
        .into_iter()
        .map(|axis| {
            let ratio = (axis.value(hyper) / axis.value(default)).max(1e-6).ln();
            -0.5 * (ratio / axis.prior_sigma()).powi(2)
        })
        .sum()
}

fn perturbative_event_log_likelihood(
    state: &PerturbativeReplayState,
    event: &LegacyReplayEvent,
    hyper: crate::quality::PerturbativeHyperParamsV3,
) -> Option<f32> {
    match event {
        LegacyReplayEvent::Comparison(event) => {
            let session = state.sessions.get(&event.session_id)?;
            let left = state.assets.get(&event.left_asset_id)?;
            let right = state.assets.get(&event.right_asset_id)?;
            let left_mean = perturbative_asset_canonical_mean(left, hyper)
                + crate::quality::perturbative_projection_mean(
                    &left.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                );
            let right_mean = perturbative_asset_canonical_mean(right, hyper)
                + crate::quality::perturbative_projection_mean(
                    &right.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                );
            let outcome = if event.winner_asset_id == event.left_asset_id {
                1.0
            } else if event.winner_asset_id == event.right_asset_id {
                -1.0
            } else {
                return None;
            };
            let delta_mean = left_mean - right_mean;
            let delta_variance = perturbative_asset_canonical_variance(left, hyper)
                + perturbative_asset_canonical_variance(right, hyper)
                + crate::quality::perturbative_projection_variance(
                    &subtract_basis(&left.perturbation_basis, &right.perturbation_basis),
                    &perturbative_session_cache(session),
                    hyper,
                );
            Some(gaussian_event_log_likelihood(
                delta_mean,
                delta_variance,
                outcome,
                crate::quality::HIERARCHICAL_DUEL_BETA,
            ))
        }
        LegacyReplayEvent::Nudge(event) => {
            let session = state.sessions.get(&event.session_id)?;
            let asset = state.assets.get(&event.asset_id)?;
            let feedback = ExternalUnaryFeedback::from_direction(event.direction).ok()?;
            let mean = perturbative_asset_canonical_mean(asset, hyper)
                + crate::quality::perturbative_projection_mean(
                    &asset.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                )
                - session.threshold_mean;
            let variance = perturbative_asset_canonical_variance(asset, hyper)
                + crate::quality::perturbative_projection_variance(
                    &asset.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                )
                + session.threshold_variance;
            Some(gaussian_event_log_likelihood(
                mean,
                variance,
                feedback.outcome(),
                feedback.beta(),
            ))
        }
        LegacyReplayEvent::Heart(event) => {
            let session = state.sessions.get(&event.session_id)?;
            let asset = state.assets.get(&event.asset_id)?;
            let outcome = if event.active { 1.0 } else { return None };
            let mean = perturbative_asset_canonical_mean(asset, hyper)
                + crate::quality::perturbative_projection_mean(
                    &asset.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                )
                - session.threshold_mean;
            let variance = perturbative_asset_canonical_variance(asset, hyper)
                + crate::quality::perturbative_projection_variance(
                    &asset.perturbation_basis,
                    &perturbative_session_cache(session),
                    hyper,
                )
                + session.threshold_variance;
            Some(gaussian_event_log_likelihood(
                mean,
                variance,
                outcome,
                crate::quality::HIERARCHICAL_UNARY_HEART_BETA,
            ))
        }
        LegacyReplayEvent::External(event) => match event.kind {
            ExternalEventKind::LocalWin | ExternalEventKind::RemoteWin => {
                let session = state.sessions.get(&event.session_id)?;
                let local_id = event.local_asset_id.as_ref()?;
                let local = state.assets.get(local_id)?;
                let remote = state.external_items.get(&event.item_id)?;
                let local_mean = perturbative_asset_canonical_mean(local, hyper)
                    + crate::quality::perturbative_projection_mean(
                        &local.perturbation_basis,
                        &perturbative_session_cache(session),
                        hyper,
                    );
                let remote_mean = perturbative_external_canonical_mean(remote, hyper)
                    + crate::quality::perturbative_projection_mean(
                        &remote.perturbation_basis,
                        &perturbative_session_cache(session),
                        hyper,
                    );
                let outcome = if matches!(event.kind, ExternalEventKind::LocalWin) {
                    1.0
                } else {
                    -1.0
                };
                let delta_mean = local_mean - remote_mean;
                let delta_variance = perturbative_asset_canonical_variance(local, hyper)
                    + perturbative_external_canonical_variance(remote, hyper)
                    + crate::quality::perturbative_projection_variance(
                        &subtract_basis(&local.perturbation_basis, &remote.perturbation_basis),
                        &perturbative_session_cache(session),
                        hyper,
                    );
                Some(gaussian_event_log_likelihood(
                    delta_mean,
                    delta_variance,
                    outcome,
                    crate::quality::HIERARCHICAL_DUEL_BETA,
                ))
            }
            ExternalEventKind::Rejected | ExternalEventKind::Kept | ExternalEventKind::Hearted => {
                let session = state.sessions.get(&event.session_id)?;
                let remote = state.external_items.get(&event.item_id)?;
                let feedback = match event.kind {
                    ExternalEventKind::Rejected => ExternalUnaryFeedback::Reject,
                    ExternalEventKind::Kept => ExternalUnaryFeedback::Accept,
                    ExternalEventKind::Hearted => ExternalUnaryFeedback::Heart,
                    _ => unreachable!(),
                };
                let mean = perturbative_external_canonical_mean(remote, hyper)
                    + crate::quality::perturbative_projection_mean(
                        &remote.perturbation_basis,
                        &perturbative_session_cache(session),
                        hyper,
                    )
                    - session.threshold_mean;
                let variance = perturbative_external_canonical_variance(remote, hyper)
                    + crate::quality::perturbative_projection_variance(
                        &remote.perturbation_basis,
                        &perturbative_session_cache(session),
                        hyper,
                    )
                    + session.threshold_variance;
                Some(gaussian_event_log_likelihood(
                    mean,
                    variance,
                    feedback.outcome(),
                    feedback.beta(),
                ))
            }
            ExternalEventKind::Selected
            | ExternalEventKind::StreamBlocked
            | ExternalEventKind::Imported => None,
        },
    }
}

fn gaussian_event_log_likelihood(mean: f32, variance: f32, outcome: f32, beta: f32) -> f32 {
    let scale = (variance.max(crate::quality::HIERARCHICAL_MIN_VARIANCE) + beta * beta).sqrt();
    let z = outcome * mean / scale.max(1e-6);
    crate::quality::standard_normal_cdf(z).clamp(1e-6, 1.0).ln()
}
