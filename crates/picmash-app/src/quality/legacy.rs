use anyhow::bail;

use super::{
    AssetQualityCachePayload, LEGACY_HEART_GLOBAL_BOOST, LEGACY_HEART_SESSION_BOOST,
    LEGACY_L2_ALPHA, LEGACY_L2_COORD, LEGACY_L2_MOOD, LEGACY_LR_NUDGE_ALPHA, LEGACY_LR_NUDGE_COORD,
    LEGACY_LR_NUDGE_FRONTIER, LEGACY_LR_NUDGE_HEAD, LEGACY_LR_NUDGE_MOOD, LEGACY_LR_NUDGE_OFFSET,
    LEGACY_LR_NUDGE_PROJECTION, LEGACY_PRIOR_PULL, LegacyAssetQualityCacheV1,
    LegacySessionQualityCacheV1, LegacySubjectQualityCacheV1, SessionQualityCachePayload,
    SubjectQualityCachePayload,
};
use crate::{
    facemash::FaceBeauty,
    model::{AssetRecord, LATENT_DIM, ProjectionModel, SessionRecord},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyUnaryFeedback {
    Less,
    More,
}

#[derive(Debug, Clone, Copy)]
pub struct LegacyUnaryTuning {
    pub signal: f32,
    pub alpha_rate: f32,
    pub coord_rate: f32,
    pub mood_rate: f32,
    pub frontier_rate: f32,
    pub offset_rate: f32,
    pub head_rate: f32,
    pub projection_rate: f32,
}

impl LegacyUnaryFeedback {
    pub fn from_direction(direction: f32) -> anyhow::Result<Self> {
        match direction.signum() as i32 {
            -1 => Ok(Self::Less),
            1 => Ok(Self::More),
            _ => bail!("legacy unary direction must be ±1"),
        }
    }

    #[must_use]
    pub const fn direction(self) -> f32 {
        match self {
            Self::Less => -1.0,
            Self::More => 1.0,
        }
    }

    #[must_use]
    pub fn tuning(self, utility: f32, frontier: f32) -> LegacyUnaryTuning {
        let signed_direction = self.direction();
        let margin = signed_direction * (utility - frontier);
        LegacyUnaryTuning {
            signal: signed_direction * (1.0 - crate::model::sigmoid(margin)),
            alpha_rate: LEGACY_LR_NUDGE_ALPHA,
            coord_rate: LEGACY_LR_NUDGE_COORD,
            mood_rate: LEGACY_LR_NUDGE_MOOD,
            frontier_rate: LEGACY_LR_NUDGE_FRONTIER,
            offset_rate: LEGACY_LR_NUDGE_OFFSET,
            head_rate: LEGACY_LR_NUDGE_HEAD,
            projection_rate: LEGACY_LR_NUDGE_PROJECTION,
        }
    }
}

#[must_use]
pub fn legacy_projection_prior(
    projection: Option<&ProjectionModel>,
    embedding: Option<&[f32]>,
) -> [f32; LATENT_DIM] {
    projection
        .and_then(|model| embedding.map(|vector| model.predict(vector)))
        .unwrap_or([0.0; LATENT_DIM])
}

pub fn legacy_batter_asset(
    alpha: &mut f32,
    coords: &mut [f32; LATENT_DIM],
    mood: &[f32; LATENT_DIM],
    prior: &[f32; LATENT_DIM],
    signal: f32,
    alpha_rate: f32,
    coord_rate: f32,
) {
    *alpha += alpha_rate * (signal - LEGACY_L2_ALPHA * *alpha);
    for ((coord, prior_axis), mood_axis) in coords
        .iter_mut()
        .zip(prior.iter().copied())
        .zip(mood.iter().copied())
    {
        *coord += coord_rate
            * (signal * mood_axis
                - LEGACY_PRIOR_PULL * (*coord - prior_axis)
                - LEGACY_L2_COORD * *coord);
    }
}

pub fn legacy_shove_mood(
    mood: &mut [f32; LATENT_DIM],
    coords: &[f32; LATENT_DIM],
    signal: f32,
    learning_rate: f32,
) {
    for (mood_axis, coord) in mood.iter_mut().zip(coords.iter().copied()) {
        *mood_axis += learning_rate * (signal * coord - LEGACY_L2_MOOD * *mood_axis);
    }
}

#[must_use]
pub fn legacy_heart_bias(asset: &AssetRecord, is_hearted: bool) -> f32 {
    asset.heart_count as f32 * LEGACY_HEART_GLOBAL_BOOST
        + if is_hearted {
            LEGACY_HEART_SESSION_BOOST
        } else {
            0.0
        }
}

#[must_use]
pub fn legacy_cache_variance(observations: u32) -> f32 {
    1.0 / (1.0 + observations as f32)
}

#[must_use]
pub fn legacy_asset_quality_payload(asset: &AssetRecord) -> AssetQualityCachePayload {
    let variance = legacy_cache_variance(asset.compare_count);
    AssetQualityCachePayload::LegacyIndependentV1(LegacyAssetQualityCacheV1 {
        alpha_mean: asset.alpha,
        alpha_variance: variance,
        coords_mean: asset.coords,
        coords_variance: [variance; LATENT_DIM],
        compare_count: asset.compare_count,
    })
}

#[must_use]
pub fn legacy_session_quality_payload(session: &SessionRecord) -> SessionQualityCachePayload {
    let observations = session
        .comparisons
        .saturating_add(session.nudges)
        .saturating_add(session.hearts);
    let variance = legacy_cache_variance(observations);
    SessionQualityCachePayload::LegacyIndependentV1(LegacySessionQualityCacheV1 {
        mood_mean: session.mood,
        mood_variance: [variance; LATENT_DIM],
        frontier_mean: session.frontier,
        frontier_variance: variance,
    })
}

#[must_use]
pub fn legacy_subject_quality_payload(
    beauty: FaceBeauty,
    duel_count: u32,
) -> SubjectQualityCachePayload {
    SubjectQualityCachePayload::LegacyIndependentV1(LegacySubjectQualityCacheV1 {
        beauty_mean: beauty.mean,
        beauty_sigma: beauty.sigma,
        duel_count,
    })
}
