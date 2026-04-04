use std::{fmt, str::FromStr};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use super::constants::{
    DEFAULT_QUALITY_PRIOR_FAMILY, DEFAULT_QUALITY_PRIOR_REVISION, PERTURBATIVE_DIM,
    PERTURBATIVE_FACE_WEIGHT_ANIME_DEFAULT, PERTURBATIVE_FACE_WEIGHT_REAL_DEFAULT,
    PERTURBATIVE_TECH_WEIGHT_REAL_DEFAULT, PERTURBATIVE_VIBE_SCALE_REAL_DEFAULT,
};
use crate::{
    asset_domain::AssetDomainLabel,
    model::{AssetId, FaceIdentityId, LATENT_DIM, RemoteItemId, SessionId},
    quality_features::VIBE_DESCRIPTOR_DIM,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityFormalVersion {
    LegacyIndependentV1,
    HierarchicalGaussianV1,
    HierarchicalPerturbativeV2,
    HierarchicalPerturbativeV3,
}

impl QualityFormalVersion {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyIndependentV1 => "legacy_independent_v1",
            Self::HierarchicalGaussianV1 => "hierarchical_gaussian_v1",
            Self::HierarchicalPerturbativeV2 => "hierarchical_perturbative_v2",
            Self::HierarchicalPerturbativeV3 => "hierarchical_perturbative_v3",
        }
    }
}

impl fmt::Display for QualityFormalVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for QualityFormalVersion {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "legacy_independent_v1" => Ok(Self::LegacyIndependentV1),
            "hierarchical_gaussian_v1" => Ok(Self::HierarchicalGaussianV1),
            "hierarchical_perturbative_v2" => Ok(Self::HierarchicalPerturbativeV2),
            "hierarchical_perturbative_v3" => Ok(Self::HierarchicalPerturbativeV3),
            _ => bail!("unknown quality formal version `{value}`"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QualityPriorFamily(String);

impl QualityPriorFamily {
    pub fn forge(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            bail!("quality prior family must not be blank");
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for QualityPriorFamily {
    fn default() -> Self {
        Self(DEFAULT_QUALITY_PRIOR_FAMILY.to_owned())
    }
}

impl fmt::Display for QualityPriorFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QualityPriorRevision(String);

impl QualityPriorRevision {
    pub fn forge(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            bail!("quality prior revision must not be blank");
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for QualityPriorRevision {
    fn default() -> Self {
        Self(DEFAULT_QUALITY_PRIOR_REVISION.to_owned())
    }
}

impl fmt::Display for QualityPriorRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityModelId {
    pub formal_version: QualityFormalVersion,
}

impl QualityModelId {
    #[must_use]
    pub const fn as_version(self) -> QualityFormalVersion {
        self.formal_version
    }
}

pub const CURRENT_RUNTIME_QUALITY_MODEL: QualityModelId = QualityModelId {
    formal_version: QualityFormalVersion::HierarchicalPerturbativeV3,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityModelRecord {
    pub formal_version: QualityFormalVersion,
    pub prior_family: QualityPriorFamily,
    pub prior_revision: QualityPriorRevision,
    pub updated_at: OffsetDateTime,
}

impl QualityModelRecord {
    #[must_use]
    pub fn runtime_default(now: OffsetDateTime) -> Self {
        Self {
            formal_version: CURRENT_RUNTIME_QUALITY_MODEL.formal_version,
            prior_family: QualityPriorFamily::default(),
            prior_revision: QualityPriorRevision::default(),
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyAssetQualityCacheV1 {
    pub alpha_mean: f32,
    pub alpha_variance: f32,
    pub coords_mean: [f32; LATENT_DIM],
    pub coords_variance: [f32; LATENT_DIM],
    pub compare_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAssetQualityCacheV1 {
    pub baseline_mean: f32,
    pub baseline_variance: f32,
    pub canonical_mean: f32,
    pub canonical_variance: f32,
    pub mood_loading_mean: Vec<f32>,
    pub mood_loading_variance: Vec<f32>,
    pub technical_mean: Option<f32>,
    pub technical_variance: Option<f32>,
    pub vibe_mean: Vec<f32>,
    pub vibe_variance: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbativeAssetQualityCacheV2 {
    pub baseline_mean: f32,
    pub baseline_variance: f32,
    pub canonical_mean: f32,
    pub canonical_variance: f32,
    pub perturbation_basis: Vec<f32>,
    pub technical_mean: Option<f32>,
    pub technical_variance: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbativeAssetQualityCacheV3 {
    pub baseline_mean: f32,
    pub baseline_variance: f32,
    pub canonical_mean: f32,
    pub canonical_variance: f32,
    pub perturbation_basis: Vec<f32>,
    pub technical_mean: Option<f32>,
    pub technical_variance: Option<f32>,
    pub domain_label: AssetDomainLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "formal_version", rename_all = "snake_case")]
pub enum AssetQualityCachePayload {
    LegacyIndependentV1(LegacyAssetQualityCacheV1),
    HierarchicalGaussianV1(HierarchicalAssetQualityCacheV1),
    HierarchicalPerturbativeV2(PerturbativeAssetQualityCacheV2),
    HierarchicalPerturbativeV3(PerturbativeAssetQualityCacheV3),
}

impl AssetQualityCachePayload {
    #[must_use]
    pub const fn formal_version(&self) -> QualityFormalVersion {
        match self {
            Self::LegacyIndependentV1(_) => QualityFormalVersion::LegacyIndependentV1,
            Self::HierarchicalGaussianV1(_) => QualityFormalVersion::HierarchicalGaussianV1,
            Self::HierarchicalPerturbativeV2(_) => QualityFormalVersion::HierarchicalPerturbativeV2,
            Self::HierarchicalPerturbativeV3(_) => QualityFormalVersion::HierarchicalPerturbativeV3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySessionQualityCacheV1 {
    pub mood_mean: [f32; LATENT_DIM],
    pub mood_variance: [f32; LATENT_DIM],
    pub frontier_mean: f32,
    pub frontier_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSessionQualityCacheV1 {
    pub semantic_mood_mean: Vec<f32>,
    pub semantic_mood_variance: Vec<f32>,
    pub vibe_mean: Vec<f32>,
    pub vibe_variance: Vec<f32>,
    pub frontier_mean: f32,
    pub frontier_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbativeSessionQualityCacheV2 {
    pub perturbation_weight_mean: Vec<f32>,
    pub perturbation_weight_variance: Vec<f32>,
    pub importance_raw_mean: f32,
    pub importance_raw_variance: f32,
    pub threshold_mean: f32,
    pub threshold_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbativeSessionQualityCacheV3 {
    pub perturbation_weight_mean: Vec<f32>,
    pub perturbation_weight_variance: Vec<f32>,
    pub importance_raw_mean: f32,
    pub importance_raw_variance: f32,
    pub threshold_mean: f32,
    pub threshold_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "formal_version", rename_all = "snake_case")]
pub enum SessionQualityCachePayload {
    LegacyIndependentV1(LegacySessionQualityCacheV1),
    HierarchicalGaussianV1(HierarchicalSessionQualityCacheV1),
    HierarchicalPerturbativeV2(PerturbativeSessionQualityCacheV2),
    HierarchicalPerturbativeV3(PerturbativeSessionQualityCacheV3),
}

impl SessionQualityCachePayload {
    #[must_use]
    pub const fn formal_version(&self) -> QualityFormalVersion {
        match self {
            Self::LegacyIndependentV1(_) => QualityFormalVersion::LegacyIndependentV1,
            Self::HierarchicalGaussianV1(_) => QualityFormalVersion::HierarchicalGaussianV1,
            Self::HierarchicalPerturbativeV2(_) => QualityFormalVersion::HierarchicalPerturbativeV2,
            Self::HierarchicalPerturbativeV3(_) => QualityFormalVersion::HierarchicalPerturbativeV3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySubjectQualityCacheV1 {
    pub beauty_mean: f32,
    pub beauty_sigma: f32,
    pub duel_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSubjectQualityCacheV1 {
    pub beauty_mean: f32,
    pub beauty_variance: f32,
    pub duel_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "formal_version", rename_all = "snake_case")]
pub enum SubjectQualityCachePayload {
    LegacyIndependentV1(LegacySubjectQualityCacheV1),
    HierarchicalGaussianV1(HierarchicalSubjectQualityCacheV1),
    HierarchicalPerturbativeV2(HierarchicalSubjectQualityCacheV1),
    HierarchicalPerturbativeV3(HierarchicalSubjectQualityCacheV1),
}

impl SubjectQualityCachePayload {
    #[must_use]
    pub const fn formal_version(&self) -> QualityFormalVersion {
        match self {
            Self::LegacyIndependentV1(_) => QualityFormalVersion::LegacyIndependentV1,
            Self::HierarchicalGaussianV1(_) => QualityFormalVersion::HierarchicalGaussianV1,
            Self::HierarchicalPerturbativeV2(_) => QualityFormalVersion::HierarchicalPerturbativeV2,
            Self::HierarchicalPerturbativeV3(_) => QualityFormalVersion::HierarchicalPerturbativeV3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoredAssetQualityCache {
    pub asset_id: AssetId,
    pub payload: AssetQualityCachePayload,
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone)]
pub struct StoredSessionQualityCache {
    pub session_id: SessionId,
    pub payload: SessionQualityCachePayload,
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone)]
pub struct StoredSubjectQualityCache {
    pub identity_id: FaceIdentityId,
    pub payload: SubjectQualityCachePayload,
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone)]
pub struct StoredExternalItemQualityCache {
    pub item_id: RemoteItemId,
    pub payload: AssetQualityCachePayload,
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone, Copy)]
pub struct HierarchicalAssetPosterior {
    pub baseline_mean: f32,
    pub baseline_variance: f32,
    pub canonical_mean: f32,
    pub canonical_variance: f32,
    pub mood_loading_mean: [f32; LATENT_DIM],
    pub mood_loading_variance: [f32; LATENT_DIM],
    pub technical_mean: Option<f32>,
    pub technical_variance: Option<f32>,
    pub vibe_mean: [f32; VIBE_DESCRIPTOR_DIM],
    pub vibe_variance: [f32; VIBE_DESCRIPTOR_DIM],
}

impl HierarchicalAssetPosterior {
    pub fn decode(payload: &AssetQualityCachePayload) -> Option<Self> {
        let AssetQualityCachePayload::HierarchicalGaussianV1(payload) = payload else {
            return None;
        };
        Some(Self {
            baseline_mean: payload.baseline_mean,
            baseline_variance: payload.baseline_variance,
            canonical_mean: payload.canonical_mean,
            canonical_variance: payload.canonical_variance,
            mood_loading_mean: payload.mood_loading_mean.as_slice().try_into().ok()?,
            mood_loading_variance: payload.mood_loading_variance.as_slice().try_into().ok()?,
            technical_mean: payload.technical_mean,
            technical_variance: payload.technical_variance,
            vibe_mean: payload.vibe_mean.as_slice().try_into().ok()?,
            vibe_variance: payload.vibe_variance.as_slice().try_into().ok()?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HierarchicalSessionPosterior {
    pub semantic_mood_mean: [f32; LATENT_DIM],
    pub semantic_mood_variance: [f32; LATENT_DIM],
    pub vibe_mean: [f32; VIBE_DESCRIPTOR_DIM],
    pub vibe_variance: [f32; VIBE_DESCRIPTOR_DIM],
    pub frontier_mean: f32,
    pub frontier_variance: f32,
}

impl HierarchicalSessionPosterior {
    pub fn decode(payload: &SessionQualityCachePayload) -> Option<Self> {
        let SessionQualityCachePayload::HierarchicalGaussianV1(payload) = payload else {
            return None;
        };
        Some(Self {
            semantic_mood_mean: payload.semantic_mood_mean.as_slice().try_into().ok()?,
            semantic_mood_variance: payload.semantic_mood_variance.as_slice().try_into().ok()?,
            vibe_mean: payload.vibe_mean.as_slice().try_into().ok()?,
            vibe_variance: payload.vibe_variance.as_slice().try_into().ok()?,
            frontier_mean: payload.frontier_mean,
            frontier_variance: payload.frontier_variance,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PerturbativeAssetPosterior {
    pub baseline_mean: f32,
    pub baseline_variance: f32,
    pub canonical_mean: f32,
    pub canonical_variance: f32,
    pub perturbation_basis: [f32; PERTURBATIVE_DIM],
    pub technical_mean: Option<f32>,
    pub technical_variance: Option<f32>,
    pub domain_label: AssetDomainLabel,
}

impl PerturbativeAssetPosterior {
    pub fn decode(payload: &AssetQualityCachePayload) -> Option<Self> {
        match payload {
            AssetQualityCachePayload::HierarchicalPerturbativeV2(payload) => Some(Self {
                baseline_mean: payload.baseline_mean,
                baseline_variance: payload.baseline_variance,
                canonical_mean: payload.canonical_mean,
                canonical_variance: payload.canonical_variance,
                perturbation_basis: payload.perturbation_basis.as_slice().try_into().ok()?,
                technical_mean: payload.technical_mean,
                technical_variance: payload.technical_variance,
                domain_label: if payload.technical_mean.is_some() {
                    AssetDomainLabel::Real
                } else {
                    AssetDomainLabel::Anime
                },
            }),
            AssetQualityCachePayload::HierarchicalPerturbativeV3(payload) => Some(Self {
                baseline_mean: payload.baseline_mean,
                baseline_variance: payload.baseline_variance,
                canonical_mean: payload.canonical_mean,
                canonical_variance: payload.canonical_variance,
                perturbation_basis: payload.perturbation_basis.as_slice().try_into().ok()?,
                technical_mean: payload.technical_mean,
                technical_variance: payload.technical_variance,
                domain_label: payload.domain_label,
            }),
            _ => None,
        }
    }

    #[must_use]
    pub fn semantic_basis(&self) -> &[f32] {
        &self.perturbation_basis[..LATENT_DIM]
    }

    #[must_use]
    pub fn vibe_basis(&self) -> &[f32] {
        &self.perturbation_basis[LATENT_DIM..]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PerturbativeSessionPosterior {
    pub perturbation_weight_mean: [f32; PERTURBATIVE_DIM],
    pub perturbation_weight_variance: [f32; PERTURBATIVE_DIM],
    pub importance_raw_mean: f32,
    pub importance_raw_variance: f32,
    pub threshold_mean: f32,
    pub threshold_variance: f32,
}

impl PerturbativeSessionPosterior {
    pub fn decode(payload: &SessionQualityCachePayload) -> Option<Self> {
        match payload {
            SessionQualityCachePayload::HierarchicalPerturbativeV2(payload) => Some(Self {
                perturbation_weight_mean: payload
                    .perturbation_weight_mean
                    .as_slice()
                    .try_into()
                    .ok()?,
                perturbation_weight_variance: payload
                    .perturbation_weight_variance
                    .as_slice()
                    .try_into()
                    .ok()?,
                importance_raw_mean: payload.importance_raw_mean,
                importance_raw_variance: payload.importance_raw_variance,
                threshold_mean: payload.threshold_mean,
                threshold_variance: payload.threshold_variance,
            }),
            SessionQualityCachePayload::HierarchicalPerturbativeV3(payload) => Some(Self {
                perturbation_weight_mean: payload
                    .perturbation_weight_mean
                    .as_slice()
                    .try_into()
                    .ok()?,
                perturbation_weight_variance: payload
                    .perturbation_weight_variance
                    .as_slice()
                    .try_into()
                    .ok()?,
                importance_raw_mean: payload.importance_raw_mean,
                importance_raw_variance: payload.importance_raw_variance,
                threshold_mean: payload.threshold_mean,
                threshold_variance: payload.threshold_variance,
            }),
            _ => None,
        }
    }

    #[must_use]
    pub fn semantic_weight_mean(&self) -> &[f32] {
        &self.perturbation_weight_mean[..LATENT_DIM]
    }

    #[must_use]
    pub fn semantic_weight_variance(&self) -> &[f32] {
        &self.perturbation_weight_variance[..LATENT_DIM]
    }

    #[must_use]
    pub fn vibe_weight_mean(&self) -> &[f32] {
        &self.perturbation_weight_mean[LATENT_DIM..]
    }

    #[must_use]
    pub fn vibe_weight_variance(&self) -> &[f32] {
        &self.perturbation_weight_variance[LATENT_DIM..]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PerturbativeHyperParamsV3 {
    pub face_weight_real: f32,
    pub face_weight_anime: f32,
    pub technical_weight_real: f32,
    pub vibe_scale_real: f32,
}

impl PerturbativeHyperParamsV3 {
    #[must_use]
    pub fn face_weight(self, domain_label: AssetDomainLabel) -> f32 {
        match domain_label {
            AssetDomainLabel::Real => self.face_weight_real,
            AssetDomainLabel::Anime => self.face_weight_anime,
        }
    }

    #[must_use]
    pub fn technical_weight(self, domain_label: AssetDomainLabel) -> f32 {
        match domain_label {
            AssetDomainLabel::Real => self.technical_weight_real,
            AssetDomainLabel::Anime => 0.0,
        }
    }

    #[must_use]
    pub fn vibe_scale(self, domain_label: AssetDomainLabel) -> f32 {
        match domain_label {
            AssetDomainLabel::Real => self.vibe_scale_real,
            AssetDomainLabel::Anime => 0.0,
        }
    }
}

impl Default for PerturbativeHyperParamsV3 {
    fn default() -> Self {
        Self {
            face_weight_real: PERTURBATIVE_FACE_WEIGHT_REAL_DEFAULT,
            face_weight_anime: PERTURBATIVE_FACE_WEIGHT_ANIME_DEFAULT,
            technical_weight_real: PERTURBATIVE_TECH_WEIGHT_REAL_DEFAULT,
            vibe_scale_real: PERTURBATIVE_VIBE_SCALE_REAL_DEFAULT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QualityReplayStats {
    pub formal_version: QualityFormalVersion,
    pub asset_count: usize,
    pub session_count: usize,
    pub subject_count: usize,
    pub comparison_events: usize,
    pub nudge_events: usize,
    pub heart_events: usize,
}

pub fn decode_quality_payload<T>(raw: &str) -> anyhow::Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(raw).context("decoding quality cache payload")
}

pub fn encode_quality_payload<T>(payload: &T) -> anyhow::Result<String>
where
    T: Serialize,
{
    serde_json::to_string(payload).context("encoding quality cache payload")
}
