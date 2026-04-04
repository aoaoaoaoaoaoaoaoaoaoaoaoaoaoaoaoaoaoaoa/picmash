use std::str::FromStr;

use anyhow::bail;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaintenanceJobKind {
    BootstrapMaintenance,
    CorpusIngest,
    LocalDirectoryRefresh,
    CorpusFaceScanBackfill,
    CorpusFaceRecognitionBackfill,
    CorpusQualityFeatureBackfill,
    ExternalFaceEmbeddingBackfill,
    QualityModelRefresh,
    IdentityReviewRefresh,
}

impl MaintenanceJobKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BootstrapMaintenance => "bootstrap_maintenance",
            Self::CorpusIngest => "corpus_ingest",
            Self::LocalDirectoryRefresh => "local_directory_refresh",
            Self::CorpusFaceScanBackfill => "corpus_face_scan_backfill",
            Self::CorpusFaceRecognitionBackfill => "corpus_face_recognition_backfill",
            Self::CorpusQualityFeatureBackfill => "corpus_quality_feature_backfill",
            Self::ExternalFaceEmbeddingBackfill => "external_face_embedding_backfill",
            Self::QualityModelRefresh => "quality_model_refresh",
            Self::IdentityReviewRefresh => "identity_review_refresh",
        }
    }
}

impl FromStr for MaintenanceJobKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bootstrap_maintenance" => Ok(Self::BootstrapMaintenance),
            "corpus_ingest" => Ok(Self::CorpusIngest),
            "local_directory_refresh" => Ok(Self::LocalDirectoryRefresh),
            "corpus_face_scan_backfill" => Ok(Self::CorpusFaceScanBackfill),
            "corpus_face_recognition_backfill" => Ok(Self::CorpusFaceRecognitionBackfill),
            "corpus_quality_feature_backfill" => Ok(Self::CorpusQualityFeatureBackfill),
            "external_face_embedding_backfill" => Ok(Self::ExternalFaceEmbeddingBackfill),
            "quality_model_refresh" => Ok(Self::QualityModelRefresh),
            "identity_review_refresh" => Ok(Self::IdentityReviewRefresh),
            _ => bail!("unknown maintenance job kind `{value}`"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaintenancePriority {
    Hot = 0,
    Warm = 10,
    Cold = 20,
}

impl MaintenancePriority {
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self as i64
    }

    pub fn from_i64(value: i64) -> anyhow::Result<Self> {
        match value {
            0 => Ok(Self::Hot),
            10 => Ok(Self::Warm),
            20 => Ok(Self::Cold),
            _ => bail!("unknown maintenance priority `{value}`"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceJobSpec {
    pub kind: MaintenanceJobKind,
    pub key: String,
    pub priority: MaintenancePriority,
    pub not_before_ts: i64,
}

impl MaintenanceJobSpec {
    #[must_use]
    pub fn singleton(
        kind: MaintenanceJobKind,
        priority: MaintenancePriority,
        not_before_ts: i64,
    ) -> Self {
        Self {
            kind,
            key: String::new(),
            priority,
            not_before_ts,
        }
    }

    #[must_use]
    pub fn keyed(
        kind: MaintenanceJobKind,
        key: impl Into<String>,
        priority: MaintenancePriority,
        not_before_ts: i64,
    ) -> Self {
        Self {
            kind,
            key: key.into(),
            priority,
            not_before_ts,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedMaintenanceJob {
    pub kind: MaintenanceJobKind,
    pub key: String,
    pub priority: MaintenancePriority,
    pub generation: i64,
}
