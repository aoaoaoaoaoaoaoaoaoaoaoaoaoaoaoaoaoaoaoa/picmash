use std::path::PathBuf;

use crate::{
    fault::{Fault, Result},
    ids::{AssetId, CollectionId, ObservationId, OccurrenceId, PromptId, SessionId, SnapshotId},
    media::{BlobDigest, RenderDigest},
};

#[derive(Debug, Clone)]
pub struct Collection {
    pub id: CollectionId,
    pub root: PathBuf,
    pub catalog_revision: u64,
}

#[derive(Debug, Clone)]
pub struct AssetOccurrence {
    pub id: OccurrenceId,
    pub asset_id: AssetId,
    pub path: PathBuf,
    pub blob: BlobDigest,
    pub render: RenderDigest,
    pub width: u32,
    pub height: u32,
    pub byte_len: u64,
    pub rotation_quarters: u8,
}

#[derive(Debug, Clone)]
pub struct AssetView {
    pub id: AssetId,
    pub occurrence: AssetOccurrence,
    pub occurrence_count: u32,
    pub favorite: bool,
    pub duel_count: u32,
    pub preference_score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ScanFailure {
    pub path: PathBuf,
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct ScanReport {
    pub collection_id: CollectionId,
    pub generation: u64,
    pub discovered_paths: usize,
    pub reused_paths: usize,
    pub visible_assets: usize,
    pub retired_occurrences: usize,
    pub failures: Vec<ScanFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanProgress {
    pub inspected_paths: usize,
    pub total_paths: usize,
    pub reused_paths: usize,
}

impl ScanProgress {
    #[must_use]
    pub fn percent(self) -> usize {
        self.inspected_paths
            .saturating_mul(100)
            .checked_div(self.total_paths)
            .unwrap_or(100)
    }
}

#[derive(Debug, Clone)]
pub struct LegacyImportReport {
    pub source_fingerprint: String,
    pub imported_assets: usize,
    pub imported_observations: usize,
    pub ambiguous_observations: usize,
    pub already_imported: bool,
}

#[derive(Debug, Clone)]
pub struct JudgmentSession {
    pub id: SessionId,
    pub collection_id: CollectionId,
    pub context_revision: String,
    pub started_at_ns: i64,
    pub ended_at_ns: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentedAsset {
    pub asset_id: AssetId,
    pub occurrence_id: OccurrenceId,
    pub render: RenderDigest,
    pub rotation_quarters: u8,
}

impl PresentedAsset {
    /// Reconstitutes an exact presentation captured by a durable outer
    /// protocol. This does not assert that the occurrence is still present;
    /// the engine checks that relationship when the presentation is used.
    pub fn from_persisted(
        asset_id: AssetId,
        occurrence_id: i64,
        render_digest: String,
        rotation_quarters: u8,
    ) -> Result<Self> {
        if occurrence_id <= 0 {
            return Err(Fault::Corrupt(format!(
                "invalid persisted occurrence id {occurrence_id}"
            )));
        }
        if rotation_quarters >= 4 {
            return Err(Fault::Corrupt(format!(
                "invalid persisted rotation {rotation_quarters}"
            )));
        }
        Ok(Self {
            asset_id,
            occurrence_id: OccurrenceId::from_raw(occurrence_id),
            render: RenderDigest::parse(render_digest)?,
            rotation_quarters,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuelVictor {
    Anchor,
    Challenger,
}

#[derive(Debug, Clone)]
pub struct ComparisonPrompt {
    pub id: PromptId,
    pub session_id: SessionId,
    pub left: PresentedAsset,
    pub right: PresentedAsset,
    pub policy_revision: String,
    pub snapshot_id: Option<SnapshotId>,
    pub issued_at_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdJudgment {
    Admit,
    Reject,
}

impl ThresholdJudgment {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::Reject => "reject",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreferenceScore {
    pub asset_id: AssetId,
    pub score: f64,
    pub duel_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PreferenceEvaluation {
    pub training_duels: usize,
    pub held_out_duels: usize,
    pub log_loss: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct PreferenceSnapshot {
    pub id: SnapshotId,
    pub collection_id: CollectionId,
    pub observation_frontier: ObservationId,
    pub catalog_revision: u64,
    pub model_revision: String,
    pub evaluation: Option<PreferenceEvaluation>,
    pub scores: Vec<PreferenceScore>,
}
