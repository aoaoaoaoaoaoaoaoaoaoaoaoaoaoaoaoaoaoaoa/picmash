use std::path::PathBuf;

use crate::{quality::AssetQualityCachePayload, quality_features::AssetQualityFeatures};

use super::{AssetId, LATENT_DIM, RemoteItemId, SessionId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalEventKind {
    Selected,
    Rejected,
    StreamBlocked,
    LocalWin,
    RemoteWin,
    Hearted,
    Imported,
    Kept,
}

impl ExternalEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::Rejected => "rejected",
            Self::StreamBlocked => "stream_blocked",
            Self::LocalWin => "local_win",
            Self::RemoteWin => "remote_win",
            Self::Hearted => "hearted",
            Self::Imported => "imported",
            Self::Kept => "kept",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssetRecord {
    pub id: AssetId,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub alpha: f32,
    pub coords: [f32; LATENT_DIM],
    pub rotation_quarters: i32,
    pub compare_count: u32,
    pub win_count: u32,
    pub heart_count: u32,
    pub is_hearted: bool,
    pub hidden: bool,
}

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub id: SessionId,
    pub corpus_id: super::CorpusId,
    pub mood: [f32; LATENT_DIM],
    pub frontier: f32,
    pub comparisons: u32,
    pub nudges: u32,
    pub hearts: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSubsourceLock {
    pub source_key: String,
    pub stream_id: i64,
}

#[derive(Debug, Clone)]
pub struct RemoteItemRecord {
    pub id: RemoteItemId,
    pub source_key: String,
    pub stream_title: String,
    pub stream_id: i64,
    pub thread_no: i64,
    pub post_no: i64,
    pub title: String,
    pub path: PathBuf,
    pub image_url: String,
    pub thumb_url: String,
    pub rotation_quarters: i32,
}

#[derive(Debug, Clone)]
pub struct RemoteCandidate {
    pub item: RemoteItemRecord,
    pub embedding: Vec<f32>,
    pub face_embedding: Option<Vec<f32>>,
    pub quality_features: Option<AssetQualityFeatures>,
    pub quality_cache: Option<AssetQualityCachePayload>,
    pub selected_count: u32,
    pub reject_count: u32,
    pub survive_count: u32,
    pub import_count: u32,
    pub win_count: u32,
    pub loss_count: u32,
    pub stream_selected_count: u32,
    pub stream_reject_count: u32,
    pub stream_survive_count: u32,
    pub stream_import_count: u32,
    pub stream_win_count: u32,
    pub stream_loss_count: u32,
    pub stream_image_count: u32,
    pub stream_last_modified: i64,
}
