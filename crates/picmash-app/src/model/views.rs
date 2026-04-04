use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::asset_domain::{AssetDomainLabel, AssetDomainPrediction};

use super::{ArenaHandle, AssetRecord, MAP_DIM, RemoteItemRecord, SIMILARITY_DIM};

#[derive(Debug, Clone)]
pub struct ArenaLocalCard {
    pub asset: AssetRecord,
    pub hearted: bool,
    pub utility: f32,
    pub quality: AssetQualitySummary,
    pub domain: AssetDomainView,
}

#[derive(Debug, Clone)]
pub struct ArenaRemoteCard {
    pub item: RemoteItemRecord,
    pub hearted: bool,
    pub stream_locked: bool,
    pub utility: f32,
    pub quality: AssetQualitySummary,
}

#[derive(Debug, Clone)]
pub enum ArenaCard {
    Local(ArenaLocalCard),
    Remote(ArenaRemoteCard),
}

impl ArenaCard {
    #[must_use]
    pub fn handle(&self) -> ArenaHandle {
        match self {
            Self::Local(card) => ArenaHandle::Local(card.asset.id.clone()),
            Self::Remote(card) => ArenaHandle::Remote(card.item.id),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArenaPair {
    pub left: ArenaCard,
    pub right: ArenaCard,
}

#[derive(Debug, Clone)]
pub struct ClusterSatellite {
    pub item: RemoteItemRecord,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct DuplicateCluster {
    pub satellites: Vec<ClusterSatellite>,
}

#[derive(Debug, Clone)]
pub struct ArenaView {
    pub pair: Option<ArenaPair>,
    pub cluster: Option<DuplicateCluster>,
}

#[derive(Debug, Clone)]
pub struct BoardEntry {
    pub asset: AssetRecord,
    pub global_score: f32,
    pub certainty: f32,
    pub hearted: bool,
    pub session_offset: f32,
    pub residual_score: f32,
    pub session_utility: f32,
    pub session_focus: f32,
    pub sampling_pull: f32,
    pub quality: AssetQualitySummary,
}

#[derive(Debug, Clone)]
pub struct ExternalSourceOption {
    pub source_key: String,
    pub label: String,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct ExternalArenaStatus {
    pub sources: Vec<ExternalSourceOption>,
    pub external_probability: u8,
    pub arena_explore_percent: u8,
    pub active_streams: usize,
    pub blocked_streams: usize,
    pub cached_items: usize,
    pub dedup_radius_percent: u8,
}

#[derive(Debug, Clone)]
pub struct ExploreEntry {
    pub asset: AssetRecord,
    pub latent: [f32; SIMILARITY_DIM],
    pub plot: [f32; MAP_DIM],
    pub domain: AssetDomainView,
    pub quality: AssetQualitySummary,
}

#[derive(Debug, Clone, Copy)]
pub struct PosteriorSummary {
    pub mean: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AssetDomainView {
    pub manual: Option<AssetDomainLabel>,
    pub predicted: Option<AssetDomainPrediction>,
}

#[derive(Debug, Clone, Copy)]
pub struct AssetQualitySummary {
    pub asset: PosteriorSummary,
    pub baseline: PosteriorSummary,
    pub semantic: Option<PosteriorSummary>,
    pub vibe: Option<PosteriorSummary>,
    pub technical: Option<PosteriorSummary>,
    pub face: Option<PosteriorSummary>,
}

#[derive(Debug, Clone)]
pub struct ExploreTriad {
    pub a: ExploreEntry,
    pub b: ExploreEntry,
    pub c: ExploreEntry,
}

#[derive(Debug, Clone)]
pub struct ExploreView {
    pub map_mode: ExploreMapMode,
    pub points: Vec<ExploreEntry>,
    pub triad: Option<ExploreTriad>,
    pub selection: Option<ExploreSelection>,
}

#[derive(Debug, Clone)]
pub struct ExplorePanels {
    pub triad: Option<ExploreTriad>,
    pub selection: Option<ExploreSelection>,
}

#[derive(Debug, Clone)]
pub struct ExploreSelection {
    pub focus: ExploreEntry,
    pub neighbors: Vec<ExploreNeighbor>,
}

#[derive(Debug, Clone)]
pub struct ExploreNeighbor {
    pub entry: ExploreEntry,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, TS)]
pub enum ExploreMapMode {
    #[serde(rename = "raw")]
    #[ts(rename = "raw")]
    #[default]
    Raw,
    #[serde(rename = "learned")]
    #[ts(rename = "learned")]
    Learned,
}

impl ExploreMapMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Learned => "learned",
        }
    }

    pub const fn distance_label(self) -> &'static str {
        match self {
            Self::Raw => "raw distance",
            Self::Learned => "5D distance",
        }
    }
}

impl std::str::FromStr for ExploreMapMode {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "raw" => Ok(Self::Raw),
            "learned" => Ok(Self::Learned),
            _ => Err("unknown explore map mode"),
        }
    }
}
