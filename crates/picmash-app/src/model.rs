use std::{collections::HashMap, path::PathBuf};

use faer::Mat;
use manifolds_rs::{UmapParams, umap};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::Rng;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::asset_domain::{AssetDomainLabel, AssetDomainPrediction};
use crate::quality::AssetQualityCachePayload;
use crate::quality_features::AssetQualityFeatures;

pub const LATENT_DIM: usize = 3;
pub const SIMILARITY_DIM: usize = 5;
pub const MAP_DIM: usize = 2;
pub const ARENA_RECENT_REPEAT_EXCLUDE: usize = 100;
pub const ORDINAL_BOOTSTRAP_TRIADS: usize = 24;
const RAW_LAYOUT_DIM: usize = 32;
const LAYOUT_MARGIN: f32 = 0.08;
const LAYOUT_SPAN: f32 = 1.0 - (LAYOUT_MARGIN * 2.0);
const PCA_EPSILON: f32 = 1e-6;
const ORDINAL_PRIOR_PULL: f32 = 0.02;
const ORDINAL_PRIOR_REFIT_RIDGE: f32 = 0.05;
const LEARNED_LAYOUT_MAX_SMACOF_POINTS: usize = 900;
const LEARNED_LAYOUT_MAX_ITERATIONS: usize = 32;
const LEARNED_LAYOUT_TOLERANCE: f32 = 1e-4;
const LAYOUT_DISTANCE_EPSILON: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorpusId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RemoteItemId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FaceId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FaceIdentityId(pub i64);

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
    pub corpus_id: CorpusId,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArenaHandle {
    Local(AssetId),
    Remote(RemoteItemId),
}

impl ArenaHandle {
    #[must_use]
    pub fn slug(&self) -> String {
        match self {
            Self::Local(asset_id) => format!("asset_{}", asset_id.0),
            Self::Remote(item_id) => format!("remote_{}", item_id.0),
        }
    }
}

impl std::str::FromStr for ArenaHandle {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(asset_id) = value.strip_prefix("asset_") {
            return Ok(Self::Local(AssetId(asset_id.to_owned())));
        }
        if let Some(remote_id) = value.strip_prefix("remote_") {
            let parsed = remote_id
                .parse::<i64>()
                .map_err(|_| "invalid remote arena handle")?;
            return Ok(Self::Remote(RemoteItemId(parsed)));
        }
        Err("unknown arena handle")
    }
}

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

#[derive(Debug, Clone)]
pub struct EmbeddingRecord {
    pub model_name: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionModel {
    pub model_name: String,
    pub dim: usize,
    pub weights: Vec<f32>,
    pub bias: [f32; LATENT_DIM],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEmbeddingHead {
    pub model_name: String,
    pub dim: usize,
    pub weights: Vec<f32>,
}

impl SessionEmbeddingHead {
    pub fn zero(model_name: String, dim: usize) -> Self {
        Self {
            model_name,
            dim,
            weights: vec![0.0; dim],
        }
    }

    pub fn score(&self, embedding: &[f32]) -> f32 {
        if embedding.len() != self.dim {
            return 0.0;
        }

        self.weights
            .iter()
            .zip(embedding.iter())
            .map(|(weight, value)| weight * value)
            .sum()
    }

    pub fn unary_step(
        &mut self,
        embedding: &[f32],
        signal: f32,
        learning_rate: f32,
        weight_decay: f32,
    ) {
        if embedding.len() != self.dim {
            return;
        }

        for (weight, value) in self.weights.iter_mut().zip(embedding.iter().copied()) {
            *weight += learning_rate * (signal * value - weight_decay * *weight);
        }
    }

    pub fn contrast_step(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
        signal: f32,
        learning_rate: f32,
        weight_decay: f32,
    ) {
        if lhs.len() != self.dim || rhs.len() != self.dim {
            return;
        }

        for ((weight, win), lose) in self
            .weights
            .iter_mut()
            .zip(lhs.iter().copied())
            .zip(rhs.iter().copied())
        {
            *weight += learning_rate * (signal * (win - lose) - weight_decay * *weight);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
pub enum SimilarityChoice {
    #[serde(rename = "ab")]
    #[ts(rename = "ab")]
    Ab,
    #[serde(rename = "ac")]
    #[ts(rename = "ac")]
    Ac,
    #[serde(rename = "bc")]
    #[ts(rename = "bc")]
    Bc,
}

impl SimilarityChoice {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ab => "ab",
            Self::Ac => "ac",
            Self::Bc => "bc",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::Ab => 0,
            Self::Ac => 1,
            Self::Bc => 2,
        }
    }
}

impl std::str::FromStr for SimilarityChoice {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ab" => Ok(Self::Ab),
            "ac" => Ok(Self::Ac),
            "bc" => Ok(Self::Bc),
            _ => Err("unknown similarity pair"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityObservation {
    pub asset_a: AssetId,
    pub asset_b: AssetId,
    pub asset_c: AssetId,
    pub choice: SimilarityChoice,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSimilarityModel {
    pub model_name: String,
    pub dim: usize,
    pub mean: Vec<f32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrdinalSimilarityModel {
    pub prior: LinearSimilarityModel,
    pub latents: HashMap<AssetId, [f32; SIMILARITY_DIM]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "geometry", rename_all = "snake_case")]
pub enum SimilarityModel {
    Linear(LinearSimilarityModel),
    Ordinal(OrdinalSimilarityModel),
}

impl LinearSimilarityModel {
    pub fn from_pca(model_name: String, embeddings: &[Vec<f32>]) -> Option<Self> {
        let dim = embeddings.first()?.len();
        if dim == 0
            || embeddings.iter().any(|embedding| {
                embedding.len() != dim || embedding.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }

        let sample_count = embeddings.len();
        let mut mean = vec![0.0; dim];
        for embedding in embeddings {
            for (slot, value) in mean.iter_mut().zip(embedding.iter().copied()) {
                *slot += value;
            }
        }
        for slot in &mut mean {
            *slot /= sample_count as f32;
        }

        let centered = embeddings
            .iter()
            .map(|embedding| centered_embedding(embedding, &mean))
            .collect::<Vec<_>>();
        let data = DMatrix::from_fn(sample_count, dim, |row, column| centered[row][column]);
        let decomposition = data.svd(false, true);
        let basis = decomposition.v_t?;

        let mut weights = vec![0.0; SIMILARITY_DIM * dim];
        for axis in 0..SIMILARITY_DIM.min(dim).min(basis.nrows()) {
            let component = basis.row(axis);
            let row = &mut weights[(axis * dim)..((axis + 1) * dim)];
            for (slot, value) in row.iter_mut().zip(component.iter()) {
                *slot = *value;
            }
        }

        Some(Self {
            model_name,
            dim,
            mean,
            weights,
        })
    }

    pub fn fit_to_targets(
        model_name: String,
        embeddings: &HashMap<AssetId, Vec<f32>>,
        latents: &HashMap<AssetId, [f32; SIMILARITY_DIM]>,
        ridge: f32,
    ) -> Option<Self> {
        let sample_ids = latents
            .keys()
            .filter(|asset_id| embeddings.contains_key(*asset_id))
            .cloned()
            .collect::<Vec<_>>();
        let first = sample_ids
            .first()
            .and_then(|asset_id| embeddings.get(asset_id))?;
        let dim = first.len();
        if dim == 0 || sample_ids.len() < 2 {
            return None;
        }

        let sample_count = sample_ids.len();
        let mut mean = vec![0.0; dim];
        for asset_id in &sample_ids {
            let embedding = embeddings.get(asset_id)?;
            if embedding.len() != dim || embedding.iter().any(|value| !value.is_finite()) {
                return None;
            }
            for (slot, value) in mean.iter_mut().zip(embedding.iter().copied()) {
                *slot += value;
            }
        }
        for slot in &mut mean {
            *slot /= sample_count as f32;
        }

        let x = DMatrix::from_fn(sample_count, dim, |row, column| {
            let embedding = &embeddings[&sample_ids[row]];
            embedding[column] - mean[column]
        });
        let y = DMatrix::from_fn(sample_count, SIMILARITY_DIM, |row, axis| {
            latents[&sample_ids[row]][axis]
        });
        let xt = x.transpose();
        let gram = (&xt * &x) + DMatrix::identity(dim, dim).scale(ridge.max(PCA_EPSILON));
        let rhs = xt * y;
        let cholesky = gram.cholesky()?;
        let solved = cholesky.solve(&rhs);
        let mut weights = vec![0.0; SIMILARITY_DIM * dim];
        for axis in 0..SIMILARITY_DIM {
            for column in 0..dim {
                weights[axis * dim + column] = solved[(column, axis)];
            }
        }

        Some(Self {
            model_name,
            dim,
            mean,
            weights,
        })
    }

    pub fn is_usable(&self, dim: usize) -> bool {
        self.dim == dim
            && self.mean.len() == dim
            && self.weights.len() == dim * SIMILARITY_DIM
            && self
                .mean
                .iter()
                .chain(self.weights.iter())
                .all(|value| value.is_finite())
    }

    pub fn project(&self, embedding: &[f32]) -> [f32; SIMILARITY_DIM] {
        if embedding.len() != self.dim || self.mean.len() != self.dim {
            return [0.0; SIMILARITY_DIM];
        }

        let mut out = [0.0; SIMILARITY_DIM];
        for (axis, slot) in out.iter_mut().enumerate() {
            let row = &self.weights[(axis * self.dim)..((axis + 1) * self.dim)];
            *slot = row
                .iter()
                .zip(embedding.iter().zip(self.mean.iter()))
                .map(|(weight, (value, mean_value))| weight * (value - mean_value))
                .sum();
        }
        finite_latent(out)
    }

    pub fn distance_sq(&self, lhs: &[f32], rhs: &[f32]) -> f32 {
        squared_similarity_gap(&self.project(lhs), &self.project(rhs))
    }

    pub fn triad_probabilities(&self, a: &[f32], b: &[f32], c: &[f32], beta: f32) -> [f32; 3] {
        triad_probabilities_from_latents(&self.project(a), &self.project(b), &self.project(c), beta)
    }

    pub fn triad_step(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        chosen: SimilarityChoice,
        learning_rate: f32,
        beta: f32,
        weight_decay: f32,
    ) {
        if [a.len(), b.len(), c.len()]
            .iter()
            .any(|length| *length != self.dim)
        {
            return;
        }

        let centered = [
            centered_embedding(a, &self.mean),
            centered_embedding(b, &self.mean),
            centered_embedding(c, &self.mean),
        ];
        let projected = [self.project(a), self.project(b), self.project(c)];
        if centered
            .iter()
            .flat_map(|vector| vector.iter())
            .chain(projected.iter().flat_map(|vector| vector.iter()))
            .any(|value| !value.is_finite())
        {
            return;
        }

        let pair_deltas = [
            difference(&projected[0], &projected[1]),
            difference(&projected[0], &projected[2]),
            difference(&projected[1], &projected[2]),
        ];
        let embed_deltas = [
            centered_delta(&centered[0], &centered[1]),
            centered_delta(&centered[0], &centered[2]),
            centered_delta(&centered[1], &centered[2]),
        ];
        let distances = pair_deltas.map(squared_norm);
        let probabilities = softmax_neg_distances(&distances, beta);
        let chosen_index = chosen.index();

        for (axis, row) in self.weights.chunks_mut(self.dim).enumerate() {
            for (column, weight) in row.iter_mut().enumerate() {
                let mut gradient = -weight_decay * *weight;
                for pair in 0..3 {
                    let target = if pair == chosen_index { 1.0 } else { 0.0 };
                    let coeff =
                        2.0 * beta * (target - probabilities[pair]) * pair_deltas[pair][axis];
                    gradient -= coeff * embed_deltas[pair][column];
                }
                *weight += learning_rate * gradient;
            }
        }

        if self.weights.iter().any(|value| !value.is_finite()) {
            self.weights.fill(0.0);
            for axis in 0..SIMILARITY_DIM.min(self.dim) {
                self.weights[axis * self.dim + axis] = 1.0;
            }
        }
    }

    pub fn is_viable_for(&self, embeddings: &[Vec<f32>]) -> bool {
        let Some(first) = embeddings.first() else {
            return false;
        };
        if self.dim == 0
            || first.len() != self.dim
            || !self.is_usable(first.len())
            || embeddings
                .iter()
                .any(|embedding| embedding.len() != self.dim)
        {
            return false;
        }

        let projections = embeddings
            .iter()
            .map(|embedding| self.project(embedding))
            .collect::<Vec<_>>();
        total_similarity_variance(&projections) > 1e-5
    }
}

impl OrdinalSimilarityModel {
    pub fn from_prior(
        prior: LinearSimilarityModel,
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> Self {
        let latents = embeddings
            .iter()
            .map(|(asset_id, embedding)| (asset_id.clone(), prior.project(embedding)))
            .collect();
        Self { prior, latents }
    }

    fn latent(&self, asset_id: &AssetId, embedding: &[f32]) -> [f32; SIMILARITY_DIM] {
        self.latents
            .get(asset_id)
            .copied()
            .unwrap_or_else(|| self.prior.project(embedding))
    }

    fn ensure_latent(&mut self, asset_id: &AssetId, embedding: &[f32]) -> [f32; SIMILARITY_DIM] {
        let latent = self.latent(asset_id, embedding);
        self.latents.entry(asset_id.clone()).or_insert(latent);
        latent
    }

    fn refit_prior(&mut self, embeddings: &HashMap<AssetId, Vec<f32>>) {
        if let Some(prior) = LinearSimilarityModel::fit_to_targets(
            self.prior.model_name.clone(),
            embeddings,
            &self.latents,
            ORDINAL_PRIOR_REFIT_RIDGE,
        ) {
            self.prior = prior;
        }
    }

    fn triad_step(
        &mut self,
        asset_a: &AssetId,
        embedding_a: &[f32],
        asset_b: &AssetId,
        embedding_b: &[f32],
        asset_c: &AssetId,
        embedding_c: &[f32],
        chosen: SimilarityChoice,
        learning_rate: f32,
        beta: f32,
        weight_decay: f32,
    ) {
        let mut latents = [
            self.ensure_latent(asset_a, embedding_a),
            self.ensure_latent(asset_b, embedding_b),
            self.ensure_latent(asset_c, embedding_c),
        ];
        let priors = [
            self.prior.project(embedding_a),
            self.prior.project(embedding_b),
            self.prior.project(embedding_c),
        ];
        let pair_deltas = [
            difference(&latents[0], &latents[1]),
            difference(&latents[0], &latents[2]),
            difference(&latents[1], &latents[2]),
        ];
        let distances = pair_deltas.map(squared_norm);
        let probabilities = softmax_neg_distances(&distances, beta);
        let chosen_index = chosen.index();
        let mut gradients = [[0.0; SIMILARITY_DIM]; 3];

        for pair in 0..3 {
            let target = if pair == chosen_index { 1.0 } else { 0.0 };
            let coeff = 2.0 * beta * (probabilities[pair] - target);
            let delta = pair_deltas[pair];
            match pair {
                0 => {
                    accumulate_scaled(&mut gradients[0], &delta, coeff);
                    accumulate_scaled(&mut gradients[1], &delta, -coeff);
                }
                1 => {
                    accumulate_scaled(&mut gradients[0], &delta, coeff);
                    accumulate_scaled(&mut gradients[2], &delta, -coeff);
                }
                2 => {
                    accumulate_scaled(&mut gradients[1], &delta, coeff);
                    accumulate_scaled(&mut gradients[2], &delta, -coeff);
                }
                _ => unreachable!(),
            }
        }

        for index in 0..3 {
            for axis in 0..SIMILARITY_DIM {
                gradients[index][axis] +=
                    2.0 * ORDINAL_PRIOR_PULL * (latents[index][axis] - priors[index][axis]);
                latents[index][axis] -=
                    learning_rate * (gradients[index][axis] + weight_decay * latents[index][axis]);
            }
            latents[index] = finite_latent(latents[index]);
        }

        self.latents.insert(asset_a.clone(), latents[0]);
        self.latents.insert(asset_b.clone(), latents[1]);
        self.latents.insert(asset_c.clone(), latents[2]);
    }

    pub fn is_viable_for(&self, embeddings: &HashMap<AssetId, Vec<f32>>) -> bool {
        if !self.prior.is_usable(self.prior.dim) || self.latents.is_empty() {
            return false;
        }
        if embeddings.values().any(|embedding| {
            embedding.len() != self.prior.dim || embedding.iter().any(|value| !value.is_finite())
        }) {
            return false;
        }
        if self
            .latents
            .values()
            .flat_map(|latent| latent.iter())
            .any(|value| !value.is_finite())
        {
            return false;
        }
        let projections = embeddings
            .iter()
            .map(|(asset_id, embedding)| self.latent(asset_id, embedding))
            .collect::<Vec<_>>();
        total_similarity_variance(&projections) > 1e-5
    }
}

impl SimilarityModel {
    pub fn from_pca(model_name: String, embeddings: &[Vec<f32>]) -> Option<Self> {
        LinearSimilarityModel::from_pca(model_name, embeddings).map(Self::Linear)
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Linear(model) => &model.model_name,
            Self::Ordinal(model) => &model.prior.model_name,
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Linear(model) => model.dim,
            Self::Ordinal(model) => model.prior.dim,
        }
    }

    pub fn project_asset(&self, asset_id: &AssetId, embedding: &[f32]) -> [f32; SIMILARITY_DIM] {
        match self {
            Self::Linear(model) => model.project(embedding),
            Self::Ordinal(model) => model.latent(asset_id, embedding),
        }
    }

    pub fn distance_sq(
        &self,
        lhs_id: &AssetId,
        lhs_embedding: &[f32],
        rhs_id: &AssetId,
        rhs_embedding: &[f32],
    ) -> f32 {
        squared_similarity_gap(
            &self.project_asset(lhs_id, lhs_embedding),
            &self.project_asset(rhs_id, rhs_embedding),
        )
    }

    pub fn triad_probabilities(
        &self,
        asset_a: &AssetId,
        embedding_a: &[f32],
        asset_b: &AssetId,
        embedding_b: &[f32],
        asset_c: &AssetId,
        embedding_c: &[f32],
        beta: f32,
    ) -> [f32; 3] {
        triad_probabilities_from_latents(
            &self.project_asset(asset_a, embedding_a),
            &self.project_asset(asset_b, embedding_b),
            &self.project_asset(asset_c, embedding_c),
            beta,
        )
    }

    pub fn is_viable_for(
        &self,
        asset_ids: &[AssetId],
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> bool {
        let corpus = asset_ids
            .iter()
            .filter_map(|asset_id| embeddings.get(asset_id).cloned())
            .collect::<Vec<_>>();
        if corpus.is_empty() {
            return false;
        }
        match self {
            Self::Linear(model) => model.is_viable_for(&corpus),
            Self::Ordinal(model) => model.is_viable_for(embeddings),
        }
    }

    pub fn bootstrap_ordinal(
        &self,
        embeddings: &HashMap<AssetId, Vec<f32>>,
        history: &[SimilarityObservation],
        learning_rate: f32,
        beta: f32,
        weight_decay: f32,
    ) -> Option<Self> {
        let prior = match self {
            Self::Linear(model) => model.clone(),
            Self::Ordinal(model) => return Some(Self::Ordinal(model.clone())),
        };
        let mut ordinal = OrdinalSimilarityModel::from_prior(prior, embeddings);
        for observation in history {
            let Some(embedding_a) = embeddings.get(&observation.asset_a) else {
                continue;
            };
            let Some(embedding_b) = embeddings.get(&observation.asset_b) else {
                continue;
            };
            let Some(embedding_c) = embeddings.get(&observation.asset_c) else {
                continue;
            };
            ordinal.triad_step(
                &observation.asset_a,
                embedding_a,
                &observation.asset_b,
                embedding_b,
                &observation.asset_c,
                embedding_c,
                observation.choice,
                learning_rate,
                beta,
                weight_decay,
            );
        }
        ordinal.refit_prior(embeddings);
        Some(Self::Ordinal(ordinal))
    }

    pub fn triad_step(
        &mut self,
        embeddings: &HashMap<AssetId, Vec<f32>>,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        chosen: SimilarityChoice,
        learning_rate: f32,
        beta: f32,
        weight_decay: f32,
    ) {
        let Some(embedding_a) = embeddings.get(asset_a) else {
            return;
        };
        let Some(embedding_b) = embeddings.get(asset_b) else {
            return;
        };
        let Some(embedding_c) = embeddings.get(asset_c) else {
            return;
        };
        match self {
            Self::Linear(model) => model.triad_step(
                embedding_a,
                embedding_b,
                embedding_c,
                chosen,
                learning_rate,
                beta,
                weight_decay,
            ),
            Self::Ordinal(model) => {
                model.triad_step(
                    asset_a,
                    embedding_a,
                    asset_b,
                    embedding_b,
                    asset_c,
                    embedding_c,
                    chosen,
                    learning_rate,
                    beta,
                    weight_decay,
                );
                model.refit_prior(embeddings);
            }
        }
    }
}

impl ProjectionModel {
    pub fn zero(model_name: String, dim: usize) -> Self {
        Self {
            model_name,
            dim,
            weights: vec![0.0; LATENT_DIM * dim],
            bias: [0.0; LATENT_DIM],
        }
    }

    pub fn predict(&self, embedding: &[f32]) -> [f32; LATENT_DIM] {
        if embedding.len() != self.dim {
            return self.bias;
        }

        let mut out = self.bias;
        for (axis, slot) in out.iter_mut().enumerate() {
            let mut acc = self.bias[axis];
            let row = &self.weights[(axis * self.dim)..((axis + 1) * self.dim)];
            for (&weight, &value) in row.iter().zip(embedding.iter()) {
                acc += weight * value;
            }
            *slot = acc;
        }
        out
    }

    pub fn gradient_step(
        &mut self,
        embedding: &[f32],
        target: &[f32; LATENT_DIM],
        learning_rate: f32,
        weight_decay: f32,
    ) {
        if embedding.len() != self.dim {
            return;
        }

        let prediction = self.predict(embedding);
        for axis in 0..LATENT_DIM {
            let error = target[axis] - prediction[axis];
            self.bias[axis] += learning_rate * error;
            let row = &mut self.weights[(axis * self.dim)..((axis + 1) * self.dim)];
            for (weight, &value) in row.iter_mut().zip(embedding.iter()) {
                *weight += learning_rate * (error * value - weight_decay * *weight);
            }
        }
    }
}

pub fn umap_reduce_points(points: &[Vec<f32>]) -> Vec<[f32; MAP_DIM]> {
    match points.len() {
        0 => return Vec::new(),
        1 => return vec![[0.5, 0.5]],
        2 => return vec![[0.28, 0.5], [0.72, 0.5]],
        3 => return vec![[0.5, 0.2], [0.26, 0.74], [0.74, 0.74]],
        _ => {}
    }

    let Some(layout) = run_umap_projection(points) else {
        return pca_reduce_points(points);
    };
    if layout
        .iter()
        .flat_map(|point| point.iter())
        .all(|value| value.is_finite())
    {
        layout
    } else {
        pca_reduce_points(points)
    }
}

pub fn learned_reduce_points(points: &[Vec<f32>]) -> Vec<[f32; MAP_DIM]> {
    match points.len() {
        0 => Vec::new(),
        1 => vec![[0.5, 0.5]],
        2 => vec![[0.28, 0.5], [0.72, 0.5]],
        3 => vec![[0.5, 0.2], [0.26, 0.74], [0.74, 0.74]],
        _ if points.len() > LEARNED_LAYOUT_MAX_SMACOF_POINTS => pca_reduce_points(points),
        _ => nonmetric_mds_reduce_points(points),
    }
}

pub fn pca_reduce_points(points: &[Vec<f32>]) -> Vec<[f32; MAP_DIM]> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return vec![[0.5, 0.5]];
    }
    let dim = points[0].len();
    if dim == 0 || points.iter().any(|point| point.len() != dim) {
        return Vec::new();
    }

    if points
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return fallback_layout(points.len());
    }

    let data = DMatrix::from_fn(points.len(), dim, |row, column| points[row][column]);
    let mean = mean_columns_dynamic(&data);
    let centered = DMatrix::from_fn(points.len(), dim, |row, column| {
        data[(row, column)] - mean[column]
    });
    let covariance = (&centered.transpose() * &centered) / points.len() as f32;
    let decomposition = SymmetricEigen::new(covariance);
    let order = sorted_eigen_indices(&decomposition.eigenvalues);
    let axes = [order[0], *order.get(1).unwrap_or(&order[0])];
    let basis = decomposition.eigenvectors.select_columns(&axes);
    let reduced = centered * basis;
    normalize_layout_points(
        reduced
            .row_iter()
            .map(|row| [row[0], row[1]])
            .collect::<Vec<_>>(),
    )
}

fn pca_reduce_points_raw(points: &[Vec<f32>]) -> Option<Vec<[f32; MAP_DIM]>> {
    let first = points.first()?;
    let dim = first.len();
    if dim == 0 || points.iter().any(|point| point.len() != dim) {
        return None;
    }
    if points
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }

    let data = DMatrix::from_fn(points.len(), dim, |row, column| points[row][column]);
    let mean = mean_columns_dynamic(&data);
    let centered = DMatrix::from_fn(points.len(), dim, |row, column| {
        data[(row, column)] - mean[column]
    });
    let covariance = (&centered.transpose() * &centered) / points.len() as f32;
    let decomposition = SymmetricEigen::new(covariance);
    let order = sorted_eigen_indices(&decomposition.eigenvalues);
    let axes = [order[0], *order.get(1).unwrap_or(&order[0])];
    let basis = decomposition.eigenvectors.select_columns(&axes);
    let reduced = centered * basis;
    Some(
        reduced
            .row_iter()
            .map(|row| [row[0], row[1]])
            .collect::<Vec<_>>(),
    )
}

fn nonmetric_mds_reduce_points(points: &[Vec<f32>]) -> Vec<[f32; MAP_DIM]> {
    let Some(dissimilarities) = pairwise_distance_matrix(points) else {
        return pca_reduce_points(points);
    };
    let Some(mut coords) = pca_reduce_points_raw(points) else {
        return pca_reduce_points(points);
    };
    if coords.len() != points.len() {
        return pca_reduce_points(points);
    }
    center_layout(&mut coords);

    let pair_order = ordered_pair_indices(&dissimilarities, points.len());
    if pair_order.is_empty() {
        return normalize_layout_points(coords);
    }

    let mut previous_stress = f32::INFINITY;
    for _ in 0..LEARNED_LAYOUT_MAX_ITERATIONS {
        let current = pairwise_distance_matrix_2d(&coords);
        let disparities = monotone_disparities(&dissimilarities, &current, &pair_order);
        let next = smacof_step(&coords, &current, &disparities);
        let stress = normalized_stress(&current, &disparities, points.len());
        coords = next;
        center_layout(&mut coords);
        if (previous_stress - stress).abs() <= LEARNED_LAYOUT_TOLERANCE {
            break;
        }
        previous_stress = stress;
    }

    normalize_layout_points(coords)
}

fn run_umap_projection(points: &[Vec<f32>]) -> Option<Vec<[f32; MAP_DIM]>> {
    let dim = points.first()?.len();
    if dim == 0 || points.iter().any(|point| point.len() != dim) {
        return None;
    }

    let data = Mat::from_fn(points.len(), dim, |row, column| {
        f64::from(points[row][column])
    });
    let params = umap_params(points.len(), dim);
    let embedding = umap(data.as_ref(), None, &params, 42, false);
    embedding_to_layout(&embedding)
}

fn umap_params(sample_count: usize, dim: usize) -> UmapParams<f64> {
    let neighbors = if dim <= SIMILARITY_DIM {
        sample_count.saturating_sub(1).clamp(6, 36)
    } else {
        sample_count.saturating_sub(1).clamp(18, 72)
    };
    let ann_type = if sample_count <= 4_096 || dim <= SIMILARITY_DIM {
        "exhaustive"
    } else {
        "nndescent"
    };
    UmapParams::new(
        Some(MAP_DIM),
        Some(neighbors),
        Some("adam_parallel".to_owned()),
        Some(ann_type.to_owned()),
        Some("pca".to_owned()),
        None,
        None,
        None,
        None,
        Some(false),
    )
}

fn embedding_to_layout(embedding: &[Vec<f64>]) -> Option<Vec<[f32; MAP_DIM]>> {
    let [xs, ys] = embedding else {
        return None;
    };
    if xs.len() != ys.len() || xs.is_empty() {
        return None;
    }

    Some(normalize_layout_points(
        xs.iter()
            .zip(ys.iter())
            .map(|(x, y)| [*x as f32, *y as f32])
            .collect(),
    ))
}

fn pairwise_distance_matrix(points: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = points.first()?;
    let dim = first.len();
    if dim == 0 || points.iter().any(|point| point.len() != dim) {
        return None;
    }
    let count = points.len();
    let mut distances = vec![0.0; count * count];
    for left in 0..count {
        for right in (left + 1)..count {
            let distance = points[left]
                .iter()
                .zip(points[right].iter())
                .map(|(lhs, rhs)| {
                    let delta = lhs - rhs;
                    delta * delta
                })
                .sum::<f32>()
                .sqrt();
            if !distance.is_finite() {
                return None;
            }
            distances[left * count + right] = distance;
            distances[right * count + left] = distance;
        }
    }
    Some(distances)
}

fn pairwise_distance_matrix_2d(points: &[[f32; MAP_DIM]]) -> Vec<f32> {
    let count = points.len();
    let mut distances = vec![0.0; count * count];
    for left in 0..count {
        for right in (left + 1)..count {
            let delta_x = points[left][0] - points[right][0];
            let delta_y = points[left][1] - points[right][1];
            let distance = (delta_x.mul_add(delta_x, delta_y * delta_y)).sqrt();
            distances[left * count + right] = distance;
            distances[right * count + left] = distance;
        }
    }
    distances
}

fn ordered_pair_indices(dissimilarities: &[f32], count: usize) -> Vec<(usize, usize)> {
    let mut pairs = (0..count)
        .flat_map(|left| ((left + 1)..count).map(move |right| (left, right)))
        .collect::<Vec<_>>();
    pairs.sort_by(|(lhs_i, lhs_j), (rhs_i, rhs_j)| {
        dissimilarities[*lhs_i * count + *lhs_j]
            .total_cmp(&dissimilarities[*rhs_i * count + *rhs_j])
            .then_with(|| lhs_i.cmp(rhs_i))
            .then_with(|| lhs_j.cmp(rhs_j))
    });
    pairs
}

fn monotone_disparities(
    dissimilarities: &[f32],
    current: &[f32],
    pair_order: &[(usize, usize)],
) -> Vec<f32> {
    let count = infer_point_count(current.len());
    let ordered = pair_order
        .iter()
        .map(|(left, right)| current[left * count + right])
        .collect::<Vec<_>>();
    let mut fitted = isotonic_non_decreasing(&ordered);
    let current_energy = ordered.iter().map(|value| value * value).sum::<f32>();
    let fitted_energy = fitted.iter().map(|value| value * value).sum::<f32>();
    let scale = if fitted_energy > PCA_EPSILON {
        (current_energy / fitted_energy).sqrt()
    } else {
        1.0
    };
    for value in &mut fitted {
        *value *= scale;
    }
    let mut disparities = vec![0.0; dissimilarities.len()];
    for ((left, right), disparity) in pair_order.iter().zip(fitted) {
        disparities[left * count + right] = disparity;
        disparities[right * count + left] = disparity;
    }
    disparities
}

fn isotonic_non_decreasing(values: &[f32]) -> Vec<f32> {
    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        weight: f32,
        mean: f32,
    }

    let mut blocks = values
        .iter()
        .enumerate()
        .map(|(index, value)| Block {
            start: index,
            end: index + 1,
            weight: 1.0,
            mean: *value,
        })
        .collect::<Vec<_>>();

    let mut cursor = 0usize;
    while cursor + 1 < blocks.len() {
        if blocks[cursor].mean <= blocks[cursor + 1].mean {
            cursor += 1;
            continue;
        }
        let merged_weight = blocks[cursor].weight + blocks[cursor + 1].weight;
        let merged_mean = (blocks[cursor].mean * blocks[cursor].weight
            + blocks[cursor + 1].mean * blocks[cursor + 1].weight)
            / merged_weight.max(PCA_EPSILON);
        blocks[cursor] = Block {
            start: blocks[cursor].start,
            end: blocks[cursor + 1].end,
            weight: merged_weight,
            mean: merged_mean,
        };
        blocks.remove(cursor + 1);
        cursor = cursor.saturating_sub(1);
    }

    let mut out = vec![0.0; values.len()];
    for block in blocks {
        for slot in out.iter_mut().take(block.end).skip(block.start) {
            *slot = block.mean;
        }
    }
    out
}

fn smacof_step(
    points: &[[f32; MAP_DIM]],
    current: &[f32],
    disparities: &[f32],
) -> Vec<[f32; MAP_DIM]> {
    let count = points.len();
    let mut next = vec![[0.0; MAP_DIM]; count];
    for left in 0..count {
        for right in (left + 1)..count {
            let index = left * count + right;
            let distance = current[index].max(LAYOUT_DISTANCE_EPSILON);
            let weight = disparities[index] / distance;
            for axis in 0..MAP_DIM {
                let delta = points[left][axis] - points[right][axis];
                next[left][axis] += weight * delta;
                next[right][axis] -= weight * delta;
            }
        }
    }
    let scale = count.max(1) as f32;
    for point in &mut next {
        for axis in point {
            *axis /= scale;
        }
    }
    next
}

fn normalized_stress(current: &[f32], disparities: &[f32], count: usize) -> f32 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for left in 0..count {
        for right in (left + 1)..count {
            let index = left * count + right;
            let delta = disparities[index] - current[index];
            numerator += delta * delta;
            denominator += current[index] * current[index];
        }
    }
    (numerator / denominator.max(PCA_EPSILON)).sqrt()
}

fn infer_point_count(matrix_len: usize) -> usize {
    (matrix_len as f32).sqrt().round() as usize
}

fn center_layout(points: &mut [[f32; MAP_DIM]]) {
    if points.is_empty() {
        return;
    }
    let mut mean = [0.0; MAP_DIM];
    for point in points.iter() {
        for axis in 0..MAP_DIM {
            mean[axis] += point[axis];
        }
    }
    for axis in &mut mean {
        *axis /= points.len() as f32;
    }
    for point in points {
        for axis in 0..MAP_DIM {
            point[axis] -= mean[axis];
        }
    }
}

#[must_use]
pub fn dot(lhs: &[f32; LATENT_DIM], rhs: &[f32; LATENT_DIM]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

#[must_use]
pub fn subtract(lhs: &[f32; LATENT_DIM], rhs: &[f32; LATENT_DIM]) -> [f32; LATENT_DIM] {
    let mut out = [0.0; LATENT_DIM];
    for axis in 0..LATENT_DIM {
        out[axis] = lhs[axis] - rhs[axis];
    }
    out
}

#[must_use]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[must_use]
pub fn canonical_utility(asset: &AssetRecord, mood: &[f32; LATENT_DIM]) -> f32 {
    asset.alpha + dot(&asset.coords, mood)
}

#[must_use]
pub fn session_utility(
    asset: &AssetRecord,
    session: &SessionRecord,
    residual_score: f32,
    exact_offset: f32,
    heart_bias: f32,
) -> f32 {
    canonical_utility(asset, &session.mood) + residual_score + exact_offset + heart_bias
}

#[must_use]
pub fn session_focus(
    asset: &AssetRecord,
    session: &SessionRecord,
    residual_score: f32,
    exact_offset: f32,
    heart_bias: f32,
) -> f32 {
    session_utility(asset, session, residual_score, exact_offset, heart_bias) - session.frontier
}

#[must_use]
pub fn certainty(compare_count: u32) -> f32 {
    1.0 - (1.0 / (1.0 + compare_count as f32).sqrt())
}

#[must_use]
pub fn weighted_choice_index<R: Rng + ?Sized>(rng: &mut R, weights: &[f32]) -> Option<usize> {
    let total: f32 = weights.iter().copied().filter(|weight| *weight > 0.0).sum();
    if total <= f32::EPSILON {
        return None;
    }

    let mut needle = rng.random_range(0.0..total);
    for (index, &weight) in weights.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        if needle <= weight {
            return Some(index);
        }
        needle -= weight;
    }
    weights
        .iter()
        .enumerate()
        .rev()
        .find(|(_, weight)| **weight > 0.0)
        .map(|(index, _)| index)
}

#[must_use]
pub fn sample_softmax_index<R: Rng + ?Sized>(
    rng: &mut R,
    scores: &[f32],
    temperature: f32,
    uniform_mix: f32,
) -> Option<usize> {
    let eligible = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| score.is_finite().then_some(index))
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return None;
    }
    if eligible.len() == 1 {
        return eligible.first().copied();
    }

    let uniform_mix = uniform_mix.clamp(0.0, 1.0);
    if uniform_mix > 0.0 && rng.random::<f32>() < uniform_mix {
        return eligible.get(rng.random_range(0..eligible.len())).copied();
    }

    let temperature = temperature.max(1e-3);
    let ceiling = eligible
        .iter()
        .map(|index| scores[*index])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = vec![0.0; scores.len()];
    for index in eligible {
        weights[index] = ((scores[index] - ceiling) / temperature).exp();
    }
    weighted_choice_index(rng, &weights)
}

fn sorted_eigen_indices(values: &DVector<f32>) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|lhs, rhs| values[*rhs].total_cmp(&values[*lhs]));
    order
}

fn centered_embedding(embedding: &[f32], mean: &[f32]) -> Vec<f32> {
    embedding
        .iter()
        .zip(mean.iter())
        .map(|(value, mean_value)| value - mean_value)
        .collect()
}

fn centered_delta(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left - right)
        .collect()
}

fn difference(lhs: &[f32; SIMILARITY_DIM], rhs: &[f32; SIMILARITY_DIM]) -> [f32; SIMILARITY_DIM] {
    let mut out = [0.0; SIMILARITY_DIM];
    for axis in 0..SIMILARITY_DIM {
        out[axis] = lhs[axis] - rhs[axis];
    }
    out
}

fn squared_norm(value: [f32; SIMILARITY_DIM]) -> f32 {
    value.iter().map(|axis| axis * axis).sum()
}

fn softmax_neg_distances(distances: &[f32; 3], beta: f32) -> [f32; 3] {
    let logits = distances.map(|distance| -beta * distance);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = logits.map(|logit| (logit - max_logit).exp());
    let total = weights.iter().sum::<f32>().max(f32::EPSILON);
    weights.map(|weight| weight / total)
}

fn mean_columns_dynamic(data: &DMatrix<f32>) -> Vec<f32> {
    (0..data.ncols())
        .map(|column| {
            let mut acc = 0.0;
            for row in 0..data.nrows() {
                acc += data[(row, column)];
            }
            acc / data.nrows().max(1) as f32
        })
        .collect()
}

fn finite_latent(mut latent: [f32; SIMILARITY_DIM]) -> [f32; SIMILARITY_DIM] {
    if latent.iter().all(|value| value.is_finite()) {
        return latent;
    }
    latent.fill(0.0);
    latent
}

fn triad_probabilities_from_latents(
    a: &[f32; SIMILARITY_DIM],
    b: &[f32; SIMILARITY_DIM],
    c: &[f32; SIMILARITY_DIM],
    beta: f32,
) -> [f32; 3] {
    let distances = [
        squared_similarity_gap(a, b),
        squared_similarity_gap(a, c),
        squared_similarity_gap(b, c),
    ];
    if distances.iter().any(|distance| !distance.is_finite()) {
        return [1.0 / 3.0; 3];
    }
    softmax_neg_distances(&distances, beta)
}

fn squared_similarity_gap(lhs: &[f32; SIMILARITY_DIM], rhs: &[f32; SIMILARITY_DIM]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left_axis, right_axis)| {
            let delta = left_axis - right_axis;
            delta * delta
        })
        .sum()
}

fn total_similarity_variance(projections: &[[f32; SIMILARITY_DIM]]) -> f32 {
    if projections.is_empty() {
        return 0.0;
    }
    (0..SIMILARITY_DIM)
        .map(|axis| axis_variance(projections.iter().map(|projection| projection[axis])))
        .sum::<f32>()
}

fn accumulate_scaled(
    target: &mut [f32; SIMILARITY_DIM],
    delta: &[f32; SIMILARITY_DIM],
    scale: f32,
) {
    for axis in 0..SIMILARITY_DIM {
        target[axis] += delta[axis] * scale;
    }
}

pub fn prepare_raw_layout_space(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    pca_whiten_points(points, RAW_LAYOUT_DIM)
}

fn pca_whiten_points(points: &[Vec<f32>], target_dim: usize) -> Vec<Vec<f32>> {
    let Some(first) = points.first() else {
        return Vec::new();
    };
    let dim = first.len();
    if dim == 0 || points.iter().any(|point| point.len() != dim) {
        return Vec::new();
    }

    if points
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return Vec::new();
    }

    let data = DMatrix::from_fn(points.len(), dim, |row, column| points[row][column]);
    let mean = mean_columns_dynamic(&data);
    let centered = DMatrix::from_fn(points.len(), dim, |row, column| {
        data[(row, column)] - mean[column]
    });
    let decomposition = centered.clone().svd(false, true);
    let Some(basis) = decomposition.v_t else {
        return points.to_vec();
    };
    let axis_count = target_dim
        .min(dim)
        .min(decomposition.singular_values.len())
        .min(basis.nrows());
    if axis_count == 0 {
        return points.to_vec();
    }

    let mut whitened = vec![vec![0.0; axis_count]; points.len()];
    let sample_scale = (points.len().max(1) as f32).sqrt();
    for axis in 0..axis_count {
        let scale = (decomposition.singular_values[axis] / sample_scale).max(PCA_EPSILON);
        let component = basis.row(axis);
        for (row_index, row) in whitened.iter_mut().enumerate() {
            let projection = component
                .iter()
                .enumerate()
                .map(|(column, value)| centered[(row_index, column)] * *value)
                .sum::<f32>();
            row[axis] = projection / scale;
        }
    }

    for axis in 0..axis_count {
        let axis_mean = whitened.iter().map(|row| row[axis]).sum::<f32>() / points.len() as f32;
        for row in &mut whitened {
            row[axis] -= axis_mean;
        }
    }

    whitened
}

fn normalize_layout_points(points: Vec<[f32; MAP_DIM]>) -> Vec<[f32; MAP_DIM]> {
    let Some(first) = points.first().copied() else {
        return Vec::new();
    };
    if points
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return fallback_layout(points.len());
    }

    let (mut min_x, mut max_x) = (first[0], first[0]);
    let (mut min_y, mut max_y) = (first[1], first[1]);
    for point in &points {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }

    let span = (max_x - min_x).max(max_y - min_y).max(PCA_EPSILON);
    let center_x = (max_x + min_x) * 0.5;
    let center_y = (max_y + min_y) * 0.5;

    spread_duplicate_layout_points(
        points
            .into_iter()
            .map(|[x, y]| {
                let normalized_x = 0.5 + ((x - center_x) / span) * LAYOUT_SPAN;
                let normalized_y = 0.5 + ((y - center_y) / span) * LAYOUT_SPAN;
                [
                    sanitize_layout_axis(normalized_x),
                    sanitize_layout_axis(normalized_y),
                ]
            })
            .collect(),
    )
}

fn fallback_layout(count: usize) -> Vec<[f32; MAP_DIM]> {
    match count {
        0 => Vec::new(),
        1 => vec![[0.5, 0.5]],
        _ => (0..count)
            .map(|index| {
                let t = index as f32 / count.max(1) as f32;
                [
                    sanitize_layout_axis(LAYOUT_MARGIN + (t * LAYOUT_SPAN)),
                    sanitize_layout_axis(0.5),
                ]
            })
            .collect(),
    }
}

fn sanitize_layout_axis(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(LAYOUT_MARGIN, 1.0 - LAYOUT_MARGIN)
    } else {
        0.5
    }
}

fn axis_variance(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let count = values.clone().count();
    if count == 0 {
        return 0.0;
    }
    let mean = values.clone().sum::<f32>() / count as f32;
    values
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / count as f32
}

fn spread_duplicate_layout_points(mut points: Vec<[f32; MAP_DIM]>) -> Vec<[f32; MAP_DIM]> {
    let mut index = 0usize;
    while index < points.len() {
        let bucket = layout_bucket(points[index]);
        let mut group = vec![index];
        let mut scan = index + 1;
        while scan < points.len() {
            if layout_bucket(points[scan]) == bucket {
                group.push(scan);
            }
            scan += 1;
        }
        if group.len() > 1 {
            let center = points[index];
            for (offset, point_index) in group.into_iter().enumerate() {
                let radius = 0.0045 * (offset as f32).sqrt();
                let angle = (offset as f32) * 2.399_963_1;
                points[point_index] = [
                    sanitize_layout_axis(center[0] + (radius * angle.cos())),
                    sanitize_layout_axis(center[1] + (radius * angle.sin())),
                ];
            }
        }
        index += 1;
    }
    points
}

fn layout_bucket(point: [f32; MAP_DIM]) -> [i32; MAP_DIM] {
    point.map(|value| (value * 20_000.0).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::{
        AssetId, LinearSimilarityModel, MAP_DIM, SimilarityChoice, learned_reduce_points,
        normalize_layout_points, pca_reduce_points, prepare_raw_layout_space,
    };

    #[test]
    fn triad_step_moves_the_projection() {
        let mut model = LinearSimilarityModel {
            model_name: "probe".to_owned(),
            dim: 2,
            mean: vec![0.0, 0.0],
            weights: vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 0.0, //
                0.0, 0.0, //
                0.0, 0.0,
            ],
        };
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 2.0];
        let before = model.weights.clone();
        model.triad_step(&a, &b, &c, SimilarityChoice::Ab, 0.08, 1.2, 0.002);
        assert_ne!(model.weights, before);
    }

    #[test]
    fn pca_layout_normalizes_points_into_viewport() {
        let points = [
            vec![-2.0, 0.0, 0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.2, 0.0, 0.0],
            vec![3.0, -1.0, 0.1, 0.0, 0.0],
        ];
        let layout = pca_reduce_points(&points);
        assert_eq!(layout.len(), points.len());
        assert!(
            layout
                .iter()
                .flat_map(|point| point.iter().copied())
                .all(|value| (0.08..=0.92).contains(&value))
        );
    }

    #[test]
    fn layout_axis_stays_inside_the_frame() {
        let points = normalize_layout_points(vec![[-4.0, 0.0], [0.0, 2.0], [7.0, -1.0]]);
        assert!(
            points
                .iter()
                .flat_map(|point| point.iter().copied())
                .all(|value| (0.08..=0.92).contains(&value))
        );
    }

    #[test]
    fn raw_layout_space_is_centered_and_dimension_reduced() {
        let points = vec![
            vec![1.0, 0.0, 0.5, 0.1],
            vec![0.0, 1.0, 0.4, 0.2],
            vec![-1.0, 0.0, 0.6, -0.1],
            vec![0.0, -1.0, 0.3, -0.2],
        ];
        let reduced = prepare_raw_layout_space(&points);
        assert_eq!(reduced.len(), points.len());
        assert!(reduced.iter().all(|row| row.len() <= 4));
        let axis_means = (0..reduced[0].len())
            .map(|axis| reduced.iter().map(|row| row[axis]).sum::<f32>() / reduced.len() as f32)
            .collect::<Vec<_>>();
        assert!(axis_means.iter().all(|mean| mean.abs() < 1e-3));
    }

    #[test]
    fn learned_layout_normalizes_points_into_viewport() {
        let points = [
            vec![-1.2, 0.0, 0.3, 0.0, 0.0],
            vec![-0.8, 0.2, 0.1, 0.0, 0.0],
            vec![0.5, 1.1, -0.4, 0.0, 0.0],
            vec![1.7, -0.9, 0.2, 0.0, 0.0],
        ];
        let layout = learned_reduce_points(&points);
        assert_eq!(layout.len(), points.len());
        assert!(
            layout
                .iter()
                .flat_map(|point| point.iter().copied())
                .all(|value| (0.08..=0.92).contains(&value))
        );
    }

    #[test]
    fn map_dim_constant_remains_two() {
        assert_eq!(MAP_DIM, 2);
    }

    #[test]
    fn asset_id_roundtrips_clone() {
        let asset = AssetId("abc".to_owned());
        assert_eq!(asset, asset.clone());
    }
}
