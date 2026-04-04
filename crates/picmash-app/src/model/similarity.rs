use std::collections::HashMap;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use super::{AssetId, SIMILARITY_DIM};

const ORDINAL_PRIOR_PULL: f32 = 0.02;
const ORDINAL_PRIOR_REFIT_RIDGE: f32 = 0.05;
const PCA_EPSILON: f32 = 1e-6;

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

#[cfg(test)]
mod tests {
    use super::{LinearSimilarityModel, SimilarityChoice};

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
}
