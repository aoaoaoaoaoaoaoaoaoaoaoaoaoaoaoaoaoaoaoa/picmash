use serde::{Deserialize, Serialize};

use super::LATENT_DIM;

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
