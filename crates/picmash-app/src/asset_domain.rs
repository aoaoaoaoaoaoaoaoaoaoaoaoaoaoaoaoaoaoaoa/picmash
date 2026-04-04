use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::model::AssetId;

const MIN_LABELS_PER_CLASS: usize = 10;
const TRAINING_EPOCHS: usize = 260;
const BASE_LEARNING_RATE: f32 = 0.34;
const WEIGHT_DECAY: f32 = 8e-4;
pub const CONFIDENCE_THRESHOLD: f32 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum AssetDomainLabel {
    Real,
    Anime,
}

impl AssetDomainLabel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Anime => "anime",
        }
    }

    pub const fn display_str(self) -> &'static str {
        match self {
            Self::Real => "3D",
            Self::Anime => "2D",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::Real => "3D photograph / live action",
            Self::Anime => "2D anime / illustration",
        }
    }

    const fn target(self) -> f32 {
        match self {
            Self::Real => 1.0,
            Self::Anime => 0.0,
        }
    }
}

impl std::str::FromStr for AssetDomainLabel {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "real" => Ok(Self::Real),
            "anime" => Ok(Self::Anime),
            _ => Err("unknown asset domain label"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssetDomainTrainingRow {
    pub asset_id: AssetId,
    pub label: AssetDomainLabel,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AssetDomainStatus {
    pub real_labels: usize,
    pub anime_labels: usize,
    pub trained: bool,
}

impl AssetDomainStatus {
    pub const fn ready(self) -> bool {
        self.trained
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AssetDomainPrediction {
    real_probability: f32,
}

impl AssetDomainPrediction {
    pub fn forge(real_probability: f32) -> Self {
        Self {
            real_probability: real_probability.clamp(0.0, 1.0),
        }
    }

    pub const fn real_probability(self) -> f32 {
        self.real_probability
    }

    pub const fn anime_probability(self) -> f32 {
        1.0 - self.real_probability
    }

    pub fn confidence(self) -> f32 {
        self.real_probability.max(self.anime_probability())
    }

    pub fn label(self) -> AssetDomainLabel {
        if self.real_probability >= 0.5 {
            AssetDomainLabel::Real
        } else {
            AssetDomainLabel::Anime
        }
    }

    pub fn leaning_display_str(self) -> &'static str {
        self.label().display_str()
    }

    pub fn decisive_label(self) -> Option<AssetDomainLabel> {
        (self.confidence() >= CONFIDENCE_THRESHOLD).then(|| self.label())
    }

    pub fn display_percent(self) -> u8 {
        (self.confidence() * 100.0).round() as u8
    }
}

#[derive(Debug, Clone)]
pub struct AssetDomainOracle {
    dim: usize,
    bias: f32,
    weights: Vec<f32>,
    status: AssetDomainStatus,
}

impl AssetDomainOracle {
    pub fn train(rows: &[AssetDomainTrainingRow]) -> Self {
        let Some(dim) = rows.first().map(|row| row.embedding.len()) else {
            return Self {
                dim: 0,
                bias: 0.0,
                weights: Vec::new(),
                status: AssetDomainStatus::default(),
            };
        };
        if dim == 0
            || rows.iter().any(|row| {
                row.embedding.len() != dim || row.embedding.iter().any(|value| !value.is_finite())
            })
        {
            return Self {
                dim: 0,
                bias: 0.0,
                weights: Vec::new(),
                status: AssetDomainStatus::default(),
            };
        }

        let status = AssetDomainStatus {
            real_labels: rows
                .iter()
                .filter(|row| matches!(row.label, AssetDomainLabel::Real))
                .count(),
            anime_labels: rows
                .iter()
                .filter(|row| matches!(row.label, AssetDomainLabel::Anime))
                .count(),
            trained: false,
        };
        if status.real_labels < MIN_LABELS_PER_CLASS || status.anime_labels < MIN_LABELS_PER_CLASS {
            return Self {
                dim,
                bias: 0.0,
                weights: vec![0.0; dim],
                status,
            };
        }

        let samples = rows
            .iter()
            .map(|row| (normalize_embedding(&row.embedding), row.label.target()))
            .collect::<Vec<_>>();

        let mut bias = 0.0f32;
        let mut weights = vec![0.0f32; dim];
        for epoch in 0..TRAINING_EPOCHS {
            let phase = 1.0 - (epoch as f32 / TRAINING_EPOCHS as f32);
            let lr = BASE_LEARNING_RATE * (0.12 + phase * 0.88);
            let mut grad_bias = 0.0f32;
            let mut grad_weights = vec![0.0f32; dim];
            for (embedding, target) in &samples {
                let margin = bias
                    + weights
                        .iter()
                        .zip(embedding.iter())
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>();
                let prediction = sigmoid(margin);
                let error = prediction - target;
                grad_bias += error;
                for (grad, value) in grad_weights.iter_mut().zip(embedding.iter()) {
                    *grad += error * value;
                }
            }
            let inv_n = 1.0 / samples.len() as f32;
            grad_bias *= inv_n;
            for (grad, weight) in grad_weights.iter_mut().zip(weights.iter()) {
                *grad = *grad * inv_n + WEIGHT_DECAY * *weight;
            }
            bias -= lr * grad_bias;
            for (weight, grad) in weights.iter_mut().zip(grad_weights.iter()) {
                *weight -= lr * grad;
            }
        }

        Self {
            dim,
            bias,
            weights,
            status: AssetDomainStatus {
                trained: true,
                ..status
            },
        }
    }

    pub const fn status(&self) -> AssetDomainStatus {
        self.status
    }

    pub fn predict(&self, embedding: &[f32]) -> Option<AssetDomainPrediction> {
        if !self.status.ready()
            || self.dim == 0
            || embedding.len() != self.dim
            || embedding.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let normalized = normalize_embedding(embedding);
        let margin = self.bias
            + self
                .weights
                .iter()
                .zip(normalized.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        Some(AssetDomainPrediction::forge(sigmoid(margin)))
    }
}

fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    embedding.iter().map(|value| value / norm).collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::{AssetDomainLabel, AssetDomainOracle, AssetDomainTrainingRow};
    use crate::model::AssetId;

    #[test]
    fn trains_separable_domain_head() {
        let mut rows = Vec::new();
        for index in 0..12 {
            rows.push(AssetDomainTrainingRow {
                asset_id: AssetId(format!("real-{index}")),
                label: AssetDomainLabel::Real,
                embedding: vec![2.0, 0.4 + index as f32 * 0.01, 0.2],
            });
            rows.push(AssetDomainTrainingRow {
                asset_id: AssetId(format!("anime-{index}")),
                label: AssetDomainLabel::Anime,
                embedding: vec![-2.0, -0.4 - index as f32 * 0.01, -0.2],
            });
        }

        let oracle = AssetDomainOracle::train(&rows);
        assert!(oracle.status().ready());

        let real = oracle.predict(&[1.8, 0.35, 0.15]).expect("real prediction");
        let anime = oracle
            .predict(&[-1.8, -0.35, -0.15])
            .expect("anime prediction");

        assert!(matches!(
            real.decisive_label(),
            Some(AssetDomainLabel::Real)
        ));
        assert!(matches!(
            anime.decisive_label(),
            Some(AssetDomainLabel::Anime)
        ));
    }

    #[test]
    fn stays_cold_until_both_classes_clear_threshold() {
        let mut rows = Vec::new();
        for index in 0..10 {
            rows.push(AssetDomainTrainingRow {
                asset_id: AssetId(format!("real-{index}")),
                label: AssetDomainLabel::Real,
                embedding: vec![1.0, index as f32],
            });
        }
        for index in 0..9 {
            rows.push(AssetDomainTrainingRow {
                asset_id: AssetId(format!("anime-{index}")),
                label: AssetDomainLabel::Anime,
                embedding: vec![-1.0, -(index as f32)],
            });
        }

        let oracle = AssetDomainOracle::train(&rows);
        assert_eq!(oracle.status().real_labels, 10);
        assert_eq!(oracle.status().anime_labels, 9);
        assert!(!oracle.status().ready());
        assert!(oracle.predict(&[0.8, 0.1]).is_none());
    }
}
