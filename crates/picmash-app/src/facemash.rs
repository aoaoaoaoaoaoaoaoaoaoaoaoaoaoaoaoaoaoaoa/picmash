//! Face beauty rating and attractiveness generalization.

use nalgebra::{DMatrix, DVector};
use skillratings::{
    Outcomes,
    weng_lin::{WengLinConfig, WengLinRating, weng_lin},
};

const FACE_BEAUTY_MEAN: f32 = 1500.0;
const FACE_BEAUTY_SIGMA: f32 = 350.0;
pub const FACE_BEAUTY_BETA: f64 = 175.0;
const ORACLE_MIN_TRAINING_DUELS: usize = 32;
const ORACLE_RIDGE: f64 = 12.0;
const ORACLE_NEWTON_STEPS: usize = 18;
const ORACLE_NEWTON_TOLERANCE: f64 = 1e-4;
const ORACLE_HESSIAN_JITTER: f64 = 1e-6;
const ORACLE_LOGIT_CLAMP: f64 = 1e-6;
const ORACLE_NOISE_FLOOR: f64 = 36.0 * 36.0;
const ORACLE_CALIBRATION_RIDGE: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBeauty {
    pub mean: f32,
    pub sigma: f32,
}

impl FaceBeauty {
    #[must_use]
    pub const fn forge(mean: f32, sigma: f32) -> Self {
        Self { mean, sigma }
    }

    #[must_use]
    pub const fn newborn() -> Self {
        Self {
            mean: FACE_BEAUTY_MEAN,
            sigma: FACE_BEAUTY_SIGMA,
        }
    }

    #[must_use]
    pub fn conservative(self) -> f32 {
        self.mean - 3.0 * self.sigma
    }

    fn into_weng_lin(self) -> WengLinRating {
        WengLinRating {
            rating: f64::from(self.mean),
            uncertainty: f64::from(self.sigma),
        }
    }

    fn from_weng_lin(rating: WengLinRating) -> Self {
        Self {
            mean: rating.rating as f32,
            sigma: rating.uncertainty as f32,
        }
    }
}

impl Default for FaceBeauty {
    fn default() -> Self {
        Self::newborn()
    }
}

#[must_use]
pub fn rate_face_win(winner: FaceBeauty, loser: FaceBeauty) -> (FaceBeauty, FaceBeauty) {
    let config = WengLinConfig {
        beta: FACE_BEAUTY_BETA,
        uncertainty_tolerance: 0.000_001,
    };
    let (winner, loser) = weng_lin(
        &winner.into_weng_lin(),
        &loser.into_weng_lin(),
        &Outcomes::WIN,
        &config,
    );
    (
        FaceBeauty::from_weng_lin(winner),
        FaceBeauty::from_weng_lin(loser),
    )
}

#[derive(Debug, Clone)]
pub struct FaceOracleDuelSample {
    pub winner_embedding: Vec<f32>,
    pub loser_embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FaceOracleCalibrationSample {
    pub embedding: Vec<f32>,
    pub beauty: FaceBeauty,
    pub compare_count: u32,
}

impl FaceOracleCalibrationSample {
    fn precision_weight(&self) -> f64 {
        let sigma = f64::from(self.beauty.sigma).max(1.0);
        let precision = 1.0 / sigma.powi(2);
        let duel_bonus = 1.0 + f64::from(self.compare_count).sqrt();
        precision * duel_bonus
    }
}

#[derive(Debug, Clone, Default)]
pub struct FaceOracleTrainingData {
    pub duels: Vec<FaceOracleDuelSample>,
    pub calibration: Vec<FaceOracleCalibrationSample>,
}

#[derive(Debug, Clone, Copy)]
pub struct FacePrediction {
    pub mean: f32,
    pub sigma: f32,
}

impl FacePrediction {
    #[must_use]
    pub fn ucb(self, beta: f32) -> f32 {
        self.mean + beta * self.sigma
    }
}

pub fn pool_embeddings<'a>(embeddings: impl IntoIterator<Item = &'a [f32]>) -> Option<Vec<f32>> {
    let mut embeddings = embeddings.into_iter().peekable();
    let dim = embeddings.peek()?.len();
    if dim == 0 {
        return None;
    }

    let mut pooled = vec![0.0_f64; dim];
    let mut seen = 0usize;
    for embedding in embeddings {
        if embedding.len() != dim {
            return None;
        }
        let norm = embedding
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 1e-9 {
            continue;
        }
        for (slot, value) in pooled.iter_mut().zip(embedding.iter()) {
            *slot += f64::from(*value) / norm;
        }
        seen += 1;
    }
    if seen == 0 {
        return None;
    }
    let pooled_norm = pooled.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
    if !pooled_norm.is_finite() || pooled_norm <= 1e-9 {
        return None;
    }
    Some(
        pooled
            .into_iter()
            .map(|value| (value / pooled_norm) as f32)
            .collect(),
    )
}

#[derive(Debug, Clone)]
pub struct FaceOracle {
    feature_mean: DVector<f64>,
    weights: DVector<f64>,
    covariance: DMatrix<f64>,
    scale: f64,
    bias: f64,
    noise_variance: f64,
}

impl FaceOracle {
    pub fn train(training: &FaceOracleTrainingData) -> Option<Self> {
        if training.duels.len() < ORACLE_MIN_TRAINING_DUELS {
            return None;
        }
        let dim = oracle_dim(training)?;
        let feature_mean = oracle_feature_mean(training, dim)?;
        let duel_rows = training
            .duels
            .iter()
            .map(|duel| duel_difference(duel, dim))
            .collect::<Option<Vec<_>>>()?;
        let (weights, covariance) = train_bradley_terry(&duel_rows, dim)?;
        let (scale, bias, noise_variance) = calibrate_oracle(
            &feature_mean,
            &weights,
            training.calibration.as_slice(),
            dim,
        );
        Some(Self {
            feature_mean,
            weights,
            covariance,
            scale,
            bias,
            noise_variance,
        })
    }

    pub fn predict(&self, embedding: &[f32]) -> Option<FacePrediction> {
        let features = self.design_features(embedding)?;
        let raw_mean = self.weights.dot(&features);
        let mean = self.bias + self.scale * raw_mean;
        let epistemic = (features.transpose() * &self.covariance * &features)[0];
        let variance = (self.scale.powi(2) * epistemic + self.noise_variance).max(0.0);
        let sigma = variance.sqrt();
        if !mean.is_finite() || !sigma.is_finite() {
            return None;
        }
        Some(FacePrediction {
            mean: mean as f32,
            sigma: sigma as f32,
        })
    }

    fn design_features(&self, embedding: &[f32]) -> Option<DVector<f64>> {
        if embedding.len() != self.feature_mean.len() {
            return None;
        }
        Some(DVector::from_iterator(
            embedding.len(),
            embedding
                .iter()
                .zip(self.feature_mean.iter())
                .map(|(value, mean)| f64::from(*value) - mean),
        ))
    }
}

fn oracle_dim(training: &FaceOracleTrainingData) -> Option<usize> {
    let duel_dim = training.duels.first()?.winner_embedding.len();
    if duel_dim == 0 {
        return None;
    }
    if training.duels.iter().any(|duel| {
        duel.winner_embedding.len() != duel_dim || duel.loser_embedding.len() != duel_dim
    }) {
        return None;
    }
    if training
        .calibration
        .iter()
        .any(|sample| sample.embedding.len() != duel_dim)
    {
        return None;
    }
    Some(duel_dim)
}

fn oracle_feature_mean(training: &FaceOracleTrainingData, dim: usize) -> Option<DVector<f64>> {
    let embeddings = if training.calibration.is_empty() {
        training
            .duels
            .iter()
            .flat_map(|duel| [&duel.winner_embedding, &duel.loser_embedding])
            .collect::<Vec<_>>()
    } else {
        training
            .calibration
            .iter()
            .map(|sample| &sample.embedding)
            .collect::<Vec<_>>()
    };
    let count = embeddings.len();
    (count > 0).then(|| {
        DVector::from_iterator(
            dim,
            (0..dim).map(|column| {
                embeddings
                    .iter()
                    .map(|embedding| f64::from(embedding[column]))
                    .sum::<f64>()
                    / count as f64
            }),
        )
    })
}

fn duel_difference(duel: &FaceOracleDuelSample, dim: usize) -> Option<DVector<f64>> {
    (duel.winner_embedding.len() == dim && duel.loser_embedding.len() == dim).then(|| {
        DVector::from_iterator(
            dim,
            duel.winner_embedding
                .iter()
                .zip(duel.loser_embedding.iter())
                .map(|(winner, loser)| f64::from(*winner) - f64::from(*loser)),
        )
    })
}

fn train_bradley_terry(
    duel_rows: &[DVector<f64>],
    dim: usize,
) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let mut weights = DVector::zeros(dim);
    let ridge = DMatrix::<f64>::identity(dim, dim).scale(ORACLE_RIDGE + ORACLE_HESSIAN_JITTER);
    for _ in 0..ORACLE_NEWTON_STEPS {
        let mut gradient = ridge.clone() * &weights;
        let mut precision = ridge.clone();
        for row in duel_rows {
            let margin = weights.dot(row).clamp(-40.0, 40.0);
            let probability = sigmoid(margin).clamp(ORACLE_LOGIT_CLAMP, 1.0 - ORACLE_LOGIT_CLAMP);
            gradient += row.scale(probability - 1.0);
            let curvature = probability * (1.0 - probability);
            precision += (row * row.transpose()).scale(curvature);
        }
        let cholesky = precision.clone().cholesky()?;
        let step = cholesky.solve(&gradient);
        if step.iter().any(|value| !value.is_finite()) {
            return None;
        }
        weights -= &step;
        if step.amax() <= ORACLE_NEWTON_TOLERANCE {
            let covariance = cholesky.solve(&DMatrix::identity(dim, dim));
            if covariance.iter().any(|value| !value.is_finite()) {
                return None;
            }
            return Some((weights, covariance));
        }
    }

    let mut precision = ridge;
    for row in duel_rows {
        let margin = weights.dot(row).clamp(-40.0, 40.0);
        let probability = sigmoid(margin).clamp(ORACLE_LOGIT_CLAMP, 1.0 - ORACLE_LOGIT_CLAMP);
        let curvature = probability * (1.0 - probability);
        precision += (row * row.transpose()).scale(curvature);
    }
    let cholesky = precision.cholesky()?;
    let covariance = cholesky.solve(&DMatrix::identity(dim, dim));
    if covariance.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some((weights, covariance))
}

fn calibrate_oracle(
    feature_mean: &DVector<f64>,
    weights: &DVector<f64>,
    calibration: &[FaceOracleCalibrationSample],
    dim: usize,
) -> (f64, f64, f64) {
    if calibration.is_empty() {
        return (
            FACE_BEAUTY_BETA,
            f64::from(FACE_BEAUTY_MEAN),
            f64::from(FACE_BEAUTY_SIGMA).powi(2),
        );
    }

    let weighted_rows = calibration
        .iter()
        .filter(|sample| sample.embedding.len() == dim)
        .map(|sample| {
            let features = DVector::from_iterator(
                dim,
                sample
                    .embedding
                    .iter()
                    .zip(feature_mean.iter())
                    .map(|(value, mean)| f64::from(*value) - mean),
            );
            let score = weights.dot(&features);
            let weight = sample.precision_weight();
            (score, f64::from(sample.beauty.mean), weight)
        })
        .collect::<Vec<_>>();

    let weight_mass = weighted_rows
        .iter()
        .map(|(_, _, weight)| *weight)
        .sum::<f64>();
    if !weight_mass.is_finite() || weight_mass <= 0.0 {
        return (
            FACE_BEAUTY_BETA,
            f64::from(FACE_BEAUTY_MEAN),
            f64::from(FACE_BEAUTY_SIGMA).powi(2),
        );
    }

    let a00 = weighted_rows
        .iter()
        .map(|(score, _, weight)| weight * score * score)
        .sum::<f64>()
        + ORACLE_CALIBRATION_RIDGE;
    let a01 = weighted_rows
        .iter()
        .map(|(score, _, weight)| weight * score)
        .sum::<f64>();
    let a11 = weight_mass + ORACLE_CALIBRATION_RIDGE;
    let b0 = weighted_rows
        .iter()
        .map(|(score, target, weight)| weight * score * target)
        .sum::<f64>();
    let b1 = weighted_rows
        .iter()
        .map(|(_, target, weight)| weight * target)
        .sum::<f64>();
    let det = a00.mul_add(a11, -(a01 * a01));
    let (scale, bias) = if det.abs() <= 1e-9 {
        (FACE_BEAUTY_BETA, f64::from(FACE_BEAUTY_MEAN))
    } else {
        ((b0 * a11 - b1 * a01) / det, (a00 * b1 - a01 * b0) / det)
    };
    let residual_mass = weighted_rows
        .iter()
        .map(|(score, target, weight)| {
            let residual = target - (bias + scale * score);
            weight * residual * residual
        })
        .sum::<f64>();
    let dof = (weight_mass - 1.0).max(1.0);
    let noise_variance = (residual_mass / dof).max(ORACLE_NOISE_FLOOR);
    (scale, bias, noise_variance)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weng_lin_winner_gains_and_uncertainty_shrinks() {
        let winner = FaceBeauty::newborn();
        let loser = FaceBeauty::newborn();
        let (winner, loser) = rate_face_win(winner, loser);
        assert!(winner.mean > FACE_BEAUTY_MEAN);
        assert!(loser.mean < FACE_BEAUTY_MEAN);
        assert!(winner.sigma < FACE_BEAUTY_SIGMA);
        assert!(loser.sigma < FACE_BEAUTY_SIGMA);
    }

    #[test]
    fn oracle_separates_arcface_duel_clusters() {
        let mut training = FaceOracleTrainingData::default();
        for i in 0..48 {
            training.duels.push(FaceOracleDuelSample {
                winner_embedding: vec![1.0, 0.2 + i as f32 * 0.01, 0.1],
                loser_embedding: vec![-1.0, -0.2 - i as f32 * 0.01, -0.1],
            });
        }
        for i in 0..24 {
            training.calibration.push(FaceOracleCalibrationSample {
                embedding: vec![1.0, 0.1 + i as f32 * 0.01, 0.1],
                beauty: FaceBeauty::forge(1780.0, 80.0),
                compare_count: 8,
            });
            training.calibration.push(FaceOracleCalibrationSample {
                embedding: vec![-1.0, -0.1 - i as f32 * 0.01, -0.1],
                beauty: FaceBeauty::forge(1220.0, 80.0),
                compare_count: 8,
            });
        }
        let oracle = FaceOracle::train(&training).expect("oracle should train");
        let hot = oracle.predict(&[1.0, 0.0, 0.0]).expect("hot prediction");
        let not = oracle.predict(&[-1.0, 0.0, 0.0]).expect("not prediction");
        assert!(hot.mean > not.mean, "hot={hot:?} not={not:?}");
        assert!(hot.sigma > 0.0);
        assert!(hot.sigma < 400.0, "hot={hot:?}");
    }

    #[test]
    fn oracle_refuses_insufficient_duels() {
        let training = FaceOracleTrainingData {
            duels: (0..8)
                .map(|i| FaceOracleDuelSample {
                    winner_embedding: vec![1.0, i as f32],
                    loser_embedding: vec![-1.0, -(i as f32)],
                })
                .collect(),
            calibration: Vec::new(),
        };
        assert!(FaceOracle::train(&training).is_none());
    }
}
