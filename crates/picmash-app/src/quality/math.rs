use crate::{
    facemash::FaceBeauty,
    model::{LATENT_DIM, sigmoid},
};

use super::{
    HIERARCHICAL_FACE_CENTER, HIERARCHICAL_FACE_SCALE, HIERARCHICAL_FACE_WEIGHT,
    HIERARCHICAL_MIN_VARIANCE, PERTURBATIVE_DIM, PerturbativeHyperParamsV3,
    PerturbativeSessionPosterior,
};

#[must_use]
pub fn hierarchical_face_latent_mean(face: FaceBeauty) -> f32 {
    (face.mean - HIERARCHICAL_FACE_CENTER) / HIERARCHICAL_FACE_SCALE
}

#[must_use]
pub fn hierarchical_face_latent_variance(face: FaceBeauty) -> f32 {
    (face.sigma / HIERARCHICAL_FACE_SCALE)
        .powi(2)
        .max(HIERARCHICAL_MIN_VARIANCE)
}

#[must_use]
pub const fn hierarchical_face_backflow_coeff() -> f32 {
    HIERARCHICAL_FACE_WEIGHT / HIERARCHICAL_FACE_SCALE
}

#[derive(Debug, Clone, Copy)]
pub struct GaussianMomentMatch {
    pub c: f32,
    pub v: f32,
    pub w: f32,
}

pub fn gaussian_duel_moment_match(
    mean_delta: f32,
    variance_delta: f32,
    outcome: f32,
    beta: f32,
) -> Option<GaussianMomentMatch> {
    let performance_variance = (variance_delta + beta * beta).max(HIERARCHICAL_MIN_VARIANCE);
    let c = performance_variance.sqrt();
    let t = (outcome * mean_delta / c).clamp(-8.0, 8.0);
    let cdf = standard_normal_cdf(t).max(1e-8);
    let pdf = standard_normal_pdf(t);
    let v = pdf / cdf;
    let w = (v * (v + t)).clamp(0.0, 0.98);
    Some(GaussianMomentMatch { c, v, w })
}

#[must_use]
pub fn posterior_accept_probability(
    utility_mean: f32,
    utility_variance: f32,
    frontier_mean: f32,
    frontier_variance: f32,
    beta: f32,
) -> f32 {
    let performance_variance =
        (utility_variance + frontier_variance + beta * beta).max(HIERARCHICAL_MIN_VARIANCE);
    standard_normal_cdf((utility_mean - frontier_mean) / performance_variance.sqrt())
}

pub fn diagonal_adf_update(
    mean: &mut f32,
    variance: &mut f32,
    jacobian: f32,
    outcome: f32,
    moments: GaussianMomentMatch,
) {
    if jacobian.abs() <= f32::EPSILON {
        return;
    }
    let variance_before = (*variance).max(HIERARCHICAL_MIN_VARIANCE);
    *mean += outcome * (variance_before * jacobian / moments.c) * moments.v;
    let shrink = (variance_before * variance_before * jacobian * jacobian
        / (moments.c * moments.c))
        * moments.w;
    *variance = (variance_before - shrink).max(HIERARCHICAL_MIN_VARIANCE);
}

#[must_use]
pub fn perturbative_importance(raw: f32) -> f32 {
    if raw >= 20.0 {
        raw
    } else if raw <= -20.0 {
        raw.exp()
    } else {
        raw.exp().ln_1p()
    }
}

#[must_use]
pub fn perturbative_importance_slope(raw: f32) -> f32 {
    sigmoid(raw)
}

#[must_use]
pub fn perturbative_projection_mean(
    basis: &[f32; PERTURBATIVE_DIM],
    session: &PerturbativeSessionPosterior,
    hyper: PerturbativeHyperParamsV3,
) -> f32 {
    let importance = perturbative_importance(session.importance_raw_mean);
    importance * perturbative_projection_dot(basis, session, hyper)
}

#[must_use]
pub fn perturbative_projection_variance(
    basis: &[f32; PERTURBATIVE_DIM],
    session: &PerturbativeSessionPosterior,
    hyper: PerturbativeHyperParamsV3,
) -> f32 {
    let importance = perturbative_importance(session.importance_raw_mean);
    let importance_slope = perturbative_importance_slope(session.importance_raw_mean);
    let centered = perturbative_projection_dot(basis, session, hyper);
    let semantic_weight_variance = basis[..LATENT_DIM]
        .iter()
        .zip(session.semantic_weight_variance().iter())
        .map(|(value, variance)| value * value * variance)
        .sum::<f32>();
    let vibe_scale = hyper.vibe_scale_real;
    let vibe_weight_variance = basis[LATENT_DIM..]
        .iter()
        .zip(session.vibe_weight_variance().iter())
        .map(|(value, variance)| vibe_scale.powi(2) * value * value * variance)
        .sum::<f32>();
    let weight_variance = semantic_weight_variance + vibe_weight_variance;
    (importance.powi(2) * weight_variance
        + importance_slope.powi(2) * centered.powi(2) * session.importance_raw_variance)
        .max(HIERARCHICAL_MIN_VARIANCE)
}

fn perturbative_projection_dot(
    basis: &[f32; PERTURBATIVE_DIM],
    session: &PerturbativeSessionPosterior,
    hyper: PerturbativeHyperParamsV3,
) -> f32 {
    let semantic = basis[..LATENT_DIM]
        .iter()
        .zip(session.semantic_weight_mean().iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    let vibe = basis[LATENT_DIM..]
        .iter()
        .zip(session.vibe_weight_mean().iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    semantic + hyper.vibe_scale_real * vibe
}

pub fn standard_normal_pdf(value: f32) -> f32 {
    const INV_SQRT_TWO_PI: f32 = 0.398_942_3;
    INV_SQRT_TWO_PI * (-0.5 * value * value).exp()
}

pub fn standard_normal_cdf(value: f32) -> f32 {
    let x = value.abs();
    let k = 1.0 / (1.0 + 0.231_641_9 * x);
    let poly = ((((1.330_274_5 * k - 1.821_255_9) * k + 1.781_478) * k - 0.356_563_78) * k
        + 0.319_381_53)
        * k;
    let approx = 1.0 - standard_normal_pdf(x) * poly;
    if value >= 0.0 { approx } else { 1.0 - approx }
}
