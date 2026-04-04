use rand::Rng;

use super::{AssetRecord, LATENT_DIM, SessionRecord};

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
