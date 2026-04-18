use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::model::{AssetId, LATENT_DIM, RemoteItemId};

const SEMANTIC_PRIOR_EPSILON: f32 = 1e-6;
const SEMANTIC_PRIOR_SCALE: f32 = 0.45;

#[derive(Debug, Clone)]
pub(super) struct SemanticPriorBasis {
    mean: Vec<f32>,
    axes: Vec<Vec<f32>>,
    scales: Vec<f32>,
}

impl SemanticPriorBasis {
    pub(super) fn fit(
        local_embeddings: &HashMap<AssetId, Vec<f32>>,
        external_embeddings: &HashMap<RemoteItemId, Vec<f32>>,
    ) -> Option<Self> {
        let dim = local_embeddings
            .values()
            .chain(external_embeddings.values())
            .map(Vec::len)
            .find(|dimension| *dimension > 0)?;
        let samples = local_embeddings
            .values()
            .chain(external_embeddings.values())
            .filter(|embedding| {
                embedding.len() == dim && embedding.iter().all(|value| value.is_finite())
            })
            .map(|embedding| normalize_embedding(embedding))
            .collect::<Vec<_>>();
        if samples.len() < 2 {
            return None;
        }

        let data = DMatrix::from_fn(samples.len(), dim, |row, column| samples[row][column]);
        let mean = mean_columns(&data);
        let centered = DMatrix::from_fn(samples.len(), dim, |row, column| {
            data[(row, column)] - mean[column]
        });
        let decomposition = centered.svd(false, true);
        let basis = decomposition.v_t?;
        let axis_count = LATENT_DIM
            .min(dim)
            .min(decomposition.singular_values.len())
            .min(basis.nrows());
        if axis_count == 0 {
            return None;
        }

        let sample_scale = (samples.len() as f32).sqrt().max(1.0);
        let axes = (0..axis_count)
            .map(|axis| canonicalize_axis(basis.row(axis).iter().copied().collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let scales = decomposition
            .singular_values
            .iter()
            .take(axis_count)
            .map(|value| (value / sample_scale).max(SEMANTIC_PRIOR_EPSILON))
            .collect::<Vec<_>>();

        Some(Self { mean, axes, scales })
    }

    pub(super) fn project(&self, embedding: &[f32]) -> [f32; LATENT_DIM] {
        let mut projected = [0.0; LATENT_DIM];
        if embedding.len() != self.mean.len() || embedding.iter().any(|value| !value.is_finite()) {
            return projected;
        }
        let normalized = normalize_embedding(embedding);
        let centered = normalized
            .iter()
            .zip(self.mean.iter())
            .map(|(value, mean)| value - mean)
            .collect::<Vec<_>>();
        for (axis, slot) in projected.iter_mut().enumerate().take(self.axes.len()) {
            let numerator = self.axes[axis]
                .iter()
                .zip(centered.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            *slot = (numerator / self.scales[axis]) * SEMANTIC_PRIOR_SCALE;
        }
        projected
    }
}

fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(SEMANTIC_PRIOR_EPSILON);
    embedding.iter().map(|value| value / norm).collect()
}

fn mean_columns(data: &DMatrix<f32>) -> Vec<f32> {
    (0..data.ncols())
        .map(|column| {
            (0..data.nrows())
                .map(|row| data[(row, column)])
                .sum::<f32>()
                / data.nrows() as f32
        })
        .collect()
}

fn canonicalize_axis(mut axis: Vec<f32>) -> Vec<f32> {
    let pivot = axis
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.abs().total_cmp(&rhs.abs()))
        .map(|(_, value)| *value)
        .unwrap_or_default();
    if pivot < 0.0 {
        for value in &mut axis {
            *value = -*value;
        }
    }
    axis
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::SemanticPriorBasis;
    use crate::model::{AssetId, RemoteItemId};

    #[test]
    fn semantic_prior_basis_projects_nonzero_latent_axes() {
        let local = HashMap::from([
            (AssetId("a".to_owned()), vec![1.0, 0.0, 0.0, 0.0]),
            (AssetId("b".to_owned()), vec![0.0, 1.0, 0.0, 0.0]),
        ]);
        let remote = HashMap::from([(RemoteItemId(1), vec![0.0, 0.0, 1.0, 0.0])]);

        let basis = SemanticPriorBasis::fit(&local, &remote).expect("fit basis");
        let left = basis.project(local.get(&AssetId("a".to_owned())).expect("left"));
        let right = basis.project(local.get(&AssetId("b".to_owned())).expect("right"));

        assert!(left.iter().any(|value| value.abs() > 1e-5));
        assert!(right.iter().any(|value| value.abs() > 1e-5));
        assert_ne!(left, right);
    }
}
