use faer::Mat;
use manifolds_rs::{UmapParams, umap};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

use super::{MAP_DIM, SIMILARITY_DIM};

const RAW_LAYOUT_DIM: usize = 32;
const LAYOUT_MARGIN: f32 = 0.08;
const LAYOUT_SPAN: f32 = 1.0 - (LAYOUT_MARGIN * 2.0);
const PCA_EPSILON: f32 = 1e-6;
const LEARNED_LAYOUT_MAX_SMACOF_POINTS: usize = 900;
const LEARNED_LAYOUT_MAX_ITERATIONS: usize = 32;
const LEARNED_LAYOUT_TOLERANCE: f32 = 1e-4;
const LAYOUT_DISTANCE_EPSILON: f32 = 1e-5;

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

pub fn prepare_raw_layout_space(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    pca_whiten_points(points, RAW_LAYOUT_DIM)
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

fn sorted_eigen_indices(values: &DVector<f32>) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|lhs, rhs| values[*rhs].total_cmp(&values[*lhs]));
    order
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

#[cfg(test)]
mod tests {
    use super::{
        MAP_DIM, learned_reduce_points, normalize_layout_points, pca_reduce_points,
        prepare_raw_layout_space,
    };

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
}
