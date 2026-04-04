use rand::{Rng, rng};

const LEAF_CAP: usize = 8;

#[derive(Debug, Clone)]
struct VpPoint<I> {
    id: I,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
enum VpNode<I> {
    Leaf {
        points: Vec<VpPoint<I>>,
    },
    Branch {
        vantage: VpPoint<I>,
        mu: f32,
        near: Box<Self>,
        far: Box<Self>,
    },
}

#[derive(Debug, Clone)]
pub struct VpTree<I> {
    root: Option<VpNode<I>>,
    len: usize,
}

impl<I: Copy> VpTree<I> {
    /// Build a VP-tree from (id, L2-normalized embedding) pairs.
    pub fn forge(points: Vec<(I, Vec<f32>)>) -> Self {
        let len = points.len();
        let vp_points = points
            .into_iter()
            .map(|(id, embedding)| VpPoint { id, embedding })
            .collect::<Vec<_>>();
        let root = erect(vp_points);
        Self { root, len }
    }

    /// Find all points within Euclidean `radius` of `query`.
    pub fn ransack(&self, query: &[f32], radius: f32) -> Vec<(I, f32)> {
        let mut hits = Vec::new();
        if let Some(root) = &self.root {
            probe(root, query, radius, &mut hits);
        }
        hits
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn erect<I: Copy>(mut points: Vec<VpPoint<I>>) -> Option<VpNode<I>> {
    if points.is_empty() {
        return None;
    }
    if points.len() <= LEAF_CAP {
        return Some(VpNode::Leaf { points });
    }
    let pivot = rng().random_range(0..points.len());
    let vantage = points.swap_remove(pivot);
    let mut dists: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, l2_dist(&vantage.embedding, &p.embedding)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    let median_idx = dists.len() / 2;
    let mu = dists.get(median_idx).map_or(0.0, |&(_, d)| d);

    let far_indices: Vec<usize> = dists[median_idx..].iter().map(|&(i, _)| i).collect();
    let mut far_set = vec![false; points.len()];
    for &i in &far_indices {
        far_set[i] = true;
    }

    let mut near_points = Vec::new();
    let mut far_points = Vec::new();
    for (i, point) in points.into_iter().enumerate() {
        if far_set[i] {
            far_points.push(point);
        } else {
            near_points.push(point);
        }
    }

    Some(VpNode::Branch {
        vantage,
        mu,
        near: Box::new(erect(near_points).unwrap_or(VpNode::Leaf { points: vec![] })),
        far: Box::new(erect(far_points).unwrap_or(VpNode::Leaf { points: vec![] })),
    })
}

fn probe<I: Copy>(node: &VpNode<I>, query: &[f32], radius: f32, hits: &mut Vec<(I, f32)>) {
    match node {
        VpNode::Leaf { points } => {
            for p in points {
                let d = l2_dist(query, &p.embedding);
                if d <= radius {
                    hits.push((p.id, d));
                }
            }
        }
        VpNode::Branch {
            vantage,
            mu,
            near,
            far,
        } => {
            let d = l2_dist(query, &vantage.embedding);
            if d <= radius {
                hits.push((vantage.id, d));
            }
            if d - radius <= *mu {
                probe(near, query, radius, hits);
            }
            if d + radius > *mu {
                probe(far, query, radius, hits);
            }
        }
    }
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RemoteItemId;

    fn point(id: i64, coords: &[f32]) -> (RemoteItemId, Vec<f32>) {
        (RemoteItemId(id), coords.to_vec())
    }

    #[test]
    fn empty_tree_ransacks_nothing() {
        let tree = VpTree::<RemoteItemId>::forge(vec![]);
        assert!(tree.is_empty());
        assert!(tree.ransack(&[0.0, 0.0], 1.0).is_empty());
    }

    #[test]
    fn exact_match_within_zero_radius() {
        let tree = VpTree::forge(vec![point(1, &[1.0, 0.0]), point(2, &[0.0, 1.0])]);
        let hits = tree.ransack(&[1.0, 0.0], 0.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, RemoteItemId(1));
    }

    #[test]
    fn radius_query_finds_neighbors() {
        let points: Vec<_> = (0..100)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 100.0;
                point(i, &[angle.cos(), angle.sin()])
            })
            .collect();
        let tree = VpTree::forge(points);
        assert_eq!(tree.len(), 100);

        let hits = tree.ransack(&[1.0, 0.0], 0.15);
        assert!(!hits.is_empty());
        for (_, d) in &hits {
            assert!(*d <= 0.15);
        }
    }

    #[test]
    fn all_points_within_huge_radius() {
        let points = vec![
            point(1, &[0.0, 0.0]),
            point(2, &[1.0, 0.0]),
            point(3, &[0.0, 1.0]),
        ];
        let tree = VpTree::forge(points);
        let hits = tree.ransack(&[0.5, 0.5], 100.0);
        assert_eq!(hits.len(), 3);
    }
}
