use super::*;

pub(super) fn flat_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let image = RgbImage::from_pixel(width, height, Rgb(rgb));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("encode png");
    bytes
}

pub(super) fn test_root(name: &str) -> std::path::PathBuf {
    let root = env::temp_dir().join(format!("picmash-{name}-{}", Ulid::new()));
    std::fs::create_dir_all(&root).expect("create temp root");
    root
}

pub(super) fn assert_close(lhs: f32, rhs: f32) {
    assert!(
        (lhs - rhs).abs() <= 1e-5,
        "expected {lhs} ~= {rhs}, delta={}",
        (lhs - rhs).abs()
    );
}

pub(super) fn assert_slice_close(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len(), "slice length mismatch");
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert_close(*left, *right);
    }
}

pub(super) fn hierarchical_test_canonical_mean(asset: &HierarchicalAssetPosterior) -> f32 {
    asset.baseline_mean
        + asset
            .technical_mean
            .map_or(0.0, |technical| HIERARCHICAL_TECH_WEIGHT * technical)
}

pub(super) fn hierarchical_test_canonical_variance(asset: &HierarchicalAssetPosterior) -> f32 {
    asset.baseline_variance
        + asset
            .technical_variance
            .map_or(0.0, |variance| HIERARCHICAL_TECH_WEIGHT.powi(2) * variance)
}

pub(super) fn hierarchical_test_sync_asset_record(
    asset: &mut crate::model::AssetRecord,
    posterior: &HierarchicalAssetPosterior,
) {
    asset.alpha = hierarchical_test_canonical_mean(posterior);
    asset.coords = posterior.mood_loading_mean;
}

pub(super) fn hierarchical_test_utility_mean(
    asset: &crate::model::AssetRecord,
    posterior: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
    exact_offset: f32,
    hearted: bool,
) -> f32 {
    hierarchical_test_canonical_mean(posterior)
        + dot(&posterior.mood_loading_mean, &session.semantic_mood_mean)
        + posterior
            .technical_mean
            .map(|_| {
                posterior
                    .vibe_mean
                    .iter()
                    .zip(session.vibe_mean.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>()
            })
            .unwrap_or_default()
        + exact_offset
        + legacy_heart_bias(asset, hearted)
}

pub(super) fn hierarchical_test_utility_variance(
    asset: &HierarchicalAssetPosterior,
    session: &HierarchicalSessionPosterior,
) -> f32 {
    let semantic = asset
        .mood_loading_mean
        .iter()
        .zip(asset.mood_loading_variance.iter())
        .zip(
            session
                .semantic_mood_mean
                .iter()
                .zip(session.semantic_mood_variance.iter()),
        )
        .map(|((loading_mean, loading_var), (mood_mean, mood_var))| {
            mood_mean.powi(2) * *loading_var
                + loading_mean.powi(2) * *mood_var
                + loading_var * mood_var
        })
        .sum::<f32>();
    let vibe = asset
        .technical_mean
        .map(|_| {
            asset
                .vibe_mean
                .iter()
                .zip(asset.vibe_variance.iter())
                .zip(session.vibe_mean.iter().zip(session.vibe_variance.iter()))
                .map(|((asset_mean, asset_var), (session_mean, session_var))| {
                    session_mean.powi(2) * *asset_var
                        + asset_mean.powi(2) * *session_var
                        + asset_var * session_var
                })
                .sum::<f32>()
        })
        .unwrap_or_default();
    hierarchical_test_canonical_variance(asset) + semantic + vibe
}

pub(super) fn hierarchical_test_asset_cache(
    posterior: &HierarchicalAssetPosterior,
) -> HierarchicalAssetQualityCacheV1 {
    HierarchicalAssetQualityCacheV1 {
        baseline_mean: posterior.baseline_mean,
        baseline_variance: posterior.baseline_variance,
        canonical_mean: hierarchical_test_canonical_mean(posterior),
        canonical_variance: hierarchical_test_canonical_variance(posterior),
        mood_loading_mean: posterior.mood_loading_mean.to_vec(),
        mood_loading_variance: posterior.mood_loading_variance.to_vec(),
        technical_mean: posterior.technical_mean,
        technical_variance: posterior.technical_variance,
        vibe_mean: posterior.vibe_mean.to_vec(),
        vibe_variance: posterior.vibe_variance.to_vec(),
    }
}

pub(super) fn hierarchical_test_session_cache(
    posterior: &HierarchicalSessionPosterior,
) -> HierarchicalSessionQualityCacheV1 {
    HierarchicalSessionQualityCacheV1 {
        semantic_mood_mean: posterior.semantic_mood_mean.to_vec(),
        semantic_mood_variance: posterior.semantic_mood_variance.to_vec(),
        vibe_mean: posterior.vibe_mean.to_vec(),
        vibe_variance: posterior.vibe_variance.to_vec(),
        frontier_mean: posterior.frontier_mean,
        frontier_variance: posterior.frontier_variance,
    }
}

pub(super) fn remote_stream(thread_no: i64, image_count: u32) -> RemoteStreamSnapshot {
    RemoteStreamSnapshot {
        thread_no,
        title: format!("thread {thread_no}"),
        semantic_slug: String::new(),
        last_modified: 1_700_000_000 + thread_no,
        reply_count: image_count,
        image_count,
        items: Vec::new(),
    }
}

pub(super) fn remote_item(thread_no: i64, post_no: i64, md5: Option<&str>) -> RemoteItemSnapshot {
    RemoteItemSnapshot {
        thread_no,
        post_no,
        title: format!("post {post_no}"),
        image_url: format!("https://example.invalid/{thread_no}/{post_no}.png"),
        thumb_url: format!("https://example.invalid/{thread_no}/{post_no}s.jpg"),
        ext: ".png".to_owned(),
        md5: md5.map(str::to_owned),
        width: 256,
        height: 256,
        file_size: 1024,
        materialized_path: None,
    }
}
