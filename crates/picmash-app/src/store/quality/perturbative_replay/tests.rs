use std::{collections::HashMap, path::PathBuf};

use super::*;

fn synth_asset(
    id: &str,
    baseline_mean: f32,
    technical_mean: Option<f32>,
) -> PerturbativeReplayAssetState {
    let domain_label = if technical_mean.is_some() {
        AssetDomainLabel::Real
    } else {
        AssetDomainLabel::Anime
    };
    let mut asset = PerturbativeReplayAssetState {
        asset: AssetRecord {
            id: AssetId(id.to_owned()),
            path: PathBuf::new(),
            width: 0,
            height: 0,
            alpha: 0.0,
            coords: [0.0; crate::model::LATENT_DIM],
            rotation_quarters: 0,
            compare_count: 0,
            win_count: 0,
            heart_count: 0,
            is_hearted: false,
            hidden: false,
        },
        baseline_mean,
        baseline_variance: 1.0,
        perturbation_basis: [0.0; crate::quality::PERTURBATIVE_DIM],
        technical_mean,
        technical_variance: technical_mean.map(|_| 1.0),
        face: None,
        domain_label,
    };
    sync_perturbative_asset_record(&mut asset, PerturbativeHyperParamsV3::default());
    asset
}

fn synth_external(
    baseline_mean: f32,
    technical_mean: Option<f32>,
) -> PerturbativeReplayExternalState {
    let domain_label = if technical_mean.is_some() {
        AssetDomainLabel::Real
    } else {
        AssetDomainLabel::Anime
    };
    PerturbativeReplayExternalState {
        baseline_mean,
        baseline_variance: 1.0,
        perturbation_basis: [0.0; crate::quality::PERTURBATIVE_DIM],
        technical_mean,
        technical_variance: technical_mean.map(|_| 1.0),
        domain_label,
    }
}

#[test]
fn gauge_fix_perturbative_technical_recenters_three_d_branch() {
    let mut state = PerturbativeReplayState {
        assets: HashMap::from([
            (AssetId("a".to_owned()), synth_asset("a", 1.0, Some(-2.0))),
            (AssetId("b".to_owned()), synth_asset("b", -1.0, Some(0.0))),
            (AssetId("c".to_owned()), synth_asset("c", 0.5, None)),
        ]),
        external_items: HashMap::from([
            (RemoteItemId(1), synth_external(0.25, Some(-1.0))),
            (RemoteItemId(2), synth_external(-0.5, None)),
        ]),
        subjects: HashMap::new(),
        sessions: HashMap::new(),
        comparison_events: 0,
        nudge_events: 0,
        heart_events: 0,
        external_events: 0,
        max_comparison_id: 0,
        max_nudge_id: 0,
        max_heart_id: 0,
        max_external_id: 0,
    };

    let hyper = PerturbativeHyperParamsV3::default();
    let before = {
        let asset = state.assets.get(&AssetId("a".to_owned())).expect("asset a");
        perturbative_asset_canonical_mean(asset, hyper)
    };
    gauge_fix_perturbative_technical(&mut state, hyper);

    let technical_means = state
        .assets
        .values()
        .filter_map(|asset| asset.technical_mean)
        .chain(
            state
                .external_items
                .values()
                .filter_map(|item| item.technical_mean),
        )
        .collect::<Vec<_>>();
    let centered = technical_means.iter().copied().sum::<f32>() / technical_means.len() as f32;
    assert!(
        centered.abs() < 1e-6,
        "technical branch mean should be zero, got {centered}"
    );

    let after = {
        let asset = state.assets.get(&AssetId("a".to_owned())).expect("asset a");
        perturbative_asset_canonical_mean(asset, hyper)
    };
    assert!(
        (before - after).abs() < 1e-5,
        "canonical utility should be invariant under technical gauge-fixing: before={before}, after={after}"
    );
}
