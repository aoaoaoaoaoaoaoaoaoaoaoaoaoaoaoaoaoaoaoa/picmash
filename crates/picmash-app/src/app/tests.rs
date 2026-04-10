use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng};

use crate::config::{
    FourChanBoardSource, FourChanContentConfig, FourChanHarvestConfig, ImportPolicy,
    LocalDirectorySource, RemoteImageFilterConfig, SourceConfig, UpstreamSource,
};

use super::{
    ReadyTargetProfile, SourceReadyFrontier,
    external::{choose_pair_source, remote_source_arena_bonuses},
};

#[test]
fn remote_ready_frontier_has_hard_cap() {
    let frontier =
        SourceReadyFrontier::new(96, HashMap::new(), 512, ReadyTargetProfile::for_remote());
    assert!(frontier.source_saturated());
}

#[test]
fn ready_frontier_saturates_streams_independently() {
    let mut frontier =
        SourceReadyFrontier::new(0, HashMap::new(), 32, ReadyTargetProfile::for_remote());
    assert!(!frontier.stream_saturated(42));
    frontier.note_ready(42);
    frontier.note_ready(42);
    assert!(!frontier.stream_saturated(42));
    frontier.note_ready(42);
    assert!(frontier.stream_saturated(42));
}

#[test]
fn choose_pair_source_respects_probability_boundaries() {
    let mut rng = StdRng::seed_from_u64(7);
    assert_eq!(
        choose_pair_source(&mut rng, 0.0, Some("local"), Some("remote")),
        Some("local")
    );
    assert_eq!(
        choose_pair_source(&mut rng, 1.0, Some("local"), Some("remote")),
        Some("remote")
    );
    assert_eq!(
        choose_pair_source(&mut rng, 0.4, Some("local"), None),
        Some("local")
    );
    assert_eq!(
        choose_pair_source(&mut rng, 0.4, None, Some("remote")),
        Some("remote")
    );
    assert_eq!(
        choose_pair_source::<_, &str>(&mut rng, 0.4, None, None),
        None
    );
}

#[test]
fn local_directory_sources_do_not_get_fake_hot_thread_bonuses() {
    let now = 1_775_765_480;
    let local = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 60,
        upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
            root: "/tmp/picmash-local-remote".into(),
            recurse: true,
            filters: RemoteImageFilterConfig::default(),
        }),
    };
    let network = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 60,
        upstream: UpstreamSource::FourChanBoard(FourChanBoardSource {
            board: "s".to_owned(),
            content: FourChanContentConfig::default(),
            harvest: FourChanHarvestConfig::default(),
            filters: RemoteImageFilterConfig::default(),
        }),
    };

    let local_bonuses = remote_source_arena_bonuses(&local, now, now, 3_788);
    assert_eq!(local_bonuses.stream_size, 0.0);
    assert_eq!(local_bonuses.freshness, 0.0);

    let network_bonuses = remote_source_arena_bonuses(&network, now, now, 300);
    assert!(network_bonuses.stream_size > 0.0);
    assert!(network_bonuses.freshness > 0.0);
}
