use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng};

use super::{ReadyTargetProfile, SourceReadyFrontier, external::choose_pair_source};

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
