#[test]
fn locking_subsource_refills_beyond_generic_remote_warm_floor() {
    let _guard = test_guard();
    let root = test_root("subsource-lock-refill");
    let corpus_root = root.join("corpus");
    let source_root = root.join("source");
    let config_root = root.join("config");
    let app_data_root = root.join("xdg-data");
    let app_cache_root = root.join("xdg-cache");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
    std::fs::create_dir_all(&config_root).expect("create config root");
    std::fs::create_dir_all(&app_data_root).expect("create data root");
    std::fs::create_dir_all(&app_cache_root).expect("create cache root");

    solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
    solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
    for index in 0..20 {
        patterned_png(
            &source_root.join("a").join(format!("remote-{index:02}.png")),
            index as u8,
        );
    }

    let mut config = app_config_with_source_mix(0.0);
    let source = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 3600,
        upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
            root: source_root,
            recurse: true,
            filters: RemoteImageFilterConfig {
                min_shortest_edge: 0,
                ..RemoteImageFilterConfig::default()
            },
        }),
    };
    let source_key = source.source_key();
    config.sources = vec![source];
    let config_path = config_root.join("config.toml");
    let config_digest = config.write(&config_path).expect("write config");
    let app_paths = AppBootPaths {
        db_path: app_data_root.join("picmash.sqlite3"),
        model_cache_root: app_cache_root.clone(),
        cache_root: app_cache_root.join("renditions"),
        source_cache_root: app_cache_root.join("sources"),
    };

    let state =
        AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
            .expect("boot app state");
    state.schedule_corpus_ingest();
    drain_maintenance(&state);
    state
        .refresh_external_sources_if_due(true)
        .expect("harvest locked source fixture");

    let locked_item_id = source_remote_items(&state, &source_key)[0];
    let locked_stream_id = state
        .read_store()
        .expect("open read store for locked stream")
        .remote_item(locked_item_id)
        .expect("load locked remote item")
        .expect("locked remote item present")
        .stream_id;
    let initial_counts = state
        .read_store()
        .expect("open read store for initial frontier counts")
        .external_stream_frontier_counts(
            &source_key,
            locked_stream_id,
            state.embedder.model_name(),
        )
        .expect("load initial frontier counts")
        .expect("initial frontier counts present");
    assert!(
        initial_counts.ready_items < EXTERNAL_LOCKED_STREAM_READY_TARGET,
        "generic remote warm floor should start below the lock-specific refill target"
    );

    state
        .set_external_subsource_lock(locked_item_id, true)
        .expect("lock subsource");
    state
        .refresh_external_sources_if_due(false)
        .expect("schedule lock-triggered refresh");
    drain_maintenance(&state);

    let replenished = state
        .read_store()
        .expect("open read store for replenished frontier counts")
        .external_stream_frontier_counts(
            &source_key,
            locked_stream_id,
            state.embedder.model_name(),
        )
        .expect("load replenished frontier counts")
        .expect("replenished frontier counts present");
    assert_eq!(
        replenished.ready_items, EXTERNAL_LOCKED_STREAM_READY_TARGET,
        "locking a stream should deepen the ready frontier immediately"
    );
}

#[test]
fn locked_refill_rebuilds_frontier_from_stored_stream_snapshots() {
    let _guard = test_guard();
    let root = test_root("subsource-lock-store-refill");
    let corpus_root = root.join("corpus");
    let source_root = root.join("source");
    let config_root = root.join("config");
    let app_data_root = root.join("xdg-data");
    let app_cache_root = root.join("xdg-cache");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
    std::fs::create_dir_all(&config_root).expect("create config root");
    std::fs::create_dir_all(&app_data_root).expect("create data root");
    std::fs::create_dir_all(&app_cache_root).expect("create cache root");

    solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
    solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
    for index in 0..20 {
        patterned_png(
            &source_root.join("a").join(format!("remote-{index:02}.png")),
            index as u8,
        );
    }

    let mut config = app_config_with_source_mix(0.0);
    let source = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 3600,
        upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
            root: source_root,
            recurse: true,
            filters: RemoteImageFilterConfig {
                min_shortest_edge: 0,
                ..RemoteImageFilterConfig::default()
            },
        }),
    };
    let source_key = source.source_key();
    config.sources = vec![source];
    let config_path = config_root.join("config.toml");
    let config_digest = config.write(&config_path).expect("write config");
    let app_paths = AppBootPaths {
        db_path: app_data_root.join("picmash.sqlite3"),
        model_cache_root: app_cache_root.clone(),
        cache_root: app_cache_root.join("renditions"),
        source_cache_root: app_cache_root.join("sources"),
    };

    let state =
        AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
            .expect("boot app state");
    state.schedule_corpus_ingest();
    drain_maintenance(&state);
    state
        .refresh_external_sources_if_due(true)
        .expect("harvest locked source fixture");

    let locked_item_id = source_remote_items(&state, &source_key)[0];
    let locked_stream_id = state
        .read_store()
        .expect("open read store for locked stream")
        .remote_item(locked_item_id)
        .expect("load locked remote item")
        .expect("locked remote item present")
        .stream_id;
    state
        .set_external_subsource_lock(locked_item_id, true)
        .expect("lock subsource");
    let lock = state
        .read_store()
        .expect("open read store for active lock")
        .session_subsource_lock(state.active.session_id)
        .expect("load active lock")
        .expect("active lock present");

    let ready_items = {
        let store = state
            .read_store()
            .expect("open read store for ready locked items");
        source_remote_items(&state, &source_key)
            .into_iter()
            .filter(|item_id| {
                store
                    .external_item_frontier_ready(*item_id, state.embedder.model_name())
                    .expect("check locked frontier readiness")
            })
            .collect::<Vec<_>>()
    };
    assert!(
        !ready_items.is_empty(),
        "fixture should start with a nonempty ready frontier"
    );
    state
        .with_write_store("withdraw_locked_store_frontier", move |store| {
            store.withdraw_external_items_from_frontier(&ready_items)
        })
        .expect("withdraw ready locked frontier");

    state
        .refill_locked_source_now(&lock)
        .expect("refill locked stream from stored snapshots");

    let replenished = state
        .read_store()
        .expect("open read store for replenished stored frontier")
        .external_stream_frontier_counts(
            &source_key,
            locked_stream_id,
            state.embedder.model_name(),
        )
        .expect("load replenished stored frontier counts")
        .expect("replenished stored frontier counts present");
    assert_eq!(
        replenished.ready_items, EXTERNAL_LOCKED_STREAM_READY_TARGET,
        "locked refill should rebuild the frontier from stored stream snapshots"
    );
}

#[test]
fn authoritative_lock_can_reuse_ready_locked_items_after_hard_refresh() {
    let _guard = test_guard();
    let root = test_root("subsource-lock-hard-refresh");
    let corpus_root = root.join("corpus");
    let source_root = root.join("source");
    let config_root = root.join("config");
    let app_data_root = root.join("xdg-data");
    let app_cache_root = root.join("xdg-cache");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
    std::fs::create_dir_all(&config_root).expect("create config root");
    std::fs::create_dir_all(&app_data_root).expect("create data root");
    std::fs::create_dir_all(&app_cache_root).expect("create cache root");

    solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
    solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
    for index in 0..20 {
        patterned_png(
            &source_root.join("a").join(format!("remote-{index:02}.png")),
            index as u8,
        );
    }

    let mut config = app_config_with_source_mix(0.0);
    let source = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 3600,
        upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
            root: source_root,
            recurse: true,
            filters: RemoteImageFilterConfig {
                min_shortest_edge: 0,
                ..RemoteImageFilterConfig::default()
            },
        }),
    };
    let source_key = source.source_key();
    config.sources = vec![source];
    let config_path = config_root.join("config.toml");
    let config_digest = config.write(&config_path).expect("write config");
    let app_paths = AppBootPaths {
        db_path: app_data_root.join("picmash.sqlite3"),
        model_cache_root: app_cache_root.clone(),
        cache_root: app_cache_root.join("renditions"),
        source_cache_root: app_cache_root.join("sources"),
    };

    let state =
        AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
            .expect("boot app state");
    state.schedule_corpus_ingest();
    drain_maintenance(&state);
    state
        .refresh_external_sources_if_due(true)
        .expect("harvest underfill fixture");

    let ready_items = {
        let store = state
            .read_store()
            .expect("open read store for ready locked items");
        source_remote_items(&state, &source_key)
            .into_iter()
            .filter(|item_id| {
                store
                    .external_item_frontier_ready(*item_id, state.embedder.model_name())
                    .expect("check locked frontier readiness")
            })
            .collect::<Vec<_>>()
    };
    let locked_item_id = ready_items[0];
    let locked_stream_id = state
        .read_store()
        .expect("open read store for original locked item")
        .remote_item(locked_item_id)
        .expect("load original locked item")
        .expect("original locked item present")
        .stream_id;
    state
        .set_external_subsource_lock(locked_item_id, true)
        .expect("lock subsource");

    let local_asset_id = state
        .read_store()
        .expect("open read store for local anchor")
        .corpus_assets(state.active.corpus_id)
        .expect("load corpus assets for local anchor")[0]
        .id
        .clone();
    for item_id in ready_items {
        let local_asset_id = local_asset_id.clone();
        state
            .with_write_store("note_temporary_locked_recent", move |store| {
                store.note_external_selected(
                    state.active.session_id,
                    state.active.corpus_id,
                    item_id,
                    &local_asset_id,
                )
            })
            .expect("mark ready locked remote as recent");
    }

    state.shatter_arena_session();
    let target = state
        .arena_target()
        .expect("authoritative target after hard refresh");
    let RedirectTarget::ArenaPair { left, right } = target else {
        panic!("expected a locked pair after hard refresh");
    };
    let reused_remote = match (&left, &right) {
        (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
        | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => *item_id,
        _ => panic!("expected a local-vs-remote pair after hard refresh"),
    };
    let reused_item = state
        .read_store()
        .expect("open read store for reused locked pair")
        .remote_item(reused_remote)
        .expect("load reused remote item")
        .expect("reused remote item present");
    assert_eq!(
        reused_item.stream_id, locked_stream_id,
        "hard refresh should keep serving the locked stream instead of bricking arena"
    );
    assert!(
        state
            .read_store()
            .expect("open read store for preserved authoritative lock")
            .session_subsource_lock(state.active.session_id)
            .expect("reload preserved authoritative lock")
            .is_some(),
        "hard refresh must preserve the lock while reusing the locked stream"
    );
}

#[test]
fn hiding_current_remote_refills_authoritative_locked_frontier_before_bricking() {
    let _guard = test_guard();
    let root = test_root("subsource-lock-hide-refill");
    let corpus_root = root.join("corpus");
    let source_root = root.join("source");
    let config_root = root.join("config");
    let app_data_root = root.join("xdg-data");
    let app_cache_root = root.join("xdg-cache");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
    std::fs::create_dir_all(&config_root).expect("create config root");
    std::fs::create_dir_all(&app_data_root).expect("create data root");
    std::fs::create_dir_all(&app_cache_root).expect("create cache root");

    solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
    solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
    for index in 0..20 {
        patterned_png(
            &source_root.join("a").join(format!("remote-{index:02}.png")),
            index as u8,
        );
    }

    let mut config = app_config_with_source_mix(0.0);
    let source = SourceConfig {
        weight: 1.0,
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: 3600,
        upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
            root: source_root,
            recurse: true,
            filters: RemoteImageFilterConfig {
                min_shortest_edge: 0,
                ..RemoteImageFilterConfig::default()
            },
        }),
    };
    let source_key = source.source_key();
    config.sources = vec![source];
    let config_path = config_root.join("config.toml");
    let config_digest = config.write(&config_path).expect("write config");
    let app_paths = AppBootPaths {
        db_path: app_data_root.join("picmash.sqlite3"),
        model_cache_root: app_cache_root.clone(),
        cache_root: app_cache_root.join("renditions"),
        source_cache_root: app_cache_root.join("sources"),
    };

    let state =
        AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
            .expect("boot app state");
    state.schedule_corpus_ingest();
    drain_maintenance(&state);
    state
        .refresh_external_sources_if_due(true)
        .expect("harvest local-directory source");

    let ready_items = {
        let store = state
            .read_store()
            .expect("open read store for ready locked items");
        source_remote_items(&state, &source_key)
            .into_iter()
            .filter(|item_id| {
                store
                    .external_item_frontier_ready(*item_id, state.embedder.model_name())
                    .expect("check locked frontier readiness")
            })
            .collect::<Vec<_>>()
    };
    let locked_item_id = ready_items[0];
    let locked_stream_id = state
        .read_store()
        .expect("open read store for locked stream")
        .remote_item(locked_item_id)
        .expect("load locked remote item")
        .expect("locked remote item present")
        .stream_id;
    state
        .set_external_subsource_lock(locked_item_id, true)
        .expect("lock subsource");

    let page = state
        .arena_page_state(None)
        .expect("arena page state under lock");
    let current = page.current.expect("current turn under lock");
    let remote_item_id = match (&current.pair().left, &current.pair().right) {
        (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
        | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => *item_id,
        _ => panic!("expected a local-vs-remote pair under lock"),
    };

    let frontier_to_withdraw = ready_items
        .into_iter()
        .filter(|item_id| *item_id != remote_item_id)
        .collect::<Vec<_>>();
    state
        .with_write_store("withdraw_locked_ready_frontier", move |store| {
            store.withdraw_external_items_from_frontier(&frontier_to_withdraw)
        })
        .expect("withdraw locked frontier to force authoritative refill");

    let outcome = state
        .apply_arena_command(ArenaCommand::Hide {
            command_id: ArenaCommandId::forge(),
            expected_revision: current.revision(),
            expected_sampler_epoch: current.sampler_epoch(),
            turn_id: current.id().clone(),
            action_token: current.action_token().clone(),
            handle: ArenaHandle::Remote(remote_item_id),
            hidden: true,
            cluster_ids: Vec::new(),
        })
        .expect("hide remote through arena command under lock");

    assert_eq!(outcome.status, ArenaCommandStatus::Applied);
    let next = outcome
        .current
        .expect("authoritative hide should refill locked frontier instead of bricking arena");
    let next_remote = match (&next.pair().left, &next.pair().right) {
        (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
        | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => *item_id,
        _ => panic!("expected a local-vs-remote pair after hiding locked remote"),
    };
    let next_item = state
        .read_store()
        .expect("open read store for refilled locked pair")
        .remote_item(next_remote)
        .expect("load refilled locked remote")
        .expect("refilled locked remote present");
    assert_eq!(
        next_item.stream_id, locked_stream_id,
        "authoritative hide should keep serving the locked stream after refilling it"
    );
    assert!(
        state
            .read_store()
            .expect("open read store for preserved lock after hide")
            .session_subsource_lock(state.active.session_id)
            .expect("reload preserved lock after hide")
            .is_some(),
        "authoritative hide should preserve the active subsource lock"
    );
}
