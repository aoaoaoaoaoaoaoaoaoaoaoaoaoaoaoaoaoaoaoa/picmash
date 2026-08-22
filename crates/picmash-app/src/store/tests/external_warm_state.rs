use crate::{
    identity::inspect_image_bytes,
    model::EmbeddingRecord,
    quality_features::{QUALITY_FEATURE_REVISION, extract_asset_quality_features},
};
use time::OffsetDateTime;

use super::{ExternalIdentityDisposition, Store, flat_png, remote_item, remote_stream, test_root};

#[test]
fn external_item_warm_state_tracks_inline_work_seams() {
    let root = test_root("external-item-warm-state");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(1003, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(256, 256, [120, 90, 180]);
    let cache_path = root.join("warm-state.png");
    std::fs::write(&cache_path, &bytes).expect("write cache");
    let identity = inspect_image_bytes(&bytes).expect("inspect identity");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1003",
            &remote_item(1003, 2004, None),
            Some(&cache_path),
        )
        .expect("upsert item");

    let cold = store
        .external_item_warm_state(
            item_id,
            "test-model",
            None,
            "face-model",
            QUALITY_FEATURE_REVISION,
        )
        .expect("cold warm state");
    assert!(!cold.needs_materialization);
    assert!(cold.needs_identity);
    assert!(cold.needs_quality_features);
    assert!(cold.needs_embedding);
    assert!(cold.needs_face_embedding);
    assert!(!cold.needs_clip_embedding);
    assert!(cold.needs_inline_work());

    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .save_external_embedding(
            item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![1.0, 0.0, 0.0],
            },
            &cache_path,
        )
        .expect("save embedding");
    store
        .save_external_item_quality_features(
            item_id,
            QUALITY_FEATURE_REVISION,
            &extract_asset_quality_features(&bytes).expect("extract features"),
        )
        .expect("save quality features");
    store
        .save_external_face_embedding(item_id, "face-model", None)
        .expect("negative-cache face embedding");

    let warm = store
        .external_item_warm_state(
            item_id,
            "test-model",
            None,
            "face-model",
            QUALITY_FEATURE_REVISION,
        )
        .expect("warm state");
    assert!(!warm.needs_materialization);
    assert!(!warm.needs_identity);
    assert!(!warm.needs_quality_features);
    assert!(!warm.needs_embedding);
    assert!(!warm.needs_clip_embedding);
    assert!(!warm.needs_face_embedding);
    assert!(!warm.needs_inline_work());
}

#[test]
fn external_source_ready_profile_ignores_missing_cached_files() {
    let root = test_root("external-ready-profile-missing-cache");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(2001, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(256, 256, [30, 90, 170]);
    let cache_path = root.join("missing-cache.png");
    std::fs::write(&cache_path, &bytes).expect("write cache");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 2001",
            &remote_item(2001, 2002, None),
            Some(&cache_path),
        )
        .expect("upsert item");
    let identity = inspect_image_bytes(&bytes).expect("inspect identity");
    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .save_external_embedding(
            item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![1.0, 0.0, 0.0],
            },
            &cache_path,
        )
        .expect("save embedding");

    let (ready_before, by_stream_before) = store
        .external_source_ready_profile("4chan:s", "test-model")
        .expect("ready profile before missing cache");
    assert_eq!(ready_before, 1);
    assert_eq!(by_stream_before.get(&stream_id), Some(&1));

    std::fs::remove_file(&cache_path).expect("remove cache");

    let warm = store
        .external_item_warm_state(
            item_id,
            "test-model",
            None,
            "face-model",
            QUALITY_FEATURE_REVISION,
        )
        .expect("warm state after missing cache");
    assert!(!warm.needs_materialization);

    let (ready_after, by_stream_after) = store
        .external_source_ready_profile("4chan:s", "test-model")
        .expect("ready profile after missing cache");
    assert_eq!(ready_after, 0);
    assert!(by_stream_after.is_empty());

    let (_, _, cached_after) = store
        .external_source_counts("4chan:s")
        .expect("source counts after missing cache");
    assert_eq!(cached_after, 0);
}

#[test]
fn external_source_ready_paths_ignore_missing_files_and_keep_frontier_order() {
    let root = test_root("external-ready-paths");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (older_stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(4001, 1))
        .expect("upsert older stream");
    assert!(!blocked);
    let (newer_stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(4002, 2))
        .expect("upsert newer stream");
    assert!(!blocked);

    let older_bytes = flat_png(256, 256, [30, 90, 170]);
    let older_path = root.join("older.png");
    std::fs::write(&older_path, &older_bytes).expect("write older cache");
    let older_item_id = store
        .upsert_external_item(
            "4chan:s",
            older_stream_id,
            "thread 4001",
            &remote_item(4001, 4010, None),
            Some(&older_path),
        )
        .expect("upsert older item");
    let older_identity = inspect_image_bytes(&older_bytes).expect("inspect older identity");
    assert_eq!(
        store
            .save_external_item_identity(older_item_id, &older_identity, &older_path)
            .expect("save older identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .save_external_embedding(
            older_item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![1.0, 0.0, 0.0],
            },
            &older_path,
        )
        .expect("save older embedding");

    let newer_stale_bytes = flat_png(256, 256, [120, 40, 200]);
    let newer_stale_path = root.join("newer-stale.png");
    std::fs::write(&newer_stale_path, &newer_stale_bytes).expect("write stale newer cache");
    let newer_stale_item_id = store
        .upsert_external_item(
            "4chan:s",
            newer_stream_id,
            "thread 4002",
            &remote_item(4002, 4011, None),
            Some(&newer_stale_path),
        )
        .expect("upsert stale newer item");
    let newer_stale_identity =
        inspect_image_bytes(&newer_stale_bytes).expect("inspect stale newer identity");
    assert_eq!(
        store
            .save_external_item_identity(
                newer_stale_item_id,
                &newer_stale_identity,
                &newer_stale_path
            )
            .expect("save stale newer identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .save_external_embedding(
            newer_stale_item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![0.0, 1.0, 0.0],
            },
            &newer_stale_path,
        )
        .expect("save stale newer embedding");

    let newer_fresh_bytes = flat_png(256, 256, [200, 120, 40]);
    let newer_fresh_path = root.join("newer-fresh.png");
    std::fs::write(&newer_fresh_path, &newer_fresh_bytes).expect("write fresh newer cache");
    let newer_fresh_item_id = store
        .upsert_external_item(
            "4chan:s",
            newer_stream_id,
            "thread 4002",
            &remote_item(4002, 4012, None),
            Some(&newer_fresh_path),
        )
        .expect("upsert fresh newer item");
    let newer_fresh_identity =
        inspect_image_bytes(&newer_fresh_bytes).expect("inspect fresh newer identity");
    assert_eq!(
        store
            .save_external_item_identity(
                newer_fresh_item_id,
                &newer_fresh_identity,
                &newer_fresh_path
            )
            .expect("save fresh newer identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .save_external_embedding(
            newer_fresh_item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![0.0, 0.0, 1.0],
            },
            &newer_fresh_path,
        )
        .expect("save fresh newer embedding");

    std::fs::remove_file(&newer_stale_path).expect("remove stale newer cache");

    let ready_paths = store
        .external_source_ready_paths("4chan:s", "test-model")
        .expect("ready paths");
    let ready_keys = ready_paths
        .into_iter()
        .map(|entry| (entry.stream_id, entry.path))
        .collect::<Vec<_>>();
    assert_eq!(
        ready_keys,
        vec![
            (newer_stream_id, newer_fresh_path),
            (older_stream_id, older_path),
        ]
    );
}

#[test]
fn external_frontier_ready_requires_canonical_identity() {
    let root = test_root("external-frontier-requires-identity");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(2101, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(256, 256, [130, 40, 210]);
    let cache_path = root.join("half-warmed.png");
    std::fs::write(&cache_path, &bytes).expect("write cache");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 2101",
            &remote_item(2101, 2102, None),
            Some(&cache_path),
        )
        .expect("upsert item");
    store
        .save_external_embedding(
            item_id,
            &EmbeddingRecord {
                model_name: "test-model".to_owned(),
                vector: vec![1.0, 0.0, 0.0],
            },
            &cache_path,
        )
        .expect("save embedding");

    assert!(
        !store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier state before identity"),
        "a half-warmed remote must not enter the arena before visual tombstoning works"
    );
    assert_eq!(
        store
            .external_source_ready_profile("4chan:s", "test-model")
            .expect("ready profile before identity")
            .0,
        0,
        "ready source accounting must not count half-warmed remotes",
    );

    let identity = inspect_image_bytes(&bytes).expect("inspect identity");
    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save identity"),
        ExternalIdentityDisposition::Active
    );
    assert!(
        store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier state after identity"),
        "identity completion should release the remote into the frontier"
    );
}

#[test]
fn external_source_recently_selected_tracks_idle_remote_activity() {
    let root = test_root("external-source-recently-selected");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let session_id = store.create_session(corpus_id).expect("create session").id;
    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(3001, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(256, 256, [90, 160, 200]);
    let cache_path = root.join("recent-selected.png");
    std::fs::write(&cache_path, &bytes).expect("write cache");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 3001",
            &remote_item(3001, 3002, None),
            Some(&cache_path),
        )
        .expect("upsert item");

    assert!(
        !store
            .external_source_recently_selected(
                session_id,
                "4chan:s",
                OffsetDateTime::now_utc().unix_timestamp() - 60,
            )
            .expect("cold recent selection")
    );

    let import_path = corpus_root.join("asset_recent.png");
    let asset_id = store
        .ingest_external_import(corpus_id, &import_path, &bytes, 0, None)
        .expect("ingest backing asset");
    store
        .note_external_selected(session_id, corpus_id, item_id, &asset_id)
        .expect("note external selection");

    assert!(
        store
            .external_source_recently_selected(
                session_id,
                "4chan:s",
                OffsetDateTime::now_utc().unix_timestamp() - 60,
            )
            .expect("warm recent selection")
    );
}
