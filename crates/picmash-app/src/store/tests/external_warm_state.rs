use crate::{
    identity::inspect_image_bytes,
    model::EmbeddingRecord,
    quality_features::{QUALITY_FEATURE_REVISION, extract_asset_quality_features},
};
use time::OffsetDateTime;

use super::{Store, flat_png, remote_item, remote_stream, test_root};

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
    assert!(cold.needs_identity);
    assert!(cold.needs_quality_features);
    assert!(cold.needs_embedding);
    assert!(cold.needs_face_embedding);
    assert!(!cold.needs_clip_embedding);
    assert!(cold.needs_inline_work());

    store
        .save_external_item_identity(item_id, &identity, &cache_path)
        .expect("save identity");
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
    assert!(!warm.needs_identity);
    assert!(!warm.needs_quality_features);
    assert!(!warm.needs_embedding);
    assert!(!warm.needs_clip_embedding);
    assert!(!warm.needs_face_embedding);
    assert!(!warm.needs_inline_work());
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
