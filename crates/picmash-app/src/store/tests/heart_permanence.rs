use super::*;

#[test]
fn perturbative_quality_replay_ignores_heart_deactivation_events() {
    let root = test_root("perturbative-heart-permanent");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::write(
        corpus_root.join("only.png"),
        flat_png(256, 256, [160, 120, 200]),
    )
    .expect("write asset");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    store
        .set_active_quality_model(&QualityModelRecord::runtime_default(
            OffsetDateTime::now_utc(),
        ))
        .expect("set perturbative model");

    let asset = store
        .corpus_assets(corpus_id)
        .expect("load assets")
        .into_iter()
        .next()
        .expect("single asset");
    let mut hearted_asset = asset.clone();
    hearted_asset.is_hearted = true;
    hearted_asset.heart_count = 1;
    let mut session = store
        .resume_or_create_session(corpus_id, time::Duration::minutes(10))
        .expect("resume session");
    session.hearts = 1;

    store
        .persist_heart_step(&session, &hearted_asset, &asset.id, true)
        .expect("persist heart");
    store
        .conn
        .execute(
            r"
            INSERT INTO heart_events (session_id, corpus_id, asset_id, created_at, active)
            VALUES (?1, ?2, ?3, ?4, 0)
            ",
            rusqlite::params![
                session.id.0,
                corpus_id.0,
                asset.id.0,
                OffsetDateTime::now_utc().unix_timestamp(),
            ],
        )
        .expect("insert stale deactivation");
    store
        .conn
        .execute("UPDATE assets SET heart_count = 0", [])
        .expect("clear asset hearts");
    store
        .conn
        .execute("UPDATE sessions SET hearts = 0", [])
        .expect("clear session hearts");
    store
        .conn
        .execute("DELETE FROM session_asset_hearts", [])
        .expect("clear heart flags");
    store
        .conn
        .execute(
            "DELETE FROM quality_asset_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )
        .expect("clear perturbative asset cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_session_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )
        .expect("clear perturbative session cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_subject_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )
        .expect("clear perturbative subject cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_replay_cursors WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )
        .expect("clear perturbative replay cursors");

    let replay = store
        .rebuild_active_quality_state(embedder.model_name())
        .expect("rebuild perturbative state");
    assert_eq!(
        replay.formal_version,
        QualityFormalVersion::HierarchicalPerturbativeV3
    );

    let restored_asset = store
        .corpus_asset(corpus_id, &asset.id)
        .expect("reload asset")
        .expect("asset present");
    let restored_session = store.session(session.id).expect("reload session");
    let restored_hearts = store
        .session_hearted_assets(session.id)
        .expect("reload heart flags");
    let restored_asset_hearts = store.hearted_assets().expect("reload asset heart flags");
    let expected_hearts = HashSet::from([asset.id.clone()]);

    assert_eq!(restored_asset.heart_count, 1);
    assert!(restored_asset.is_hearted);
    assert_eq!(restored_session.hearts, 1);
    assert_eq!(restored_hearts, expected_hearts);
    assert_eq!(restored_asset_hearts, expected_hearts);
}
