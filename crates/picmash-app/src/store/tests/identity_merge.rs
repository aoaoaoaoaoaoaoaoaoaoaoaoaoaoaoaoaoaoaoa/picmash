use super::*;

#[test]
fn identity_merge_replays_beauty_and_ignores_collapsed_self_duels() {
    let root = test_root("facemash-identity-replay");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    for (name, color) in [
        ("a.png", [140, 120, 100]),
        ("b.png", [150, 125, 105]),
        ("c.png", [160, 130, 110]),
    ] {
        std::fs::write(corpus_root.join(name), flat_png(512, 512, color)).expect("write image");
    }

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let session_id = store.create_session(corpus_id).expect("create session").id;
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let mut assets = store.corpus_assets(corpus_id).expect("load assets");
    assets.sort_by(|left, right| left.path.cmp(&right.path));
    assert!(
        assets.len() >= 2,
        "expected at least two ingested assets, got {}",
        assets.len()
    );

    let template = DetectedFace {
        bbox: crate::face::FaceBbox {
            x: 96.0,
            y: 72.0,
            w: 192.0,
            h: 224.0,
        },
        landmarks: FaceLandmarks([
            (136.0, 144.0),
            (228.0, 146.0),
            (182.0, 190.0),
            (148.0, 236.0),
            (220.0, 238.0),
        ]),
        confidence: 0.96,
    };
    let template_c = DetectedFace {
        bbox: crate::face::FaceBbox {
            x: 192.0,
            y: 124.0,
            w: 168.0,
            h: 204.0,
        },
        landmarks: FaceLandmarks([
            (228.0, 184.0),
            (308.0, 186.0),
            (270.0, 224.0),
            (236.0, 274.0),
            (304.0, 276.0),
        ]),
        confidence: 0.94,
    };

    let face_a = store
        .insert_face(
            Some(&assets[0].id),
            None,
            "detector:test",
            &template,
            Some("a.png"),
            Some(("face-embed:test", &[0.1f32, 0.2, 0.3, 0.4])),
            None,
        )
        .expect("insert face a");
    let face_b = store
        .insert_face(
            Some(&assets[1].id),
            None,
            "detector:test",
            &template,
            Some("b.png"),
            Some(("face-embed:test", &[0.2f32, 0.1, 0.4, 0.3])),
            None,
        )
        .expect("insert face b");
    let face_c = store
        .insert_face(
            Some(&assets[assets.len().saturating_sub(1)].id),
            None,
            "detector:test",
            &template_c,
            Some("c.png"),
            Some(("face-embed:test", &[0.4f32, 0.3, 0.2, 0.1])),
            None,
        )
        .expect("insert face c");

    let face_a_record = store
        .face_by_id(face_a)
        .expect("load face a")
        .expect("face a exists");
    let face_b_record = store
        .face_by_id(face_b)
        .expect("load face b")
        .expect("face b exists");
    let (a_after_b, b_after_a) =
        rate_face_win(face_a_record.identity.beauty, face_b_record.identity.beauty);
    store
        .record_face_comparison(session_id, face_a, face_b, a_after_b, b_after_a)
        .expect("record intra-person duel");

    let face_a_record = store
        .face_by_id(face_a)
        .expect("reload face a")
        .expect("face a exists");
    let face_c_record = store
        .face_by_id(face_c)
        .expect("load face c")
        .expect("face c exists");
    let (a_after_c, c_after_a) =
        rate_face_win(face_a_record.identity.beauty, face_c_record.identity.beauty);
    store
        .record_face_comparison(session_id, face_a, face_c, a_after_c, c_after_a)
        .expect("record live duel");

    store
        .fuse_face_identities(face_a, face_b)
        .expect("merge identities");
    store
        .rebuild_identity_beauty()
        .expect("rebuild merged beauty");

    let face_a_record = store
        .face_by_id(face_a)
        .expect("reload face a")
        .expect("face a exists");
    let face_b_record = store
        .face_by_id(face_b)
        .expect("reload face b")
        .expect("face b exists");
    let face_c_record = store
        .face_by_id(face_c)
        .expect("reload face c")
        .expect("face c exists");

    assert_eq!(face_a_record.identity.id, face_b_record.identity.id);
    assert_eq!(face_a_record.identity.duel_count, 1);
    assert_eq!(face_b_record.identity.duel_count, 1);
    assert_eq!(face_c_record.identity.duel_count, 1);
    assert!(
        face_a_record.identity.beauty.mean > 1500.0,
        "merged identity should keep only the external win"
    );
    assert!(
        face_c_record.identity.beauty.mean < 1500.0,
        "external loser should still be penalized"
    );
}

#[test]
fn naming_unnamed_face_into_existing_subject_merges_but_named_alias_collapse_is_rejected() {
    let root = test_root("identity-naming-merge-policy");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    for (name, color) in [
        ("a.png", [90, 100, 110]),
        ("b.png", [100, 110, 120]),
        ("c.png", [110, 120, 130]),
    ] {
        std::fs::write(corpus_root.join(name), flat_png(512, 512, color)).expect("write image");
    }

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let mut assets = store.corpus_assets(corpus_id).expect("load assets");
    assets.sort_by(|left, right| left.path.cmp(&right.path));

    let template = DetectedFace {
        bbox: crate::face::FaceBbox {
            x: 96.0,
            y: 72.0,
            w: 192.0,
            h: 224.0,
        },
        landmarks: FaceLandmarks([
            (136.0, 144.0),
            (228.0, 146.0),
            (182.0, 190.0),
            (148.0, 236.0),
            (220.0, 238.0),
        ]),
        confidence: 0.96,
    };

    let face_a = store
        .insert_face(
            Some(&assets[0].id),
            None,
            "detector:test",
            &template,
            Some("a.png"),
            Some(("face-embed:test", &[0.1f32, 0.2, 0.3, 0.4])),
            None,
        )
        .expect("insert face a");
    let face_b = store
        .insert_face(
            Some(&assets[1].id),
            None,
            "detector:test",
            &template,
            Some("b.png"),
            Some(("face-embed:test", &[0.2f32, 0.1, 0.4, 0.3])),
            None,
        )
        .expect("insert face b");
    let face_c = store
        .insert_face(
            Some(&assets[2].id),
            None,
            "detector:test",
            &template,
            Some("c.png"),
            Some(("face-embed:test", &[0.4f32, 0.3, 0.2, 0.1])),
            None,
        )
        .expect("insert face c");

    store
        .rename_face_identity(face_a, "alice")
        .expect("name alice");
    store.rename_face_identity(face_b, "bob").expect("name bob");
    store
        .rename_face_identity(face_c, "alice")
        .expect("merge unnamed face into alice");

    let face_a_record = store
        .face_by_id(face_a)
        .expect("load face a")
        .expect("face a exists");
    let face_b_record = store
        .face_by_id(face_b)
        .expect("load face b")
        .expect("face b exists");
    let face_c_record = store
        .face_by_id(face_c)
        .expect("load face c")
        .expect("face c exists");

    assert_eq!(
        face_a_record.identity.id, face_c_record.identity.id,
        "unnamed face should merge into the named subject when assigned that name"
    );
    assert_ne!(face_a_record.identity.id, face_b_record.identity.id);

    let error = store
        .rename_face_identity(face_b, "alice")
        .expect_err("named bob must not silently collapse into existing alice");
    assert!(
        error
            .to_string()
            .contains("cannot rename distinct named face identity"),
        "unexpected naming rejection: {error:#}"
    );
}
