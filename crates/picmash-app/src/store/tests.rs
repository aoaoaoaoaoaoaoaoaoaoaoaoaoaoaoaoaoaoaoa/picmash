use std::{
    collections::{HashMap, HashSet},
    env,
    io::Cursor,
};

use image::{Rgb, RgbImage};
use rusqlite::Connection;
use time::OffsetDateTime;
use ulid::Ulid;

use super::{ExternalIdentityDisposition, Store};
use crate::{
    asset_domain::AssetDomainLabel,
    face::{DetectedFace, FaceLandmarks},
    facemash::rate_face_win,
    identity::inspect_image_bytes,
    maintenance::{MaintenanceJobKind, MaintenanceJobSpec, MaintenancePriority},
    model::{
        AssetId, CorpusId, EmbeddingRecord, ProjectionModel, SessionEmbeddingHead,
        SessionSubsourceLock, dot, session_utility, sigmoid, subtract,
    },
    onnx::OnnxEngine,
    quality::{
        HIERARCHICAL_DUEL_BETA, HIERARCHICAL_TECH_WEIGHT, HierarchicalAssetPosterior,
        HierarchicalAssetQualityCacheV1, HierarchicalSessionPosterior,
        HierarchicalSessionQualityCacheV1, LEGACY_EXACT_OFFSET_EPSILON, LEGACY_L2_FRONTIER,
        LEGACY_L2_OFFSET, LEGACY_L2_SESSION_HEAD, LEGACY_LR_DUEL_ALPHA, LEGACY_LR_DUEL_COORD,
        LEGACY_LR_DUEL_HEAD, LEGACY_LR_DUEL_MOOD, LEGACY_LR_PROJECTION,
        LEGACY_PROJECTION_WEIGHT_DECAY, LegacyUnaryFeedback, PerturbativeAssetPosterior,
        PerturbativeSessionPosterior, QualityFormalVersion, QualityModelRecord,
        diagonal_adf_update, gaussian_duel_moment_match, legacy_batter_asset, legacy_heart_bias,
        legacy_projection_prior, legacy_shove_mood,
    },
    quality_features::{StoredAssetQualityFeatures, extract_asset_quality_features},
    sources::{RemoteItemSnapshot, RemoteStreamSnapshot},
};

mod external_warm_state;
mod heart_permanence;
mod support;
use support::*;

#[test]
fn session_subsource_lock_round_trips() {
    let root = test_root("session-subsource-lock");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");

    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let session = store
        .resume_or_create_session(corpus_id, time::Duration::minutes(10))
        .expect("resume session");

    assert_eq!(
        store
            .session_subsource_lock(session.id)
            .expect("load empty session lock"),
        None
    );

    store
        .set_session_subsource_lock(session.id, "4chan:s", 77)
        .expect("set session lock");
    assert_eq!(
        store
            .session_subsource_lock(session.id)
            .expect("reload session lock"),
        Some(SessionSubsourceLock {
            source_key: "4chan:s".to_owned(),
            stream_id: 77,
        })
    );

    store
        .clear_session_subsource_lock(session.id)
        .expect("clear session lock");
    assert_eq!(
        store
            .session_subsource_lock(session.id)
            .expect("reload cleared session lock"),
        None
    );
}

#[test]
fn migrates_legacy_asset_schema_without_rewriting_asset_id() {
    let root = test_root("legacy-migrate");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let image_path = corpus_root.join("legacy.png");
    std::fs::write(&image_path, flat_png(640, 480, [90, 120, 180])).expect("write legacy png");
    let db_path = root.join("picmash.sqlite3");
    let legacy_asset_id = "legacy-byte-hash";
    let corpus_root_string = corpus_root.to_string_lossy().into_owned();
    let image_path_string = image_path.to_string_lossy().into_owned();

    let conn = Connection::open(&db_path).expect("open legacy sqlite");
    conn.execute_batch(
        r"
            CREATE TABLE corpora (
                id INTEGER PRIMARY KEY,
                root_path TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE assets (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                rotation_quarters INTEGER NOT NULL DEFAULT 0,
                alpha REAL NOT NULL DEFAULT 0.0,
                c0 REAL NOT NULL DEFAULT 0.0,
                c1 REAL NOT NULL DEFAULT 0.0,
                c2 REAL NOT NULL DEFAULT 0.0,
                heart_count INTEGER NOT NULL DEFAULT 0,
                compare_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE corpus_assets (
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                hidden INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (corpus_id, asset_id)
            );
            CREATE UNIQUE INDEX idx_corpus_assets_path ON corpus_assets(corpus_id, path);
            ",
    )
    .expect("create legacy schema");
    conn.execute(
        "INSERT INTO corpora (id, root_path, created_at) VALUES (1, ?1, 0)",
        rusqlite::params![corpus_root_string],
    )
    .expect("insert corpus");
    conn.execute(
            r"
            INSERT INTO assets (
                id, created_at, rotation_quarters, alpha, c0, c1, c2, heart_count, compare_count, win_count
            ) VALUES (?1, 0, 0, 1.25, 0.0, 0.0, 0.0, 2, 3, 4)
            ",
            rusqlite::params![legacy_asset_id],
        )
        .expect("insert legacy asset");
    conn.execute(
        "INSERT INTO corpus_assets (corpus_id, asset_id, path, hidden) VALUES (1, ?1, ?2, 1)",
        rusqlite::params![legacy_asset_id, image_path_string],
    )
    .expect("insert legacy corpus asset");
    drop(conn);

    let store = Store::open(&db_path).expect("open migrated store");
    let asset = store
        .corpus_asset(CorpusId(1), &AssetId(legacy_asset_id.to_owned()))
        .expect("load migrated asset")
        .expect("migrated asset present");
    assert_eq!(asset.id.0, legacy_asset_id);
    assert_eq!(asset.path, image_path);
    assert!(asset.hidden);

    let preferred = store
            .conn
            .query_row(
                "SELECT preferred_blob_id, visual_key, pixel_width, pixel_height FROM assets WHERE id = ?1",
                rusqlite::params![legacy_asset_id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .expect("query migrated asset identity");
    assert!(preferred.0.is_some());
    assert!(preferred.1.is_some());
    assert_eq!(preferred.2, 640);
    assert_eq!(preferred.3, 480);
}

#[test]
fn ingest_merges_visual_variants_and_promotes_higher_resolution() {
    let root = test_root("visual-merge");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let low_path = corpus_root.join("low.png");
    let high_path = corpus_root.join("high.png");
    let low_bytes = flat_png(64, 64, [120, 80, 40]);
    let high_bytes = flat_png(512, 512, [120, 80, 40]);
    std::fs::write(&low_path, &low_bytes).expect("write low variant");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest low variant");
    let low_asset = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(low_asset.len(), 1);
    let durable_asset_id = low_asset[0].id.clone();

    std::fs::write(&high_path, &high_bytes).expect("write high variant");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("reingest high variant");

    let merged_assets = store.corpus_assets(corpus_id).expect("load merged assets");
    assert_eq!(merged_assets.len(), 1);
    assert_eq!(merged_assets[0].id, durable_asset_id);
    assert_eq!(merged_assets[0].path, high_path);

    let variant_count = store
        .conn
        .query_row(
            "SELECT COUNT(*) FROM corpus_assets WHERE corpus_id = ?1 AND asset_id = ?2",
            rusqlite::params![corpus_id.0, durable_asset_id.0],
            |row| row.get::<_, i64>(0),
        )
        .expect("count merged variants");
    assert_eq!(variant_count, 2);

    let high_identity = inspect_image_bytes(&high_bytes).expect("inspect high variant");
    let preferred = store
        .conn
        .query_row(
            "SELECT preferred_blob_id, pixel_width, pixel_height FROM assets WHERE id = ?1",
            rusqlite::params![durable_asset_id.0],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .expect("query preferred baseline");
    assert_eq!(
        preferred.0.as_deref(),
        Some(high_identity.blob_id.0.as_str())
    );
    assert_eq!(preferred.1, 512);
    assert_eq!(preferred.2, 512);
}

#[test]
fn legacy_quality_replay_rebuilds_derived_state_from_event_truth() {
    let root = test_root("legacy-quality-replay");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let left_path = corpus_root.join("left.png");
    let right_path = corpus_root.join("right.png");
    std::fs::write(&left_path, flat_png(640, 480, [220, 120, 90])).expect("write left png");
    std::fs::write(&right_path, flat_png(640, 480, [90, 150, 220])).expect("write right png");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    store
        .set_active_quality_model(&QualityModelRecord {
            formal_version: QualityFormalVersion::LegacyIndependentV1,
            prior_family: Default::default(),
            prior_revision: Default::default(),
            updated_at: OffsetDateTime::now_utc(),
        })
        .expect("pin legacy quality model");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");

    let assets = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(assets.len(), 2);
    let mut left = assets
        .iter()
        .find(|asset| asset.path == left_path)
        .cloned()
        .expect("left asset present");
    let mut right = assets
        .iter()
        .find(|asset| asset.path == right_path)
        .cloned()
        .expect("right asset present");

    let model_name = "test-dino";
    let left_embedding = vec![0.7, -0.2, 0.3, 0.9];
    let right_embedding = vec![-0.4, 0.8, -0.1, 0.5];
    store
        .save_embedding(
            &left.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: left_embedding.clone(),
            },
        )
        .expect("save left embedding");
    store
        .save_embedding(
            &right.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: right_embedding.clone(),
            },
        )
        .expect("save right embedding");

    let mut session = store.create_session(corpus_id).expect("create session");
    let mut exact_offsets = HashMap::<AssetId, f32>::new();
    let mut hearted = HashSet::<AssetId>::new();
    let mut embedding_head: Option<SessionEmbeddingHead>;
    let mut projection: Option<ProjectionModel>;

    let left_before = left.clone();
    let right_before = right.clone();
    projection = Some(ProjectionModel::zero(
        model_name.to_owned(),
        left_embedding.len(),
    ));
    let left_utility = session_utility(&left, &session, 0.0, 0.0, legacy_heart_bias(&left, false));
    let right_utility =
        session_utility(&right, &session, 0.0, 0.0, legacy_heart_bias(&right, false));
    let err = 1.0 - sigmoid(left_utility - right_utility);
    let coord_gap = subtract(&left.coords, &right.coords);
    let left_prior = legacy_projection_prior(projection.as_ref(), Some(&left_embedding));
    let right_prior = legacy_projection_prior(projection.as_ref(), Some(&right_embedding));
    legacy_batter_asset(
        &mut left.alpha,
        &mut left.coords,
        &session.mood,
        &left_prior,
        err,
        LEGACY_LR_DUEL_ALPHA,
        LEGACY_LR_DUEL_COORD,
    );
    legacy_batter_asset(
        &mut right.alpha,
        &mut right.coords,
        &session.mood,
        &right_prior,
        -err,
        LEGACY_LR_DUEL_ALPHA,
        LEGACY_LR_DUEL_COORD,
    );
    for (mood, gap) in session.mood.iter_mut().zip(coord_gap.iter().copied()) {
        *mood += LEGACY_LR_DUEL_MOOD * err * gap;
    }
    left.compare_count += 1;
    right.compare_count += 1;
    left.win_count += 1;
    session.comparisons += 1;
    let mut head = SessionEmbeddingHead::zero(model_name.to_owned(), left_embedding.len());
    head.contrast_step(
        &left_embedding,
        &right_embedding,
        err,
        LEGACY_LR_DUEL_HEAD,
        LEGACY_L2_SESSION_HEAD,
    );
    if let Some(model) = &mut projection {
        model.gradient_step(
            &left_embedding,
            &left.coords,
            LEGACY_LR_PROJECTION,
            LEGACY_PROJECTION_WEIGHT_DECAY,
        );
        model.gradient_step(
            &right_embedding,
            &right.coords,
            LEGACY_LR_PROJECTION,
            LEGACY_PROJECTION_WEIGHT_DECAY,
        );
    }
    embedding_head = Some(head);
    store
        .persist_duel_step(
            &session,
            &left_before,
            &right_before,
            &left,
            &right,
            &left.id,
            left_utility,
            right_utility,
            projection.as_ref(),
            embedding_head.as_ref(),
        )
        .expect("persist duel");

    let residual_right = embedding_head
        .as_ref()
        .expect("embedding head")
        .score(&right_embedding);
    let utility_before = session_utility(
        &right,
        &session,
        residual_right,
        0.0,
        legacy_heart_bias(&right, false),
    );
    let frontier_before = session.frontier;
    let tuning = LegacyUnaryFeedback::More.tuning(utility_before, frontier_before);
    let right_prior = legacy_projection_prior(projection.as_ref(), Some(&right_embedding));
    let right_before_coords = right.coords;
    legacy_batter_asset(
        &mut right.alpha,
        &mut right.coords,
        &session.mood,
        &right_prior,
        tuning.signal,
        tuning.alpha_rate,
        tuning.coord_rate,
    );
    legacy_shove_mood(
        &mut session.mood,
        &right_before_coords,
        tuning.signal,
        tuning.mood_rate,
    );
    session.frontier +=
        tuning.frontier_rate * (-tuning.signal - LEGACY_L2_FRONTIER * session.frontier);
    let next_offset = tuning.offset_rate * (tuning.signal - LEGACY_L2_OFFSET * 0.0);
    embedding_head.as_mut().expect("embedding head").unary_step(
        &right_embedding,
        tuning.signal,
        tuning.head_rate,
        LEGACY_L2_SESSION_HEAD,
    );
    if let Some(model) = &mut projection {
        model.gradient_step(
            &right_embedding,
            &right.coords,
            tuning.projection_rate,
            LEGACY_PROJECTION_WEIGHT_DECAY,
        );
    }
    session.nudges += 1;
    if next_offset.abs() >= LEGACY_EXACT_OFFSET_EPSILON {
        exact_offsets.insert(right.id.clone(), next_offset);
    }
    store
        .persist_nudge_step(
            &session,
            &right,
            &right.id,
            1.0,
            utility_before,
            frontier_before,
            tuning.signal,
            next_offset,
            projection.as_ref(),
            embedding_head.as_ref(),
        )
        .expect("persist nudge");

    assert!(!hearted.contains(&left.id));
    hearted.insert(left.id.clone());
    left.heart_count += 1;
    session.hearts += 1;
    store
        .persist_heart_step(&session, &left, &left.id, true)
        .expect("persist heart");

    let expected_left = store
        .corpus_asset(corpus_id, &left.id)
        .expect("load expected left")
        .expect("expected left present");
    let expected_right = store
        .corpus_asset(corpus_id, &right.id)
        .expect("load expected right")
        .expect("expected right present");
    let expected_session = store.session(session.id).expect("load expected session");
    let expected_offsets = store
        .session_asset_offsets(session.id)
        .expect("load expected offsets");
    let expected_hearts = store
        .session_hearted_assets(session.id)
        .expect("load expected hearts");
    let expected_projection = store
        .projection_model(model_name)
        .expect("load expected projection")
        .expect("projection present");
    let expected_head = store
        .session_embedding_head(session.id, model_name)
        .expect("load expected head")
        .expect("head present");

    store
        .conn
        .execute(
            "UPDATE assets SET alpha = 0.0, c0 = 0.0, c1 = 0.0, c2 = 0.0, heart_count = 0, compare_count = 0, win_count = 0",
            [],
        )
        .expect("corrupt assets");
    store
        .conn
        .execute(
            "UPDATE sessions SET z0 = 0.0, z1 = 0.0, z2 = 0.0, frontier = 0.0, comparisons = 0, nudges = 0, hearts = 0",
            [],
        )
        .expect("corrupt sessions");
    store
        .conn
        .execute("DELETE FROM session_asset_offsets", [])
        .expect("clear offsets");
    store
        .conn
        .execute("DELETE FROM session_asset_hearts", [])
        .expect("clear hearts");
    store
        .conn
        .execute("DELETE FROM projection_models", [])
        .expect("clear projections");
    store
        .conn
        .execute("DELETE FROM session_embedding_heads", [])
        .expect("clear session heads");

    let replay = store
        .rebuild_active_quality_state(model_name)
        .expect("rebuild legacy quality state");
    assert_eq!(
        replay.formal_version,
        QualityFormalVersion::LegacyIndependentV1
    );
    assert_eq!(replay.comparison_events, 1);
    assert_eq!(replay.nudge_events, 1);
    assert_eq!(replay.heart_events, 1);

    let restored_left = store
        .corpus_asset(corpus_id, &left.id)
        .expect("load restored left")
        .expect("restored left present");
    let restored_right = store
        .corpus_asset(corpus_id, &right.id)
        .expect("load restored right")
        .expect("restored right present");
    let restored_session = store.session(session.id).expect("load restored session");
    let restored_offsets = store
        .session_asset_offsets(session.id)
        .expect("load restored offsets");
    let restored_hearts = store
        .session_hearted_assets(session.id)
        .expect("load restored hearts");
    let restored_projection = store
        .projection_model(model_name)
        .expect("load restored projection")
        .expect("restored projection present");
    let restored_head = store
        .session_embedding_head(session.id, model_name)
        .expect("load restored head")
        .expect("restored head present");

    assert_close(restored_left.alpha, expected_left.alpha);
    assert_slice_close(&restored_left.coords, &expected_left.coords);
    assert_eq!(restored_left.heart_count, expected_left.heart_count);
    assert_eq!(restored_left.compare_count, expected_left.compare_count);
    assert_eq!(restored_left.win_count, expected_left.win_count);
    assert_close(restored_right.alpha, expected_right.alpha);
    assert_slice_close(&restored_right.coords, &expected_right.coords);
    assert_eq!(restored_right.heart_count, expected_right.heart_count);
    assert_eq!(restored_right.compare_count, expected_right.compare_count);
    assert_eq!(restored_right.win_count, expected_right.win_count);
    assert_slice_close(&restored_session.mood, &expected_session.mood);
    assert_close(restored_session.frontier, expected_session.frontier);
    assert_eq!(restored_session.comparisons, expected_session.comparisons);
    assert_eq!(restored_session.nudges, expected_session.nudges);
    assert_eq!(restored_session.hearts, expected_session.hearts);
    assert_eq!(restored_offsets, expected_offsets);
    assert_eq!(restored_hearts, expected_hearts);
    assert_slice_close(&restored_projection.bias, &expected_projection.bias);
    assert_slice_close(&restored_projection.weights, &expected_projection.weights);
    assert_slice_close(&restored_head.weights, &expected_head.weights);

    let left_cache = store
        .asset_quality_cache(&left.id, QualityFormalVersion::LegacyIndependentV1)
        .expect("load left quality cache")
        .expect("left quality cache present");
    let session_cache = store
        .session_quality_cache(session.id, QualityFormalVersion::LegacyIndependentV1)
        .expect("load session quality cache")
        .expect("session quality cache present");
    match left_cache.payload {
        crate::quality::AssetQualityCachePayload::LegacyIndependentV1(payload) => {
            assert_close(payload.alpha_mean, expected_left.alpha);
            assert_slice_close(&payload.coords_mean, &expected_left.coords);
            assert_eq!(payload.compare_count, expected_left.compare_count);
        }
        payload => panic!("unexpected asset cache payload: {payload:?}"),
    }
    match session_cache.payload {
        crate::quality::SessionQualityCachePayload::LegacyIndependentV1(payload) => {
            assert_slice_close(&payload.mood_mean, &expected_session.mood);
            assert_close(payload.frontier_mean, expected_session.frontier);
        }
        payload => panic!("unexpected session cache payload: {payload:?}"),
    }
}

#[test]
fn hierarchical_quality_replay_rebuilds_derived_state_from_event_truth() {
    let root = test_root("hierarchical-quality-replay");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let left_path = corpus_root.join("left.png");
    let right_path = corpus_root.join("right.png");
    let left_bytes = flat_png(1024, 768, [220, 120, 90]);
    let right_bytes = flat_png(960, 640, [90, 150, 220]);
    std::fs::write(&left_path, &left_bytes).expect("write left png");
    std::fs::write(&right_path, &right_bytes).expect("write right png");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");

    let assets = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(assets.len(), 2);
    let mut left = assets
        .iter()
        .find(|asset| asset.path == left_path)
        .cloned()
        .expect("left asset present");
    let mut right = assets
        .iter()
        .find(|asset| asset.path == right_path)
        .cloned()
        .expect("right asset present");
    let model_name = "test-dino-hier";
    let left_embedding = vec![0.7, -0.2, 0.3, 0.9];
    let right_embedding = vec![-0.4, 0.8, -0.1, 0.5];
    store
        .save_embedding(
            &left.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: left_embedding.clone(),
            },
        )
        .expect("save left embedding");
    store
        .save_embedding(
            &right.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: right_embedding.clone(),
            },
        )
        .expect("save right embedding");
    store
        .set_asset_domain_label(&left.id, AssetDomainLabel::Real)
        .expect("label left as 3d");
    store
        .set_asset_domain_label(&right.id, AssetDomainLabel::Real)
        .expect("label right as 3d");
    store
        .save_asset_quality_features(&StoredAssetQualityFeatures {
            asset_id: left.id.clone(),
            extractor_revision: crate::quality_features::QUALITY_FEATURE_REVISION.to_owned(),
            features: extract_asset_quality_features(&left_bytes).expect("extract left features"),
            updated_at: OffsetDateTime::now_utc(),
        })
        .expect("save left quality features");
    store
        .save_asset_quality_features(&StoredAssetQualityFeatures {
            asset_id: right.id.clone(),
            extractor_revision: crate::quality_features::QUALITY_FEATURE_REVISION.to_owned(),
            features: extract_asset_quality_features(&right_bytes).expect("extract right features"),
            updated_at: OffsetDateTime::now_utc(),
        })
        .expect("save right quality features");

    let mut session = store.create_session(corpus_id).expect("create session");
    let replay = store
        .rebuild_active_quality_state(model_name)
        .expect("rebuild hierarchical replay");
    assert_eq!(
        replay.formal_version,
        QualityFormalVersion::HierarchicalGaussianV1
    );

    left = store
        .corpus_assets(corpus_id)
        .expect("reload assets after replay")
        .into_iter()
        .find(|asset| asset.id == left.id)
        .expect("reload left");
    right = store
        .corpus_assets(corpus_id)
        .expect("reload assets after replay")
        .into_iter()
        .find(|asset| asset.id == right.id)
        .expect("reload right");
    session = store
        .session(session.id)
        .expect("reload session after replay");

    let mut left_quality = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(&left.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("load left hierarchical cache")
            .expect("left hierarchical cache present")
            .payload,
    )
    .expect("decode left hierarchical cache");
    let mut right_quality = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(&right.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("load right hierarchical cache")
            .expect("right hierarchical cache present")
            .payload,
    )
    .expect("decode right hierarchical cache");
    let mut session_quality = HierarchicalSessionPosterior::decode(
        &store
            .session_quality_cache(session.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("load hierarchical session cache")
            .expect("hierarchical session cache present")
            .payload,
    )
    .expect("decode hierarchical session cache");

    let left_before = left.clone();
    let right_before = right.clone();
    let left_utility =
        hierarchical_test_utility_mean(&left, &left_quality, &session_quality, 0.0, false);
    let right_utility =
        hierarchical_test_utility_mean(&right, &right_quality, &session_quality, 0.0, false);
    let delta_mean = left_utility - right_utility;
    let delta_variance = hierarchical_test_utility_variance(&left_quality, &session_quality)
        + hierarchical_test_utility_variance(&right_quality, &session_quality);
    let moments =
        gaussian_duel_moment_match(delta_mean, delta_variance, 1.0, HIERARCHICAL_DUEL_BETA)
            .expect("hierarchical duel moments");
    diagonal_adf_update(
        &mut left_quality.baseline_mean,
        &mut left_quality.baseline_variance,
        1.0,
        1.0,
        moments,
    );
    diagonal_adf_update(
        &mut right_quality.baseline_mean,
        &mut right_quality.baseline_variance,
        -1.0,
        1.0,
        moments,
    );
    for axis in 0..crate::model::LATENT_DIM {
        diagonal_adf_update(
            &mut left_quality.mood_loading_mean[axis],
            &mut left_quality.mood_loading_variance[axis],
            session_quality.semantic_mood_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut right_quality.mood_loading_mean[axis],
            &mut right_quality.mood_loading_variance[axis],
            -session_quality.semantic_mood_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut session_quality.semantic_mood_mean[axis],
            &mut session_quality.semantic_mood_variance[axis],
            left_quality.mood_loading_mean[axis] - right_quality.mood_loading_mean[axis],
            1.0,
            moments,
        );
    }
    if let (Some(mean), Some(variance)) = (
        &mut left_quality.technical_mean,
        &mut left_quality.technical_variance,
    ) {
        diagonal_adf_update(mean, variance, HIERARCHICAL_TECH_WEIGHT, 1.0, moments);
    }
    if let (Some(mean), Some(variance)) = (
        &mut right_quality.technical_mean,
        &mut right_quality.technical_variance,
    ) {
        diagonal_adf_update(mean, variance, -HIERARCHICAL_TECH_WEIGHT, 1.0, moments);
    }
    for axis in 0..crate::quality_features::VIBE_DESCRIPTOR_DIM {
        diagonal_adf_update(
            &mut left_quality.vibe_mean[axis],
            &mut left_quality.vibe_variance[axis],
            session_quality.vibe_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut right_quality.vibe_mean[axis],
            &mut right_quality.vibe_variance[axis],
            -session_quality.vibe_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut session_quality.vibe_mean[axis],
            &mut session_quality.vibe_variance[axis],
            left_quality.vibe_mean[axis] - right_quality.vibe_mean[axis],
            1.0,
            moments,
        );
    }
    left_quality.canonical_mean = hierarchical_test_canonical_mean(&left_quality);
    left_quality.canonical_variance = hierarchical_test_canonical_variance(&left_quality);
    right_quality.canonical_mean = hierarchical_test_canonical_mean(&right_quality);
    right_quality.canonical_variance = hierarchical_test_canonical_variance(&right_quality);

    left.compare_count += 1;
    left.win_count += 1;
    right.compare_count += 1;
    session.comparisons += 1;
    session.mood = session_quality.semantic_mood_mean;
    hierarchical_test_sync_asset_record(&mut left, &left_quality);
    hierarchical_test_sync_asset_record(&mut right, &right_quality);

    let mut expected_head = SessionEmbeddingHead::zero(model_name.to_owned(), left_embedding.len());
    expected_head.contrast_step(
        &left_embedding,
        &right_embedding,
        1.0 - sigmoid(delta_mean),
        crate::quality::LEGACY_LR_DUEL_HEAD,
        LEGACY_L2_SESSION_HEAD,
    );

    store
        .persist_hierarchical_duel_step(
            &session,
            &left_before,
            &right_before,
            &left,
            &right,
            &left.id,
            left_utility,
            right_utility,
            Some(&expected_head),
            &hierarchical_test_asset_cache(&left_quality),
            &hierarchical_test_asset_cache(&right_quality),
            &hierarchical_test_session_cache(&session_quality),
        )
        .expect("persist hierarchical duel step");

    let feedback = LegacyUnaryFeedback::More;
    let utility_before =
        hierarchical_test_utility_mean(&left, &left_quality, &session_quality, 0.0, false);
    let frontier_before = session.frontier;
    let tuning = feedback.tuning(utility_before, frontier_before);
    let expected_offset = tuning.offset_rate * tuning.signal;
    session.frontier +=
        tuning.frontier_rate * (-tuning.signal - LEGACY_L2_FRONTIER * session.frontier);
    session.nudges += 1;
    expected_head.unary_step(
        &left_embedding,
        tuning.signal,
        tuning.head_rate,
        LEGACY_L2_SESSION_HEAD,
    );
    let expected_session_cache = hierarchical_test_session_cache(&HierarchicalSessionPosterior {
        frontier_mean: session.frontier,
        ..session_quality
    });

    store
        .persist_hierarchical_nudge_step(
            &session,
            &left,
            &left.id,
            feedback.direction(),
            utility_before,
            frontier_before,
            tuning.signal,
            expected_offset,
            Some(&expected_head),
            &hierarchical_test_asset_cache(&left_quality),
            &expected_session_cache,
        )
        .expect("persist hierarchical nudge step");

    right.heart_count += 1;
    session.hearts += 1;
    store
        .persist_heart_step(&session, &right, &right.id, true)
        .expect("persist heart step");

    let expected_left = left.clone();
    let expected_right = right.clone();
    let expected_session = session.clone();

    store
        .conn
        .execute(
            r"
            UPDATE assets
            SET alpha = 99.0,
                c0 = -99.0,
                c1 = -99.0,
                c2 = -99.0,
                heart_count = 0,
                compare_count = 0,
                win_count = 0
            ",
            [],
        )
        .expect("corrupt assets");
    store
        .conn
        .execute(
            r"
            UPDATE sessions
            SET z0 = 88.0,
                z1 = 88.0,
                z2 = 88.0,
                frontier = -88.0,
                comparisons = 0,
                nudges = 0,
                hearts = 0
            ",
            [],
        )
        .expect("corrupt sessions");
    store
        .conn
        .execute(
            "DELETE FROM quality_asset_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )
        .expect("delete hierarchical asset cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_session_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )
        .expect("delete hierarchical session cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_subject_cache WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )
        .expect("delete hierarchical subject cache");
    store
        .conn
        .execute(
            "DELETE FROM quality_replay_cursors WHERE formal_version = ?1",
            rusqlite::params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )
        .expect("delete hierarchical replay cursors");
    store
        .conn
        .execute("DELETE FROM session_asset_offsets", [])
        .expect("delete offsets");
    store
        .conn
        .execute("DELETE FROM session_asset_hearts", [])
        .expect("delete session hearts");
    store
        .conn
        .execute("DELETE FROM session_embedding_heads", [])
        .expect("delete embedding heads");

    let replay = store
        .rebuild_active_quality_state(model_name)
        .expect("rebuild hierarchical replay from event truth");
    assert_eq!(
        replay.formal_version,
        QualityFormalVersion::HierarchicalGaussianV1
    );

    let restored_assets = store
        .corpus_assets(corpus_id)
        .expect("load restored assets");
    let restored_left = restored_assets
        .iter()
        .find(|asset| asset.id == expected_left.id)
        .cloned()
        .expect("restored left asset");
    let restored_right = restored_assets
        .iter()
        .find(|asset| asset.id == expected_right.id)
        .cloned()
        .expect("restored right asset");
    let restored_session = store
        .session(expected_session.id)
        .expect("restored session");
    let restored_offsets = store
        .session_asset_offsets(expected_session.id)
        .expect("restored offsets");
    let restored_hearts = store
        .session_hearted_assets(expected_session.id)
        .expect("restored hearts");
    let restored_head = store
        .session_embedding_head(expected_session.id, model_name)
        .expect("load restored embedding head")
        .expect("restored embedding head present");

    let restored_left_cache = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(
                &expected_left.id,
                QualityFormalVersion::HierarchicalGaussianV1,
            )
            .expect("restored left cache")
            .expect("restored left cache present")
            .payload,
    )
    .expect("decode restored left cache");
    let restored_right_cache = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(
                &expected_right.id,
                QualityFormalVersion::HierarchicalGaussianV1,
            )
            .expect("restored right cache")
            .expect("restored right cache present")
            .payload,
    )
    .expect("decode restored right cache");
    let restored_session_cache = HierarchicalSessionPosterior::decode(
        &store
            .session_quality_cache(
                expected_session.id,
                QualityFormalVersion::HierarchicalGaussianV1,
            )
            .expect("restored session cache")
            .expect("restored session cache present")
            .payload,
    )
    .expect("decode restored session cache");

    assert_close(
        restored_left.alpha,
        hierarchical_test_canonical_mean(&restored_left_cache),
    );
    assert_slice_close(
        &restored_left.coords,
        &restored_left_cache.mood_loading_mean,
    );
    assert_eq!(restored_left.compare_count, expected_left.compare_count);
    assert_eq!(restored_left.win_count, expected_left.win_count);
    assert_eq!(restored_left.heart_count, expected_left.heart_count);
    assert_close(
        restored_right.alpha,
        hierarchical_test_canonical_mean(&restored_right_cache),
    );
    assert_slice_close(
        &restored_right.coords,
        &restored_right_cache.mood_loading_mean,
    );
    assert_eq!(restored_right.compare_count, expected_right.compare_count);
    assert_eq!(restored_right.win_count, expected_right.win_count);
    assert_eq!(restored_right.heart_count, expected_right.heart_count);
    assert_slice_close(
        &restored_session.mood,
        &restored_session_cache.semantic_mood_mean,
    );
    assert_close(
        restored_session.frontier,
        restored_session_cache.frontier_mean,
    );
    assert_eq!(restored_session.comparisons, expected_session.comparisons);
    assert_eq!(restored_session.nudges, expected_session.nudges);
    assert_eq!(restored_session.hearts, expected_session.hearts);
    assert_eq!(
        restored_offsets,
        HashMap::from([(expected_left.id.clone(), expected_offset)])
    );
    assert_eq!(restored_hearts, HashSet::from([expected_right.id.clone()]));
    assert_slice_close(&restored_head.weights, &expected_head.weights);

    assert_close(
        restored_left_cache.canonical_mean,
        hierarchical_test_canonical_mean(&restored_left_cache),
    );
    assert_close(
        restored_left_cache.canonical_variance,
        hierarchical_test_canonical_variance(&restored_left_cache),
    );
    assert_close(
        restored_right_cache.canonical_mean,
        hierarchical_test_canonical_mean(&restored_right_cache),
    );
    assert_close(
        restored_right_cache.canonical_variance,
        hierarchical_test_canonical_variance(&restored_right_cache),
    );
    assert_slice_close(
        &restored_session_cache.semantic_mood_mean,
        &restored_session.mood,
    );
    assert_close(
        restored_session_cache.frontier_mean,
        restored_session.frontier,
    );
}

#[test]
fn hierarchical_quality_replay_learns_from_remote_rejects() {
    let root = test_root("hierarchical-remote-reject");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let session = store.create_session(corpus_id).expect("create session");
    let model_name = "test-dino-external-reject";
    store
        .upsert_external_source("4chan:w", "w", "4chan_board", "w", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:w", &remote_stream(4001, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(960, 640, [180, 140, 110]);
    let cached_path = root.join("remote.png");
    std::fs::write(&cached_path, &bytes).expect("write remote cache");
    let item_id = store
        .upsert_external_item(
            "4chan:w",
            stream_id,
            "thread 4001",
            &remote_item(4001, 5001, None),
            Some(&cached_path),
        )
        .expect("upsert external item");
    store
        .save_external_embedding(
            item_id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: vec![0.5, -0.3, 0.1, 0.8],
            },
            &cached_path,
        )
        .expect("save external embedding");
    store
        .save_external_item_quality_features(
            item_id,
            crate::quality_features::QUALITY_FEATURE_REVISION,
            &extract_asset_quality_features(&bytes).expect("extract remote features"),
        )
        .expect("save external quality features");
    store
        .reject_external_item(
            session.id,
            corpus_id,
            item_id,
            None,
            crate::model::ExternalEventKind::Rejected,
        )
        .expect("record remote reject");

    let replay = store
        .rebuild_active_quality_state(model_name)
        .expect("rebuild hierarchical replay from remote reject");
    assert_eq!(
        replay.formal_version,
        QualityFormalVersion::HierarchicalGaussianV1
    );

    let remote_cache = store
        .external_item_quality_cache(item_id, QualityFormalVersion::HierarchicalGaussianV1)
        .expect("load external quality cache")
        .expect("external quality cache present");
    let remote = HierarchicalAssetPosterior::decode(&remote_cache.payload)
        .expect("decode external hierarchical cache");
    assert!(
        remote.baseline_mean < 0.0,
        "remote reject should depress baseline quality, got {}",
        remote.baseline_mean
    );

    let session_cache = store
        .session_quality_cache(session.id, QualityFormalVersion::HierarchicalGaussianV1)
        .expect("load hierarchical session cache")
        .expect("hierarchical session cache present");
    let session_quality = HierarchicalSessionPosterior::decode(&session_cache.payload)
        .expect("decode hierarchical session cache");
    assert!(
        session_quality.frontier_mean > 0.0,
        "remote reject should raise the session frontier, got {}",
        session_quality.frontier_mean
    );
}

#[test]
fn recent_arena_asset_ids_merge_local_duels_with_remote_selections() {
    let root = test_root("arena-recent-assets");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    for (name, rgb) in [
        ("a.png", [120, 80, 40]),
        ("b.png", [80, 120, 40]),
        ("c.png", [40, 80, 120]),
    ] {
        std::fs::write(corpus_root.join(name), flat_png(256, 256, rgb)).expect("write asset");
    }

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let session = store.create_session(corpus_id).expect("create session");
    let assets = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(assets.len(), 3);

    store
        .conn
        .execute(
            r"
            INSERT INTO comparisons (
                session_id,
                corpus_id,
                left_asset_id,
                right_asset_id,
                winner_asset_id,
                created_at,
                left_utility,
                right_utility
            ) VALUES (?1, ?2, ?3, ?4, ?5, 10, 0.0, 0.0)
            ",
            rusqlite::params![
                session.id.0,
                corpus_id.0,
                assets[0].id.0,
                assets[1].id.0,
                assets[0].id.0,
            ],
        )
        .expect("insert comparison");

    store
        .upsert_external_source("4chan:w", "w", "4chan_board", "w", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:w", &remote_stream(9001, 1))
        .expect("upsert stream");
    assert!(!blocked);
    let remote_path = root.join("remote-selection.png");
    std::fs::write(&remote_path, flat_png(640, 480, [180, 140, 110])).expect("write remote");
    let item_id = store
        .upsert_external_item(
            "4chan:w",
            stream_id,
            "thread 9001",
            &remote_item(9001, 9002, None),
            Some(&remote_path),
        )
        .expect("upsert external item");
    let remote_identity =
        inspect_image_bytes(&std::fs::read(&remote_path).expect("read remote identity payload"))
            .expect("inspect remote identity");
    assert!(
        matches!(
            store
                .save_external_item_identity(item_id, &remote_identity, &remote_path)
                .expect("save remote identity"),
            ExternalIdentityDisposition::Active
        ),
        "distinct remote selection should stay frontier-eligible"
    );
    store
        .conn
        .execute(
            r"
            INSERT INTO external_events (
                session_id,
                corpus_id,
                source_key,
                stream_id,
                item_id,
                local_asset_id,
                event_kind,
                created_at
            ) VALUES (?1, ?2, '4chan:w', ?3, ?4, ?5, 'selected', 20)
            ",
            rusqlite::params![
                session.id.0,
                corpus_id.0,
                stream_id,
                item_id.0,
                assets[2].id.0,
            ],
        )
        .expect("insert external selection");

    let recent = store
        .recent_arena_asset_ids(session.id, 3)
        .expect("load recent arena assets");
    assert_eq!(
        recent,
        vec![
            assets[2].id.clone(),
            assets[0].id.clone(),
            assets[1].id.clone()
        ]
    );

    let recent_visual_keys = store
        .recent_arena_visual_keys(session.id, 4)
        .expect("load recent arena visual keys");
    assert_eq!(
        recent_visual_keys,
        vec![
            assets[2]
                .visual_key
                .clone()
                .expect("seed asset c visual key"),
            remote_identity.visual_key.clone(),
            assets[0]
                .visual_key
                .clone()
                .expect("seed asset a visual key"),
            assets[1]
                .visual_key
                .clone()
                .expect("seed asset b visual key"),
        ]
    );
}

#[test]
fn recent_external_recency_queries_include_result_events() {
    let root = test_root("recent-external-recency-result-events");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    for (name, rgb) in [("a.png", [120, 80, 40]), ("b.png", [80, 120, 40])] {
        std::fs::write(corpus_root.join(name), flat_png(256, 256, rgb)).expect("write asset");
    }

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let session = store.create_session(corpus_id).expect("create session");
    let assets = store.corpus_assets(corpus_id).expect("load assets");

    store
        .upsert_external_source("4chan:w", "w", "4chan_board", "w", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:w", &remote_stream(9101, 1))
        .expect("upsert stream");
    assert!(!blocked);
    let remote_path = root.join("remote-result.png");
    std::fs::write(&remote_path, flat_png(640, 480, [180, 140, 110])).expect("write remote");
    let item_id = store
        .upsert_external_item(
            "4chan:w",
            stream_id,
            "thread 9101",
            &remote_item(9101, 9102, None),
            Some(&remote_path),
        )
        .expect("upsert external item");
    let remote_identity =
        inspect_image_bytes(&std::fs::read(&remote_path).expect("read remote identity payload"))
            .expect("inspect remote identity");
    assert!(
        matches!(
            store
                .save_external_item_identity(item_id, &remote_identity, &remote_path)
                .expect("save remote identity"),
            ExternalIdentityDisposition::Active
        ),
        "distinct remote result item should stay frontier-eligible"
    );

    store
        .conn
        .execute(
            r"
            INSERT INTO external_events (
                session_id,
                corpus_id,
                source_key,
                stream_id,
                item_id,
                local_asset_id,
                event_kind,
                created_at
            ) VALUES (?1, ?2, '4chan:w', ?3, ?4, ?5, 'rejected', 30)
            ",
            rusqlite::params![
                session.id.0,
                corpus_id.0,
                stream_id,
                item_id.0,
                assets[0].id.0
            ],
        )
        .expect("insert external result event");

    let recent_items = store
        .recent_selected_external_item_ids(session.id, 1)
        .expect("load recent external items");
    assert_eq!(recent_items, vec![item_id]);

    let recent_streams = store
        .recent_selected_external_stream_ids(session.id, 1)
        .expect("load recent external streams");
    assert_eq!(recent_streams, vec![stream_id]);

    let recent_sources = store
        .recent_selected_external_source_keys(session.id, 1)
        .expect("load recent external source keys");
    assert_eq!(recent_sources, vec!["4chan:w".to_owned()]);

    assert!(
        store
            .external_source_recently_selected(session.id, "4chan:w", 0)
            .expect("check recent source selection"),
        "non-import external result events should count as recent source exposure"
    );

    let recent_visual_keys = store
        .recent_arena_visual_keys(session.id, 2)
        .expect("load recent arena visual keys");
    assert_eq!(
        recent_visual_keys,
        vec![
            assets[0]
                .visual_key
                .clone()
                .expect("seed asset a visual key"),
            remote_identity.visual_key,
        ]
    );
}

#[test]
fn hierarchical_quality_replay_wakes_semantic_branch_from_embedding_prior() {
    let root = test_root("hierarchical-semantic-prior");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let left_path = corpus_root.join("left.png");
    let right_path = corpus_root.join("right.png");
    std::fs::write(&left_path, flat_png(640, 640, [180, 120, 90])).expect("write left");
    std::fs::write(&right_path, flat_png(640, 640, [90, 140, 210])).expect("write right");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    store
        .set_active_quality_model(&QualityModelRecord {
            formal_version: QualityFormalVersion::HierarchicalGaussianV1,
            prior_family: Default::default(),
            prior_revision: Default::default(),
            updated_at: OffsetDateTime::now_utc(),
        })
        .expect("pin hierarchical quality model");

    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");

    let assets = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(assets.len(), 2);
    let mut left = assets
        .iter()
        .find(|asset| asset.path == left_path)
        .cloned()
        .expect("left asset");
    let mut right = assets
        .iter()
        .find(|asset| asset.path == right_path)
        .cloned()
        .expect("right asset");

    let model_name = "test-dino-semantic-prior";
    store
        .save_embedding(
            &left.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
            },
        )
        .expect("save left embedding");
    store
        .save_embedding(
            &right.id,
            &EmbeddingRecord {
                model_name: model_name.to_owned(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
            },
        )
        .expect("save right embedding");

    let mut session = store.create_session(corpus_id).expect("create session");
    store
        .rebuild_active_quality_state(model_name)
        .expect("seed hierarchical replay state");

    let mut left_quality = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(&left.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("left quality cache")
            .expect("left quality cache present")
            .payload,
    )
    .expect("decode left posterior");
    let mut right_quality = HierarchicalAssetPosterior::decode(
        &store
            .asset_quality_cache(&right.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("right quality cache")
            .expect("right quality cache present")
            .payload,
    )
    .expect("decode right posterior");
    let mut session_quality = HierarchicalSessionPosterior::decode(
        &store
            .session_quality_cache(session.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("session quality cache")
            .expect("session quality cache present")
            .payload,
    )
    .expect("decode session posterior");

    assert!(
        left_quality
            .mood_loading_mean
            .iter()
            .chain(right_quality.mood_loading_mean.iter())
            .any(|value| value.abs() > 1e-5),
        "semantic prior should seed nonzero asset loadings"
    );
    assert!(
        session_quality
            .semantic_mood_mean
            .iter()
            .all(|value| value.abs() <= 1e-6),
        "session semantic mood should start at zero"
    );

    let left_before = left.clone();
    let right_before = right.clone();
    let left_utility =
        hierarchical_test_utility_mean(&left, &left_quality, &session_quality, 0.0, false);
    let right_utility =
        hierarchical_test_utility_mean(&right, &right_quality, &session_quality, 0.0, false);
    let delta_mean = left_utility - right_utility;
    let delta_variance = hierarchical_test_utility_variance(&left_quality, &session_quality)
        + hierarchical_test_utility_variance(&right_quality, &session_quality);
    let moments =
        gaussian_duel_moment_match(delta_mean, delta_variance, 1.0, HIERARCHICAL_DUEL_BETA)
            .expect("hierarchical duel moments");

    diagonal_adf_update(
        &mut left_quality.baseline_mean,
        &mut left_quality.baseline_variance,
        1.0,
        1.0,
        moments,
    );
    diagonal_adf_update(
        &mut right_quality.baseline_mean,
        &mut right_quality.baseline_variance,
        -1.0,
        1.0,
        moments,
    );
    for axis in 0..crate::model::LATENT_DIM {
        diagonal_adf_update(
            &mut left_quality.mood_loading_mean[axis],
            &mut left_quality.mood_loading_variance[axis],
            session_quality.semantic_mood_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut right_quality.mood_loading_mean[axis],
            &mut right_quality.mood_loading_variance[axis],
            -session_quality.semantic_mood_mean[axis],
            1.0,
            moments,
        );
        diagonal_adf_update(
            &mut session_quality.semantic_mood_mean[axis],
            &mut session_quality.semantic_mood_variance[axis],
            left_quality.mood_loading_mean[axis] - right_quality.mood_loading_mean[axis],
            1.0,
            moments,
        );
    }

    left.compare_count += 1;
    left.win_count += 1;
    right.compare_count += 1;
    session.comparisons += 1;
    session.mood = session_quality.semantic_mood_mean;
    hierarchical_test_sync_asset_record(&mut left, &left_quality);
    hierarchical_test_sync_asset_record(&mut right, &right_quality);

    store
        .persist_hierarchical_duel_step(
            &session,
            &left_before,
            &right_before,
            &left,
            &right,
            &left.id,
            left_utility,
            right_utility,
            None,
            &hierarchical_test_asset_cache(&left_quality),
            &hierarchical_test_asset_cache(&right_quality),
            &hierarchical_test_session_cache(&session_quality),
        )
        .expect("persist hierarchical duel step");

    store
        .rebuild_active_quality_state(model_name)
        .expect("rebuild hierarchical replay state");

    let restored_session_quality = HierarchicalSessionPosterior::decode(
        &store
            .session_quality_cache(session.id, QualityFormalVersion::HierarchicalGaussianV1)
            .expect("restored session cache")
            .expect("restored session cache present")
            .payload,
    )
    .expect("decode restored session posterior");

    assert!(
        restored_session_quality
            .semantic_mood_mean
            .iter()
            .any(|value| value.abs() > 1e-4),
        "semantic branch should wake after replaying one duel"
    );
}

#[test]
fn rejecting_remote_tombstones_future_reposts_by_visual_identity() {
    let root = test_root("external-identity-tombstone");
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
        .upsert_external_stream("4chan:s", &remote_stream(1001, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(240, 320, [180, 120, 90]);
    let identity = inspect_image_bytes(&bytes).expect("inspect remote identity");
    let cache_one = root.join("cached-one.png");
    std::fs::write(&cache_one, &bytes).expect("write first cache");

    let first_item = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1001",
            &remote_item(1001, 2001, None),
            Some(&cache_one),
        )
        .expect("upsert first item");
    assert_eq!(
        store
            .save_external_item_identity(first_item, &identity, &cache_one)
            .expect("save first identity"),
        ExternalIdentityDisposition::Active
    );
    store
        .reject_external_item(
            session_id,
            corpus_id,
            first_item,
            None,
            crate::model::ExternalEventKind::Rejected,
        )
        .expect("reject first item");

    let cache_two = root.join("cached-two.png");
    std::fs::write(&cache_two, &bytes).expect("write repost cache");
    let repost = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1001",
            &remote_item(1001, 2002, None),
            Some(&cache_two),
        )
        .expect("upsert repost");
    assert_eq!(
        store
            .save_external_item_identity(repost, &identity, &cache_two)
            .expect("save repost identity"),
        ExternalIdentityDisposition::Tombstoned,
        "repost should be pre-hidden by remote tombstone"
    );
    assert!(
        store
            .external_item_hidden(repost)
            .expect("load repost hidden"),
        "repost row should be hidden"
    );
}

#[test]
fn rejecting_remote_tombstones_matching_corpus_assets_by_visual_identity() {
    let root = test_root("external-identity-tombstone-hides-corpus");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let local_path = corpus_root.join("local.png");
    std::fs::write(&local_path, flat_png(240, 320, [180, 120, 90])).expect("write local png");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let embedder = OnnxEngine::disabled_for_tests();
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let session_id = store.create_session(corpus_id).expect("create session").id;
    let local_asset = store
        .corpus_assets(corpus_id)
        .expect("load corpus assets")
        .into_iter()
        .next()
        .expect("corpus asset present");
    assert!(!local_asset.hidden, "fresh local asset should be visible");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(1001, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = std::fs::read(&local_path).expect("read local bytes");
    let identity = inspect_image_bytes(&bytes).expect("inspect remote identity");
    let cache_path = root.join("cached.png");
    std::fs::write(&cache_path, &bytes).expect("write remote cache");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1001",
            &remote_item(1001, 2001, None),
            Some(&cache_path),
        )
        .expect("upsert remote item");
    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save remote identity"),
        ExternalIdentityDisposition::Resolved(local_asset.id.clone()),
        "matching remote should resolve onto the existing corpus asset",
    );
    store
        .reject_external_item(
            session_id,
            corpus_id,
            item_id,
            Some(&local_asset.id),
            crate::model::ExternalEventKind::Rejected,
        )
        .expect("reject remote item");

    let reloaded = store
        .corpus_asset(corpus_id, &local_asset.id)
        .expect("reload tombstoned corpus asset")
        .expect("tombstoned corpus asset present");
    assert!(
        reloaded.hidden,
        "visual tombstone should hide matching corpus assets from visibility"
    );
}

#[test]
fn imported_remote_leaves_frontier_even_if_still_visible() {
    let root = test_root("external-import-frontier");
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
        .upsert_external_stream("4chan:s", &remote_stream(1002, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(256, 256, [100, 140, 200]);
    let cache_path = root.join("frontier-cached.png");
    std::fs::write(&cache_path, &bytes).expect("write cache");
    let identity = inspect_image_bytes(&bytes).expect("inspect remote identity");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1002",
            &remote_item(1002, 2003, None),
            Some(&cache_path),
        )
        .expect("upsert item");
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
    assert!(
        store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier before import"),
        "candidate should be frontier-ready before import"
    );

    let import_path = corpus_root.join("imported.png");
    std::fs::write(&import_path, &bytes).expect("write import");
    let asset_id = store
        .ingest_external_import(corpus_id, &import_path, &bytes, 0, None)
        .expect("ingest imported asset");
    store
        .link_external_import(session_id, corpus_id, item_id, &asset_id)
        .expect("link external import");

    assert!(
        !store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier after import"),
        "imported remote should leave frontier"
    );
}

#[test]
fn remote_repost_of_existing_asset_resolves_before_frontier() {
    let root = test_root("remote-resolves-existing-asset");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    let bytes = flat_png(256, 256, [135, 80, 210]);
    let import_path = corpus_root.join("canonical.png");
    std::fs::write(&import_path, &bytes).expect("write canonical import");
    let asset_id = store
        .ingest_external_import(corpus_id, &import_path, &bytes, 0, None)
        .expect("ingest canonical asset");

    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(1004, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let cache_path = root.join("remote-repost.png");
    std::fs::write(&cache_path, &bytes).expect("write repost cache");
    let identity = inspect_image_bytes(&bytes).expect("inspect repost identity");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1004",
            &remote_item(1004, 2005, None),
            Some(&cache_path),
        )
        .expect("upsert repost");

    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save repost identity"),
        ExternalIdentityDisposition::Resolved(asset_id.clone())
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

    assert_eq!(
        store
            .external_item_resolved_asset_id(item_id)
            .expect("load resolved asset"),
        Some(asset_id)
    );
    assert!(
        !store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("resolved repost frontier state"),
        "resolved repost should never become frontier-ready"
    );
}

#[test]
fn local_ingest_resolves_existing_remote_visual_duplicate() {
    let root = test_root("local-ingest-resolves-remote-duplicate");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .upsert_external_source("4chan:s", "s", "4chan_board", "s", None)
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("4chan:s", &remote_stream(1005, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let bytes = flat_png(320, 240, [75, 130, 215]);
    let identity = inspect_image_bytes(&bytes).expect("inspect remote identity");
    let cache_path = root.join("pending-remote.png");
    std::fs::write(&cache_path, &bytes).expect("write pending cache");
    let item_id = store
        .upsert_external_item(
            "4chan:s",
            stream_id,
            "thread 1005",
            &remote_item(1005, 2006, None),
            Some(&cache_path),
        )
        .expect("upsert pending remote");
    assert_eq!(
        store
            .save_external_item_identity(item_id, &identity, &cache_path)
            .expect("save pending identity"),
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
        .expect("save pending embedding");
    assert!(
        store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier before local ingest"),
        "unresolved remote should initially be frontier-ready"
    );

    let import_path = corpus_root.join("fresh-local.png");
    std::fs::write(&import_path, &bytes).expect("write fresh local import");
    let asset_id = store
        .ingest_external_import(corpus_id, &import_path, &bytes, 0, None)
        .expect("ingest local asset");

    assert_eq!(
        store
            .external_item_resolved_asset_id(item_id)
            .expect("resolved remote after local ingest"),
        Some(asset_id)
    );
    assert!(
        !store
            .external_item_frontier_ready(item_id, "test-model")
            .expect("frontier after local ingest"),
        "local ingest should withdraw visually identical remotes from frontier"
    );
}

#[test]
fn withdrawing_missing_local_directory_items_is_reversible() {
    let root = test_root("local-directory-withdraw");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source(
            "local:test",
            "dir:dump",
            "local_directory",
            "/tmp/dump",
            None,
        )
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("local:test", &remote_stream(9001, 2))
        .expect("upsert stream");
    assert!(!blocked);

    let live_path = root.join("live.png");
    let missing_path = root.join("missing.png");
    std::fs::write(&live_path, flat_png(128, 128, [10, 20, 30])).expect("write live path");
    std::fs::write(&missing_path, flat_png(128, 128, [40, 50, 60])).expect("write missing path");

    let live_item = store
        .upsert_external_item(
            "local:test",
            stream_id,
            "thread 9001",
            &remote_item(9001, 111, None),
            Some(&live_path),
        )
        .expect("upsert live item");
    let missing_item = store
        .upsert_external_item(
            "local:test",
            stream_id,
            "thread 9001",
            &remote_item(9001, 222, None),
            Some(&missing_path),
        )
        .expect("upsert missing item");

    let withdrawn = store
        .withdraw_missing_external_items("local:test", &[111])
        .expect("withdraw missing items");
    assert_eq!(withdrawn, 1);
    assert!(
        store
            .remote_item(missing_item)
            .expect("load missing item")
            .is_none(),
        "withdrawn item should leave the frontier cleanly"
    );
    assert!(
        store
            .remote_item(live_item)
            .expect("load live item")
            .is_some(),
        "live item should remain available"
    );

    let reappeared = store
        .upsert_external_item(
            "local:test",
            stream_id,
            "thread 9001",
            &remote_item(9001, 222, None),
            Some(&missing_path),
        )
        .expect("re-upsert missing item");
    assert_eq!(reappeared, missing_item);
    assert!(
        store
            .remote_item(missing_item)
            .expect("load restored item")
            .is_some(),
        "reappearing path should become available again"
    );
}

#[test]
fn withdrawing_specific_external_items_clears_cached_path() {
    let root = test_root("withdraw-external-items");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");
    let store = Store::open(&db_path).expect("open store");

    store
        .upsert_external_source(
            "local:test",
            "dir:dump",
            "local_directory",
            "/tmp/dump",
            None,
        )
        .expect("upsert source");
    let (stream_id, blocked) = store
        .upsert_external_stream("local:test", &remote_stream(9002, 1))
        .expect("upsert stream");
    assert!(!blocked);

    let item_path = root.join("candidate.png");
    std::fs::write(&item_path, flat_png(128, 128, [70, 80, 90])).expect("write candidate");
    let item_id = store
        .upsert_external_item(
            "local:test",
            stream_id,
            "thread 9002",
            &remote_item(9002, 333, None),
            Some(&item_path),
        )
        .expect("upsert item");

    let withdrawn = store
        .withdraw_external_items_from_frontier(&[item_id])
        .expect("withdraw explicit item");
    assert_eq!(withdrawn, 1);
    assert!(
        store
            .remote_item(item_id)
            .expect("load item after withdraw")
            .is_none(),
        "withdrawn item should no longer resolve as a live remote"
    );
}

#[test]
fn store_open_waits_for_busy_writer_instead_of_immediate_lock_failure() {
    let root = test_root("sqlite-busy-timeout");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let db_path = root.join("picmash.sqlite3");

    let mut store = Store::open(&db_path).expect("open primary store");
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store.create_session(corpus_id).expect("seed session");

    let tx = store.conn.transaction().expect("open write transaction");
    tx.execute(
        "UPDATE corpora SET created_at = created_at WHERE id = ?1",
        rusqlite::params![corpus_id.0],
    )
    .expect("acquire write lock");

    let db_path_thread = db_path.clone();
    let worker = std::thread::spawn(move || -> anyhow::Result<()> {
        let store = Store::open(&db_path_thread)?;
        store.create_session(corpus_id)?;
        Ok(())
    });

    std::thread::sleep(std::time::Duration::from_millis(200));
    tx.commit().expect("release write lock");

    worker
        .join()
        .expect("join blocked writer")
        .expect("secondary writer should succeed after waiting");
}

#[test]
fn maintenance_jobs_coalesce_and_survive_newer_generations() {
    let root = test_root("maintenance-coalesce");
    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");

    store
        .enqueue_maintenance_job(&MaintenanceJobSpec::keyed(
            MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
            "4chan:h",
            MaintenancePriority::Cold,
            0,
        ))
        .expect("enqueue initial job");
    store
        .enqueue_maintenance_job(&MaintenanceJobSpec::keyed(
            MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
            "4chan:h",
            MaintenancePriority::Hot,
            0,
        ))
        .expect("enqueue coalesced job");

    let claimed = store
        .claim_next_maintenance_job()
        .expect("claim coalesced job")
        .expect("job should exist");
    assert_eq!(
        claimed.kind,
        MaintenanceJobKind::ExternalFaceEmbeddingBackfill
    );
    assert_eq!(claimed.key, "4chan:h");
    assert_eq!(claimed.priority, MaintenancePriority::Hot);
    assert_eq!(claimed.generation, 2);

    store
        .enqueue_maintenance_job(&MaintenanceJobSpec::keyed(
            MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
            "4chan:h",
            MaintenancePriority::Warm,
            0,
        ))
        .expect("enqueue newer generation while running");

    store
        .complete_maintenance_job(&claimed)
        .expect("complete first claimed generation");
    assert!(
        store
            .has_pending_maintenance_job(
                MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
                "4chan:h",
            )
            .expect("pending job lookup"),
        "newer generation should remain queued after completing stale claim"
    );

    let reclaimed = store
        .claim_next_maintenance_job()
        .expect("reclaim newer generation")
        .expect("newer generation should be runnable");
    assert_eq!(reclaimed.generation, 3);
    assert_eq!(reclaimed.priority, MaintenancePriority::Hot);

    store
        .complete_maintenance_job(&reclaimed)
        .expect("complete newest generation");
    assert!(
        !store
            .has_pending_maintenance_job(
                MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
                "4chan:h",
            )
            .expect("final pending job lookup"),
        "final generation should clear the queue"
    );
}

#[test]
fn facemash_identity_candidates_exclude_hidden_corpus_assets() {
    let root = test_root("facemash-hidden-assets");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    let image_path = corpus_root.join("face.png");
    std::fs::write(&image_path, flat_png(512, 512, [140, 120, 100])).expect("write image");

    let db_path = root.join("picmash.sqlite3");
    let mut store = Store::open(&db_path).expect("open store");
    let embedder = OnnxEngine::disabled_for_tests();
    let corpus_id = store.ensure_corpus_id(&corpus_root).expect("ensure corpus");
    store
        .ingest_corpus(&corpus_root, corpus_id, &embedder)
        .expect("ingest corpus");
    let asset = store
        .corpus_assets(corpus_id)
        .expect("load assets")
        .into_iter()
        .next()
        .expect("asset present");

    let face = DetectedFace {
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
    store
        .insert_face(
            Some(&asset.id),
            None,
            "detector:test",
            &face,
            Some("aligned.png"),
            Some(("face-embed:test", &[0.1f32, 0.2, 0.3, 0.4])),
            None,
        )
        .expect("insert face");

    assert_eq!(
        store
            .facemash_identity_count(corpus_id, "detector:test", 80.0)
            .expect("facemash identity count before hide"),
        1
    );
    assert_eq!(
        store
            .facemash_identity_candidates(corpus_id, "detector:test", 80.0, 8)
            .expect("facemash candidates before hide")
            .len(),
        1
    );

    store
        .set_hidden(corpus_id, &asset.id, true)
        .expect("hide asset in corpus");

    assert_eq!(
        store
            .facemash_identity_count(corpus_id, "detector:test", 80.0)
            .expect("facemash identity count after hide"),
        0
    );
    assert!(
        store
            .facemash_identity_candidates(corpus_id, "detector:test", 80.0, 8)
            .expect("facemash candidates after hide")
            .is_empty(),
        "hidden assets must not leave identities in the facemash frontier"
    );
}

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

#[test]
fn perturbative_quality_replay_rebuilds_from_duel_truth() {
    let root = test_root("perturbative-replay");
    let corpus_root = root.join("corpus");
    std::fs::create_dir_all(&corpus_root).expect("create corpus root");
    std::fs::write(
        corpus_root.join("left.png"),
        flat_png(256, 256, [200, 140, 120]),
    )
    .expect("write left");
    std::fs::write(
        corpus_root.join("right.png"),
        flat_png(256, 256, [80, 110, 170]),
    )
    .expect("write right");

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

    let assets = store.corpus_assets(corpus_id).expect("load assets");
    assert_eq!(assets.len(), 2);
    let left = assets[0].clone();
    let right = assets[1].clone();
    let mut left_after = left.clone();
    left_after.compare_count = 1;
    left_after.win_count = 1;
    let mut right_after = right.clone();
    right_after.compare_count = 1;
    let mut session = store
        .resume_or_create_session(corpus_id, time::Duration::minutes(10))
        .expect("resume session");
    session.comparisons = 1;

    store
        .persist_duel_step(
            &session,
            &left,
            &right,
            &left_after,
            &right_after,
            &left.id,
            0.0,
            0.0,
            None,
            None,
        )
        .expect("record duel");

    let stats = store
        .rebuild_active_quality_state(embedder.model_name())
        .expect("rebuild perturbative state");
    assert_eq!(
        stats.formal_version,
        QualityFormalVersion::HierarchicalPerturbativeV3
    );

    let left_cache = store
        .asset_quality_cache(&left.id, QualityFormalVersion::HierarchicalPerturbativeV3)
        .expect("load left cache")
        .expect("missing left cache");
    let right_cache = store
        .asset_quality_cache(&right.id, QualityFormalVersion::HierarchicalPerturbativeV3)
        .expect("load right cache")
        .expect("missing right cache");
    let session_cache = store
        .session_quality_cache(session.id, QualityFormalVersion::HierarchicalPerturbativeV3)
        .expect("load session cache")
        .expect("missing session cache");

    let left_quality =
        PerturbativeAssetPosterior::decode(&left_cache.payload).expect("decode left cache");
    let right_quality =
        PerturbativeAssetPosterior::decode(&right_cache.payload).expect("decode right cache");
    let session_quality =
        PerturbativeSessionPosterior::decode(&session_cache.payload).expect("decode session cache");

    assert!(
        left_quality.baseline_mean > right_quality.baseline_mean,
        "left baseline {} should exceed right baseline {} after a left win",
        left_quality.baseline_mean,
        right_quality.baseline_mean
    );
    assert!(
        session_quality.threshold_variance > 0.0,
        "threshold variance should be positive"
    );
    assert_eq!(
        left_quality.perturbation_basis.len(),
        crate::quality::PERTURBATIVE_DIM
    );
}
