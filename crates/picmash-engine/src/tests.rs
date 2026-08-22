use std::{fs, io::Cursor, path::Path};

use image::{ImageFormat, Rgb, RgbImage};
use tempfile::tempdir;

use rusqlite::{Connection, params};

use crate::{CommandId, Engine, Fault, ThresholdJudgment, inspect_bytes};

fn write_image(path: &Path, color: [u8; 3]) -> Vec<u8> {
    let image = RgbImage::from_pixel(32, 24, Rgb(color));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("encode fixture");
    fs::write(path, bytes).expect("write fixture");
    fs::read(path).expect("read fixture")
}

#[test]
fn catalog_judgment_and_preference_form_one_durable_law() {
    let temporary = tempdir().expect("temporary directory");
    let corpus = temporary.path().join("corpus");
    fs::create_dir(&corpus).expect("create corpus");
    let _ = write_image(&corpus.join("a.png"), [210, 10, 20]);
    let _ = write_image(&corpus.join("b.png"), [20, 190, 30]);
    let _ = write_image(&corpus.join("c.png"), [30, 40, 180]);

    let database = temporary.path().join("state/picmash.db");
    let engine = Engine::open(&database).expect("open engine");
    let scan = engine.scan(&corpus).expect("scan corpus");
    assert_eq!(scan.visible_assets, 3);
    assert!(scan.failures.is_empty());

    let session = engine
        .start_session(scan.collection_id, "test-context-v1")
        .expect("start session");
    let prompt = engine
        .propose_comparison(&session.id)
        .expect("propose comparison");
    let mirrored = engine
        .propose_comparison(&session.id)
        .expect("propose balanced comparison");
    assert_eq!(prompt.left.asset_id, mirrored.right.asset_id);
    assert_eq!(prompt.right.asset_id, mirrored.left.asset_id);
    let command = CommandId::parse("duel-1").expect("command");
    let observation = engine
        .record_comparison(&prompt.id, &prompt.left.asset_id, &command, Some(125))
        .expect("record comparison");
    assert_eq!(
        engine
            .record_comparison(&prompt.id, &prompt.left.asset_id, &command, Some(125))
            .expect("retry comparison"),
        observation
    );
    assert!(matches!(
        engine.record_comparison(&prompt.id, &prompt.right.asset_id, &command, Some(125)),
        Err(Fault::CommandCollision)
    ));

    engine
        .record_threshold(
            &session.id,
            &prompt.left,
            ThresholdJudgment::Admit,
            &CommandId::parse("threshold-1").expect("command"),
            None,
        )
        .expect("record threshold");
    let views = engine.assets(scan.collection_id).expect("load assets");
    let third = views
        .iter()
        .find(|asset| asset.id != prompt.left.asset_id && asset.id != prompt.right.asset_id)
        .expect("third asset");
    let triad = [
        prompt.left.clone(),
        prompt.right.clone(),
        crate::PresentedAsset {
            asset_id: third.id.clone(),
            occurrence_id: third.occurrence.id,
            render: third.occurrence.render.clone(),
            rotation_quarters: third.occurrence.rotation_quarters,
        },
    ];
    engine
        .record_similarity(
            &session.id,
            &triad,
            [&triad[0].asset_id, &triad[1].asset_id],
            "fixture-representation-v1",
            &CommandId::parse("similarity-1").expect("command"),
            Some(200),
        )
        .expect("record similarity");

    engine
        .set_favorite(
            &session.id,
            &prompt.left.asset_id,
            true,
            &CommandId::parse("favorite-on").expect("command"),
        )
        .expect("favorite asset");
    engine
        .set_favorite(
            &session.id,
            &prompt.left.asset_id,
            false,
            &CommandId::parse("favorite-off").expect("command"),
        )
        .expect("unfavorite asset");
    assert!(
        !engine
            .assets(scan.collection_id)
            .expect("reload assets")
            .into_iter()
            .find(|asset| asset.id == prompt.left.asset_id)
            .expect("judged asset")
            .favorite
    );

    let snapshot = engine
        .rebuild_preferences(scan.collection_id)
        .expect("build preference snapshot");
    let left = snapshot
        .scores
        .iter()
        .find(|score| score.asset_id == prompt.left.asset_id)
        .expect("winner score");
    let right = snapshot
        .scores
        .iter()
        .find(|score| score.asset_id == prompt.right.asset_id)
        .expect("loser score");
    assert!(left.score > right.score);
    assert_eq!(left.duel_count, 1);
    assert_eq!(right.duel_count, 1);

    drop(engine);
    let reopened = Engine::open(&database).expect("reopen engine");
    assert_eq!(
        reopened
            .latest_preferences(scan.collection_id)
            .expect("load preference snapshot")
            .expect("preference snapshot")
            .id,
        snapshot.id
    );
}

#[test]
fn legacy_import_is_read_only_idempotent_and_rejects_old_derived_state() {
    let temporary = tempdir().expect("temporary directory");
    let corpus = temporary.path().join("legacy-corpus");
    fs::create_dir(&corpus).expect("create corpus");
    let a_path = corpus.join("a.png");
    let b_path = corpus.join("b.png");
    let a_bytes = write_image(&a_path, [200, 20, 10]);
    let b_bytes = write_image(&b_path, [10, 30, 210]);
    let a_identity = inspect_bytes(&a_bytes).expect("inspect a");
    let b_identity = inspect_bytes(&b_bytes).expect("inspect b");
    let legacy = temporary.path().join("legacy.db");
    let connection = Connection::open(&legacy).expect("open legacy database");
    connection
        .execute_batch(
            "
            CREATE TABLE corpora(id INTEGER PRIMARY KEY, root_path TEXT, created_at INTEGER);
            CREATE TABLE assets(
                id TEXT PRIMARY KEY, created_at INTEGER, render_hash TEXT,
                rotation_quarters INTEGER
            );
            CREATE TABLE corpus_assets(
                corpus_id INTEGER, path TEXT, asset_id TEXT, blob_id TEXT,
                blob_width INTEGER, blob_height INTEGER, blob_bytes INTEGER, hidden INTEGER
            );
            CREATE TABLE sessions(
                id INTEGER PRIMARY KEY, corpus_id INTEGER, started_at INTEGER, ended_at INTEGER
            );
            CREATE TABLE comparisons(
                id INTEGER PRIMARY KEY, session_id INTEGER, left_asset_id TEXT,
                right_asset_id TEXT, winner_asset_id TEXT, created_at INTEGER
            );
            CREATE TABLE nudge_events(
                id INTEGER PRIMARY KEY, session_id INTEGER, asset_id TEXT,
                direction REAL, created_at INTEGER
            );
            CREATE TABLE heart_events(
                id INTEGER PRIMARY KEY, session_id INTEGER, asset_id TEXT,
                active INTEGER, created_at INTEGER
            );
            CREATE TABLE similarity_triads(
                id INTEGER PRIMARY KEY, corpus_id INTEGER, model_name TEXT,
                asset_a_id TEXT, asset_b_id TEXT, asset_c_id TEXT,
                chosen_pair TEXT, created_at INTEGER
            );
            ",
        )
        .expect("create legacy schema");
    let legacy_render =
        |render: &crate::RenderDigest| render.as_str().replacen("rgba-v1:", "render:v1:", 1);
    connection
        .execute(
            "INSERT INTO corpora VALUES (1, ?1, 1)",
            [corpus.to_string_lossy().as_ref()],
        )
        .expect("insert corpus");
    connection
        .execute(
            "INSERT INTO assets VALUES ('a', 1, ?1, 0), ('b', 1, ?2, 1)",
            params![
                legacy_render(&a_identity.render),
                legacy_render(&b_identity.render)
            ],
        )
        .expect("insert assets");
    connection
        .execute(
            "INSERT INTO corpus_assets VALUES
                (1, ?1, 'a', ?2, 32, 24, ?3, 0),
                (1, ?4, 'b', ?5, 32, 24, ?6, 0)",
            params![
                a_path.to_string_lossy().as_ref(),
                a_identity.blob.as_str().trim_start_matches("blake3:"),
                a_identity.byte_len,
                b_path.to_string_lossy().as_ref(),
                b_identity.blob.as_str().trim_start_matches("blake3:"),
                b_identity.byte_len,
            ],
        )
        .expect("insert occurrences");
    connection
        .execute_batch(
            "
            INSERT INTO sessions VALUES (1, 1, 1, NULL);
            INSERT INTO comparisons VALUES (1, 1, 'a', 'b', 'a', 2);
            INSERT INTO nudge_events VALUES (1, 1, 'a', 1.0, 3);
            INSERT INTO heart_events VALUES (1, 1, 'a', 1, 4);
            INSERT INTO heart_events VALUES (2, 1, 'a', 0, 5);
            ",
        )
        .expect("insert observations");
    drop(connection);

    let engine = Engine::open(temporary.path().join("engine.db")).expect("open engine");
    let report = engine
        .import_legacy(&legacy)
        .expect("import legacy database");
    assert_eq!(report.imported_assets, 2);
    assert_eq!(report.imported_observations, 4);
    assert_eq!(report.ambiguous_observations, 0);
    assert!(!report.already_imported);
    let repeated = engine.import_legacy(&legacy).expect("repeat import");
    assert!(repeated.already_imported);

    let collection = engine.scan(&corpus).expect("rescan imported corpus");
    let assets = engine
        .assets(collection.collection_id)
        .expect("load imported assets");
    assert_eq!(assets.len(), 2);
    assert!(assets.iter().all(|asset| !asset.favorite));
    let preference = engine
        .rebuild_preferences(collection.collection_id)
        .expect("project imported duel");
    assert!(
        preference
            .scores
            .iter()
            .find(|score| score.asset_id.as_str() == "a")
            .expect("winner")
            .score
            > preference
                .scores
                .iter()
                .find(|score| score.asset_id.as_str() == "b")
                .expect("loser")
                .score
    );
}
