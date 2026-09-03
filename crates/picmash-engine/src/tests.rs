use std::{fs, io::Cursor, path::Path};

use anyhow::{Context as _, Result};
use image::{ImageFormat, Rgb, RgbImage};
use tempfile::tempdir;

use rusqlite::{Connection, params};

use crate::{CommandId, Engine, Fault, ThresholdJudgment, inspect_bytes};

fn write_image(path: &Path, color: [u8; 3]) -> Result<Vec<u8>> {
    let image = RgbImage::from_pixel(32, 24, Rgb(color));
    let mut bytes = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)?;
    fs::write(path, bytes)?;
    Ok(fs::read(path)?)
}

#[test]
fn catalog_seals_reuse_only_unchanged_exact_images() -> Result<()> {
    let temporary = tempdir()?;
    let corpus = temporary.path().join("corpus");
    fs::create_dir(&corpus)?;
    let a_path = corpus.join("a.png");
    let b_path = corpus.join("b.png");
    let _a = write_image(&a_path, [210, 10, 20])?;
    let _b = write_image(&b_path, [20, 190, 30])?;
    let _copied = fs::copy(&a_path, corpus.join("z-copy.png"))?;

    let database = temporary.path().join("picmash.db");
    let engine = Engine::open(&database)?;
    let cold = engine.scan(&corpus)?;
    assert_eq!(cold.reused_paths, 0);
    let before = engine.assets(cold.collection_id)?;
    let a_before = before
        .iter()
        .find(|asset| asset.occurrence.path == a_path)
        .context("first a identity")?
        .clone();
    assert_eq!(a_before.occurrence_count, 2);
    let b_before = before
        .iter()
        .find(|asset| asset.occurrence.path == b_path)
        .context("first b identity")?
        .id
        .clone();

    Connection::open(&database)?.execute("UPDATE pm_occurrences SET file_seal = NULL", [])?;
    assert_eq!(engine.scan(&corpus)?.reused_paths, 3);
    let _changed = write_image(&a_path, [40, 50, 220])?;
    let changed = engine.scan(&corpus)?;
    assert_eq!(changed.reused_paths, 2);
    let after = engine.assets(changed.collection_id)?;
    assert_ne!(
        after
            .iter()
            .find(|asset| asset.occurrence.path == a_path)
            .context("changed a identity")?
            .id,
        a_before.id
    );
    assert_eq!(
        after
            .iter()
            .find(|asset| asset.occurrence.path == b_path)
            .context("stable b identity")?
            .id,
        b_before
    );
    Ok(())
}

#[test]
fn catalog_judgment_and_preference_form_one_durable_law() -> Result<()> {
    let temporary = tempdir()?;
    let corpus = temporary.path().join("corpus");
    fs::create_dir(&corpus)?;
    let _a = write_image(&corpus.join("a.png"), [210, 10, 20])?;
    let _b = write_image(&corpus.join("b.png"), [20, 190, 30])?;
    let _c = write_image(&corpus.join("c.png"), [30, 40, 180])?;

    let database = temporary.path().join("state/picmash.db");
    let engine = Engine::open(&database)?;
    let scan = engine.scan(&corpus)?;
    assert_eq!(scan.visible_assets, 3);
    assert!(scan.failures.is_empty());

    let session = engine.start_session(scan.collection_id, "test-context-v1")?;
    let prompt = engine.propose_comparison(&session.id)?;
    let mirrored = engine.propose_comparison(&session.id)?;
    assert_eq!(prompt.left.asset_id, mirrored.right.asset_id);
    assert_eq!(prompt.right.asset_id, mirrored.left.asset_id);
    let command = CommandId::parse("duel-1")?;
    let observation =
        engine.record_comparison(&prompt.id, &prompt.left.asset_id, &command, Some(125))?;
    assert_eq!(
        engine.record_comparison(&prompt.id, &prompt.left.asset_id, &command, Some(125))?,
        observation
    );
    assert!(matches!(
        engine.record_comparison(&prompt.id, &prompt.right.asset_id, &command, Some(125)),
        Err(Fault::CommandCollision)
    ));

    engine.record_threshold(
        &session.id,
        &prompt.left,
        ThresholdJudgment::Admit,
        &CommandId::parse("threshold-1")?,
        None,
    )?;
    let views = engine.assets(scan.collection_id)?;
    let third = views
        .iter()
        .find(|asset| asset.id != prompt.left.asset_id && asset.id != prompt.right.asset_id)
        .context("third asset")?;
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
    engine.record_similarity(
        &session.id,
        &triad,
        [&triad[0].asset_id, &triad[1].asset_id],
        "fixture-representation-v1",
        &CommandId::parse("similarity-1")?,
        Some(200),
    )?;

    engine.set_favorite(
        &session.id,
        &prompt.left.asset_id,
        true,
        &CommandId::parse("favorite-on")?,
    )?;
    engine.set_favorite(
        &session.id,
        &prompt.left.asset_id,
        false,
        &CommandId::parse("favorite-off")?,
    )?;
    assert!(
        !engine
            .assets(scan.collection_id)?
            .into_iter()
            .find(|asset| asset.id == prompt.left.asset_id)
            .context("judged asset")?
            .favorite
    );

    let snapshot = engine.rebuild_preferences(scan.collection_id)?;
    let left = snapshot
        .scores
        .iter()
        .find(|score| score.asset_id == prompt.left.asset_id)
        .context("winner score")?;
    let right = snapshot
        .scores
        .iter()
        .find(|score| score.asset_id == prompt.right.asset_id)
        .context("loser score")?;
    assert!(left.score > right.score);
    assert_eq!(left.duel_count, 1);
    assert_eq!(right.duel_count, 1);

    drop(engine);
    let reopened = Engine::open(&database)?;
    assert_eq!(
        reopened
            .latest_preferences(scan.collection_id)?
            .context("preference snapshot")?
            .id,
        snapshot.id
    );
    Ok(())
}

#[test]
fn remote_archive_reserves_order_and_exact_presentation() -> Result<()> {
    let temporary = tempdir()?;
    let corpus = temporary.path().join("corpus");
    fs::create_dir(&corpus)?;
    let _a = write_image(&corpus.join("a.png"), [210, 10, 20])?;
    let _b = write_image(&corpus.join("b.png"), [20, 190, 30])?;
    let _c = write_image(&corpus.join("c.png"), [30, 40, 180])?;

    let database = temporary.path().join("picmash.db");
    let engine = Engine::open(&database)?;
    let collection = engine.scan(&corpus)?.collection_id;
    let session = engine.start_session(collection, "remote-reservation-v1")?;
    let view = engine
        .assets(collection)?
        .into_iter()
        .next()
        .context("anchor")?;
    let anchor = crate::PresentedAsset {
        asset_id: view.id,
        occurrence_id: view.occurrence.id,
        render: view.occurrence.render,
        rotation_quarters: view.occurrence.rotation_quarters,
    };
    let remote_command = CommandId::parse("remote-duel")?;
    let reserved = engine.reserve_promoted_comparison(
        &session.id,
        "remote-item-1",
        &anchor,
        0,
        crate::DuelVictor::Challenger,
        &remote_command,
        125,
    )?;
    assert_eq!(
        Connection::open(&database)?.query_row(
            "SELECT COUNT(*) FROM pm_asset_duels WHERE observation_id = ?1",
            [reserved.get()],
            |row| row.get::<_, i64>(0),
        )?,
        0
    );

    let local_prompt = engine.propose_comparison(&session.id)?;
    let later = engine.record_comparison(
        &local_prompt.id,
        &local_prompt.left.asset_id,
        &CommandId::parse("later-local-duel")?,
        Some(80),
    )?;
    assert!(reserved < later);
    let _rotation = engine.rotate_occurrence(anchor.occurrence_id, 1)?;
    engine.end_session(&session.id)?;

    let challenger_path = corpus.join("remote.png");
    let _remote = write_image(&challenger_path, [220, 160, 20])?;
    let challenger = engine.ingest_occurrence(collection, &challenger_path)?;
    assert_eq!(
        engine.record_promoted_comparison(
            &session.id,
            "remote-item-1",
            &anchor,
            &challenger,
            0,
            crate::DuelVictor::Challenger,
            &remote_command,
            125,
        )?,
        reserved
    );
    let (left_rotation, winner) = Connection::open(&database)?.query_row(
        "SELECT d.left_rotation_quarters, d.winner_asset_id
         FROM pm_asset_duels d WHERE d.observation_id = ?1",
        [reserved.get()],
        |row| Ok((row.get::<_, u8>(0)?, row.get::<_, String>(1)?)),
    )?;
    assert_eq!(left_rotation, anchor.rotation_quarters);
    assert_eq!(winner, challenger.as_str());

    let favorite_session = engine.start_session(collection, "remote-favorite-v1")?;
    let favorite_command = CommandId::parse("remote-favorite")?;
    let favorite = engine.reserve_promoted_favorite(
        &favorite_session.id,
        "remote-item-2",
        &favorite_command,
    )?;
    engine.end_session(&favorite_session.id)?;
    assert_eq!(
        engine.set_promoted_favorite(
            &favorite_session.id,
            "remote-item-2",
            &challenger,
            &favorite_command,
        )?,
        favorite
    );
    assert!(
        engine
            .assets(collection)?
            .into_iter()
            .find(|asset| asset.id == challenger)
            .context("promoted favorite")?
            .favorite
    );
    Ok(())
}

#[test]
fn legacy_import_is_read_only_idempotent_and_rejects_old_derived_state() -> Result<()> {
    let temporary = tempdir()?;
    let corpus = temporary.path().join("legacy-corpus");
    fs::create_dir(&corpus)?;
    let a_path = corpus.join("a.png");
    let b_path = corpus.join("b.png");
    let a_bytes = write_image(&a_path, [200, 20, 10])?;
    let b_bytes = write_image(&b_path, [10, 30, 210])?;
    let a_identity = inspect_bytes(&a_bytes)?;
    let b_identity = inspect_bytes(&b_bytes)?;
    let legacy = temporary.path().join("legacy.db");
    let connection = Connection::open(&legacy)?;
    connection.execute_batch(
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
            CREATE TABLE asset_external_provenance(asset_id TEXT);
            ",
    )?;
    let legacy_render =
        |render: &crate::RenderDigest| render.as_str().replacen("rgba-v1:", "render:v1:", 1);
    connection.execute(
        "INSERT INTO corpora VALUES (1, ?1, 1)",
        [corpus.to_string_lossy().as_ref()],
    )?;
    connection.execute(
        "INSERT INTO assets VALUES ('a', 1, ?1, 0), ('b', 1, ?2, 1)",
        params![
            legacy_render(&a_identity.render),
            legacy_render(&b_identity.render)
        ],
    )?;
    connection.execute(
        "INSERT INTO corpus_assets VALUES
                (1, ?1, 'a', ?2, 32, 24, ?3, 0),
                (1, ?4, 'b', ?5, 32, 24, ?6, 0)",
        params![
            a_path.to_string_lossy().as_ref(),
            a_identity.blob.as_str().trim_start_matches("blake3:"),
            i64::try_from(a_identity.byte_len)?,
            b_path.to_string_lossy().as_ref(),
            b_identity.blob.as_str().trim_start_matches("blake3:"),
            i64::try_from(b_identity.byte_len)?,
        ],
    )?;
    connection.execute_batch(
        "
            INSERT INTO sessions VALUES (1, 1, 1, NULL);
            INSERT INTO comparisons VALUES (1, 1, 'a', 'b', 'a', 2);
            INSERT INTO nudge_events VALUES (1, 1, 'a', 1.0, 3);
            INSERT INTO heart_events VALUES (1, 1, 'a', 1, 4);
            INSERT INTO heart_events VALUES (2, 1, 'a', 0, 5);
            INSERT INTO asset_external_provenance VALUES ('a');
            ",
    )?;
    drop(connection);

    let engine = Engine::open(temporary.path().join("engine.db"))?;
    let report = engine.import_legacy(&legacy)?;
    assert_eq!(report.imported_assets, 2);
    assert_eq!(report.imported_observations, 4);
    assert_eq!(report.ambiguous_observations, 0);
    assert!(!report.already_imported);
    let repeated = engine.import_legacy(&legacy)?;
    assert!(repeated.already_imported);

    let collection = engine.scan(&corpus)?;
    let assets = engine.assets(collection.collection_id)?;
    assert_eq!(assets.len(), 2);
    assert!(assets.iter().all(|asset| !asset.favorite));
    let preference = engine.rebuild_preferences(collection.collection_id)?;
    assert!(
        preference
            .scores
            .iter()
            .find(|score| score.asset_id.as_str() == "a")
            .context("winner")?
            .score
            > preference
                .scores
                .iter()
                .find(|score| score.asset_id.as_str() == "b")
                .context("loser")?
                .score
    );
    Ok(())
}
