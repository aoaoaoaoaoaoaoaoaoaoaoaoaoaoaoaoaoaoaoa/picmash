use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use crate::fault::{Fault, Result};

pub const SCHEMA_VERSION: i64 = 1;

pub fn configure(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::from_secs(15))?;
    connection.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;",
    )?;
    Ok(())
}

pub fn migrate(connection: &mut Connection, now_ns: i64) -> Result<()> {
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS pm_schema_migrations (
             version INTEGER PRIMARY KEY,
             applied_at_ns INTEGER NOT NULL
         ) STRICT;",
    )?;
    let current = tx.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM pm_schema_migrations",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if current > SCHEMA_VERSION {
        return Err(Fault::Corrupt(format!(
            "database schema {current} is newer than supported schema {SCHEMA_VERSION}"
        )));
    }
    if current == 0 {
        install_v1(&tx)?;
        tx.execute(
            "INSERT INTO pm_schema_migrations(version, applied_at_ns) VALUES (?1, ?2)",
            params![SCHEMA_VERSION, now_ns],
        )?;
    }
    let violation = tx
        .prepare("PRAGMA foreign_key_check")?
        .query_row([], |row| row.get::<_, String>(0))
        .optional()?;
    if let Some(table) = violation {
        return Err(Fault::Corrupt(format!(
            "foreign-key violation in table {table}"
        )));
    }
    tx.commit()?;
    Ok(())
}

fn install_v1(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        r"
        CREATE TABLE pm_collections (
            id INTEGER PRIMARY KEY,
            root BLOB NOT NULL UNIQUE,
            scan_generation INTEGER NOT NULL DEFAULT 0 CHECK(scan_generation >= 0),
            catalog_revision INTEGER NOT NULL DEFAULT 0 CHECK(catalog_revision >= 0),
            created_at_ns INTEGER NOT NULL
        ) STRICT;

        CREATE TABLE pm_assets (
            id TEXT PRIMARY KEY,
            render_digest TEXT NOT NULL UNIQUE,
            identity_authority TEXT NOT NULL CHECK(identity_authority IN ('exact', 'legacy')),
            created_at_ns INTEGER NOT NULL,
            UNIQUE(id, render_digest)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_occurrences (
            id INTEGER PRIMARY KEY,
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            path BLOB NOT NULL,
            asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            blob_digest TEXT NOT NULL,
            width INTEGER NOT NULL CHECK(width >= 0),
            height INTEGER NOT NULL CHECK(height >= 0),
            byte_len INTEGER NOT NULL CHECK(byte_len >= 0),
            rotation_quarters INTEGER NOT NULL DEFAULT 0 CHECK(rotation_quarters BETWEEN 0 AND 3),
            present INTEGER NOT NULL DEFAULT 1 CHECK(present IN (0, 1)),
            last_seen_generation INTEGER NOT NULL CHECK(last_seen_generation >= 0),
            UNIQUE(collection_id, path),
            UNIQUE(id, asset_id)
        ) STRICT;

        CREATE INDEX pm_occurrences_asset
            ON pm_occurrences(collection_id, asset_id, present);

        CREATE TABLE pm_collection_assets (
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN (0, 1)),
            favorite INTEGER NOT NULL DEFAULT 0 CHECK(favorite IN (0, 1)),
            updated_at_ns INTEGER NOT NULL,
            PRIMARY KEY(collection_id, asset_id)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_scan_failures (
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            path BLOB NOT NULL,
            generation INTEGER NOT NULL CHECK(generation >= 0),
            error TEXT NOT NULL,
            PRIMARY KEY(collection_id, path)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_sessions (
            id TEXT PRIMARY KEY,
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            context_revision TEXT NOT NULL,
            started_at_ns INTEGER NOT NULL,
            ended_at_ns INTEGER
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_prompts (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES pm_sessions(id) ON DELETE CASCADE,
            left_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            right_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            left_occurrence_id INTEGER NOT NULL REFERENCES pm_occurrences(id),
            right_occurrence_id INTEGER NOT NULL REFERENCES pm_occurrences(id),
            left_render_digest TEXT NOT NULL,
            right_render_digest TEXT NOT NULL,
            left_rotation_quarters INTEGER NOT NULL CHECK(left_rotation_quarters BETWEEN 0 AND 3),
            right_rotation_quarters INTEGER NOT NULL CHECK(right_rotation_quarters BETWEEN 0 AND 3),
            policy_revision TEXT NOT NULL,
            snapshot_id INTEGER REFERENCES pm_preference_snapshots(id),
            issued_at_ns INTEGER NOT NULL,
            answered_observation_id INTEGER REFERENCES pm_observations(id),
            CHECK(left_asset_id != right_asset_id),
            FOREIGN KEY(left_occurrence_id, left_asset_id)
                REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(right_occurrence_id, right_asset_id)
                REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(left_asset_id, left_render_digest)
                REFERENCES pm_assets(id, render_digest),
            FOREIGN KEY(right_asset_id, right_render_digest)
                REFERENCES pm_assets(id, render_digest)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT NOT NULL UNIQUE,
            payload_digest TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES pm_sessions(id) ON DELETE CASCADE,
            recorded_at_ns INTEGER NOT NULL,
            prompt_revision TEXT NOT NULL,
            policy_revision TEXT,
            representation_revision TEXT,
            response_ms INTEGER CHECK(response_ms IS NULL OR response_ms >= 0),
            ordering_authority TEXT NOT NULL CHECK(ordering_authority IN ('recorded', 'legacy_inferred')),
            legacy_source TEXT,
            kind TEXT NOT NULL CHECK(kind IN ('asset_duel', 'asset_threshold', 'favorite_set', 'similarity_triad'))
        ) STRICT;

        CREATE INDEX pm_observations_session_frontier
            ON pm_observations(session_id, id);

        CREATE TABLE pm_asset_duels (
            observation_id INTEGER PRIMARY KEY REFERENCES pm_observations(id) ON DELETE CASCADE,
            prompt_id TEXT UNIQUE REFERENCES pm_prompts(id),
            left_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            right_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            winner_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            left_occurrence_id INTEGER REFERENCES pm_occurrences(id),
            right_occurrence_id INTEGER REFERENCES pm_occurrences(id),
            left_render_digest TEXT NOT NULL,
            right_render_digest TEXT NOT NULL,
            left_rotation_quarters INTEGER NOT NULL CHECK(left_rotation_quarters BETWEEN 0 AND 3),
            right_rotation_quarters INTEGER NOT NULL CHECK(right_rotation_quarters BETWEEN 0 AND 3),
            CHECK(left_asset_id != right_asset_id),
            CHECK(winner_asset_id IN (left_asset_id, right_asset_id)),
            FOREIGN KEY(left_occurrence_id, left_asset_id)
                REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(right_occurrence_id, right_asset_id)
                REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(left_asset_id, left_render_digest)
                REFERENCES pm_assets(id, render_digest),
            FOREIGN KEY(right_asset_id, right_render_digest)
                REFERENCES pm_assets(id, render_digest)
        ) STRICT, WITHOUT ROWID;

        CREATE INDEX pm_asset_duels_pair
            ON pm_asset_duels(left_asset_id, right_asset_id, observation_id);

        CREATE TABLE pm_asset_thresholds (
            observation_id INTEGER PRIMARY KEY REFERENCES pm_observations(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            occurrence_id INTEGER REFERENCES pm_occurrences(id),
            render_digest TEXT NOT NULL,
            rotation_quarters INTEGER NOT NULL CHECK(rotation_quarters BETWEEN 0 AND 3),
            judgment TEXT NOT NULL CHECK(judgment IN ('admit', 'reject')),
            FOREIGN KEY(occurrence_id, asset_id) REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(asset_id, render_digest) REFERENCES pm_assets(id, render_digest)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_favorite_events (
            observation_id INTEGER PRIMARY KEY REFERENCES pm_observations(id) ON DELETE CASCADE,
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            FOREIGN KEY(collection_id, asset_id)
                REFERENCES pm_collection_assets(collection_id, asset_id)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_similarity_triads (
            observation_id INTEGER PRIMARY KEY REFERENCES pm_observations(id) ON DELETE CASCADE,
            a_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            b_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            c_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            nearest_low_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            nearest_high_asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            a_occurrence_id INTEGER REFERENCES pm_occurrences(id),
            b_occurrence_id INTEGER REFERENCES pm_occurrences(id),
            c_occurrence_id INTEGER REFERENCES pm_occurrences(id),
            a_render_digest TEXT NOT NULL,
            b_render_digest TEXT NOT NULL,
            c_render_digest TEXT NOT NULL,
            a_rotation_quarters INTEGER NOT NULL CHECK(a_rotation_quarters BETWEEN 0 AND 3),
            b_rotation_quarters INTEGER NOT NULL CHECK(b_rotation_quarters BETWEEN 0 AND 3),
            c_rotation_quarters INTEGER NOT NULL CHECK(c_rotation_quarters BETWEEN 0 AND 3),
            CHECK(a_asset_id != b_asset_id AND a_asset_id != c_asset_id AND b_asset_id != c_asset_id),
            CHECK(nearest_low_asset_id < nearest_high_asset_id),
            CHECK(nearest_low_asset_id IN (a_asset_id, b_asset_id, c_asset_id)),
            CHECK(nearest_high_asset_id IN (a_asset_id, b_asset_id, c_asset_id)),
            FOREIGN KEY(a_occurrence_id, a_asset_id) REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(b_occurrence_id, b_asset_id) REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(c_occurrence_id, c_asset_id) REFERENCES pm_occurrences(id, asset_id),
            FOREIGN KEY(a_asset_id, a_render_digest) REFERENCES pm_assets(id, render_digest),
            FOREIGN KEY(b_asset_id, b_render_digest) REFERENCES pm_assets(id, render_digest),
            FOREIGN KEY(c_asset_id, c_render_digest) REFERENCES pm_assets(id, render_digest)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_preference_snapshots (
            id INTEGER PRIMARY KEY,
            collection_id INTEGER NOT NULL REFERENCES pm_collections(id) ON DELETE CASCADE,
            model_revision TEXT NOT NULL,
            observation_frontier INTEGER NOT NULL CHECK(observation_frontier >= 0),
            catalog_revision INTEGER NOT NULL CHECK(catalog_revision >= 0),
            created_at_ns INTEGER NOT NULL,
            training_duels INTEGER,
            held_out_duels INTEGER,
            held_out_log_loss REAL,
            held_out_accuracy REAL,
            UNIQUE(collection_id, model_revision, observation_frontier, catalog_revision)
        ) STRICT;

        CREATE TABLE pm_preference_scores (
            snapshot_id INTEGER NOT NULL REFERENCES pm_preference_snapshots(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES pm_assets(id),
            score REAL NOT NULL,
            duel_count INTEGER NOT NULL CHECK(duel_count >= 0),
            PRIMARY KEY(snapshot_id, asset_id)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE pm_legacy_imports (
            source_fingerprint TEXT PRIMARY KEY,
            source_path BLOB NOT NULL,
            imported_at_ns INTEGER NOT NULL,
            imported_assets INTEGER NOT NULL CHECK(imported_assets >= 0),
            imported_observations INTEGER NOT NULL CHECK(imported_observations >= 0),
            ambiguous_observations INTEGER NOT NULL CHECK(ambiguous_observations >= 0)
        ) STRICT, WITHOUT ROWID;
        ",
    )?;
    Ok(())
}
