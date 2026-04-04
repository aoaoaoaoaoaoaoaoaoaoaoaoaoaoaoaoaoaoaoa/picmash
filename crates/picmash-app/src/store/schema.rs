use super::faces::{assign_face_identity_tx, create_face_identity_tx};
use super::*;

impl Store {
    pub(super) fn init_schema(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            r"
            CREATE TABLE IF NOT EXISTS corpora (
                id INTEGER PRIMARY KEY,
                root_path TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                preferred_blob_id TEXT,
                visual_key TEXT,
                pixel_width INTEGER NOT NULL DEFAULT 0,
                pixel_height INTEGER NOT NULL DEFAULT 0,
                rotation_quarters INTEGER NOT NULL DEFAULT 0,
                alpha REAL NOT NULL DEFAULT 0.0,
                c0 REAL NOT NULL DEFAULT 0.0,
                c1 REAL NOT NULL DEFAULT 0.0,
                c2 REAL NOT NULL DEFAULT 0.0,
                heart_count INTEGER NOT NULL DEFAULT 0,
                hearted INTEGER NOT NULL DEFAULT 0,
                compare_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS corpus_assets (
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                blob_id TEXT,
                blob_width INTEGER NOT NULL DEFAULT 0,
                blob_height INTEGER NOT NULL DEFAULT 0,
                blob_bytes INTEGER NOT NULL DEFAULT 0,
                hidden INTEGER NOT NULL DEFAULT 0,
                last_seen_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (corpus_id, path)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (asset_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                started_at INTEGER NOT NULL,
                last_touched_at INTEGER NOT NULL DEFAULT 0,
                ended_at INTEGER,
                z0 REAL NOT NULL DEFAULT 0.0,
                z1 REAL NOT NULL DEFAULT 0.0,
                z2 REAL NOT NULL DEFAULT 0.0,
                frontier REAL NOT NULL DEFAULT 0.0,
                comparisons INTEGER NOT NULL DEFAULT 0,
                nudges INTEGER NOT NULL DEFAULT 0,
                hearts INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                left_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                right_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                winner_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                left_utility REAL NOT NULL,
                right_utility REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_asset_offsets (
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                offset REAL NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS session_asset_hearts (
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS session_subsource_locks (
                session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                source_key TEXT NOT NULL,
                stream_id INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS asset_domain_labels (
                asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nudge_events (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                direction REAL NOT NULL,
                created_at INTEGER NOT NULL,
                utility REAL NOT NULL,
                frontier REAL NOT NULL,
                signal REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS heart_events (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                active INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_embedding_heads (
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                weights BLOB NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS quality_model_registry (
                slot TEXT PRIMARY KEY,
                formal_version TEXT NOT NULL,
                prior_family TEXT NOT NULL,
                prior_revision TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quality_prior_artifacts (
                formal_version TEXT NOT NULL,
                prior_family TEXT NOT NULL,
                prior_revision TEXT NOT NULL,
                artifact_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, prior_family, prior_revision, artifact_key)
            );

            CREATE TABLE IF NOT EXISTS maintenance_jobs (
                kind TEXT NOT NULL,
                job_key TEXT NOT NULL,
                priority INTEGER NOT NULL,
                next_run_at INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                running_generation INTEGER,
                attempts INTEGER NOT NULL,
                last_error TEXT,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (kind, job_key)
            );

            CREATE INDEX IF NOT EXISTS idx_maintenance_jobs_due
                ON maintenance_jobs (running_generation, priority, next_run_at, updated_at);

            CREATE TABLE IF NOT EXISTS quality_asset_cache (
                formal_version TEXT NOT NULL,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, asset_id)
            );

            CREATE TABLE IF NOT EXISTS quality_session_cache (
                formal_version TEXT NOT NULL,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, session_id)
            );

            CREATE TABLE IF NOT EXISTS quality_subject_cache (
                formal_version TEXT NOT NULL,
                identity_id INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, identity_id)
            );

            CREATE TABLE IF NOT EXISTS quality_external_item_cache (
                formal_version TEXT NOT NULL,
                item_id INTEGER NOT NULL REFERENCES external_items(id) ON DELETE CASCADE,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, item_id)
            );

            CREATE TABLE IF NOT EXISTS quality_replay_cursors (
                formal_version TEXT NOT NULL,
                cursor_key TEXT NOT NULL,
                cursor_value INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (formal_version, cursor_key)
            );

            CREATE TABLE IF NOT EXISTS asset_quality_features (
                asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
                extractor_revision TEXT NOT NULL,
                technical_payload TEXT NOT NULL,
                vibe_payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projection_models (
                model_name TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                bias0 REAL NOT NULL,
                bias1 REAL NOT NULL,
                bias2 REAL NOT NULL,
                weights BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS similarity_models (
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                mean BLOB NOT NULL,
                weights BLOB NOT NULL,
                kind TEXT NOT NULL DEFAULT 'linear',
                payload BLOB,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (corpus_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS similarity_triads (
                id INTEGER PRIMARY KEY,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                asset_a_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                asset_b_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                asset_c_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                chosen_pair TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_sources (
                source_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                board TEXT NOT NULL,
                last_scanned_at INTEGER,
                last_error TEXT,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_item_quality_features (
                item_id INTEGER PRIMARY KEY REFERENCES external_items(id) ON DELETE CASCADE,
                extractor_revision TEXT NOT NULL,
                technical_payload TEXT NOT NULL,
                vibe_payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_streams (
                id INTEGER PRIMARY KEY,
                source_key TEXT NOT NULL REFERENCES external_sources(source_key) ON DELETE CASCADE,
                thread_no INTEGER NOT NULL,
                title TEXT NOT NULL,
                semantic_slug TEXT NOT NULL DEFAULT '',
                last_modified INTEGER NOT NULL,
                reply_count INTEGER NOT NULL DEFAULT 0,
                image_count INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                blocked INTEGER NOT NULL DEFAULT 0,
                last_seen_at INTEGER NOT NULL,
                last_scanned_at INTEGER,
                selected_count INTEGER NOT NULL DEFAULT 0,
                reject_count INTEGER NOT NULL DEFAULT 0,
                survive_count INTEGER NOT NULL DEFAULT 0,
                import_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                loss_count INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                UNIQUE (source_key, thread_no)
            );

            CREATE TABLE IF NOT EXISTS external_items (
                id INTEGER PRIMARY KEY,
                source_key TEXT NOT NULL REFERENCES external_sources(source_key) ON DELETE CASCADE,
                stream_id INTEGER NOT NULL REFERENCES external_streams(id) ON DELETE CASCADE,
                thread_no INTEGER NOT NULL,
                post_no INTEGER NOT NULL,
                stream_title TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                image_url TEXT NOT NULL,
                thumb_url TEXT NOT NULL,
                ext TEXT NOT NULL,
                md5 TEXT,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                cached_path TEXT,
                blob_id TEXT,
                visual_key TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                embedding BLOB,
                rotation_quarters INTEGER NOT NULL DEFAULT 0,
                hidden INTEGER NOT NULL DEFAULT 0,
                imported_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
                last_seen_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                selected_count INTEGER NOT NULL DEFAULT 0,
                reject_count INTEGER NOT NULL DEFAULT 0,
                survive_count INTEGER NOT NULL DEFAULT 0,
                import_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                loss_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE (source_key, post_no)
            );

            CREATE TABLE IF NOT EXISTS external_events (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                source_key TEXT NOT NULL REFERENCES external_sources(source_key) ON DELETE CASCADE,
                stream_id INTEGER NOT NULL REFERENCES external_streams(id) ON DELETE CASCADE,
                item_id INTEGER NOT NULL REFERENCES external_items(id) ON DELETE CASCADE,
                local_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
                event_kind TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS asset_external_provenance (
                asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
                source_key TEXT NOT NULL REFERENCES external_sources(source_key) ON DELETE CASCADE,
                stream_id INTEGER NOT NULL REFERENCES external_streams(id) ON DELETE CASCADE,
                item_id INTEGER NOT NULL UNIQUE REFERENCES external_items(id) ON DELETE CASCADE,
                imported_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_item_tombstones (
                blob_id TEXT NOT NULL UNIQUE,
                visual_key TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            );
            ",
        )?;
        self.ensure_column("sessions", "last_touched_at", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("sessions", "ended_at", "INTEGER")?;
        self.ensure_column("sessions", "frontier", "REAL NOT NULL DEFAULT 0.0")?;
        self.ensure_column("sessions", "nudges", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("sessions", "hearts", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("assets", "preferred_blob_id", "TEXT")?;
        self.ensure_column("assets", "visual_key", "TEXT")?;
        self.ensure_column("assets", "pixel_width", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("assets", "pixel_height", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("assets", "heart_count", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("assets", "hearted", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("heart_events", "active", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column(
            "similarity_models",
            "kind",
            "TEXT NOT NULL DEFAULT 'linear'",
        )?;
        self.ensure_column("similarity_models", "payload", "BLOB")?;
        if !self.has_column("corpus_assets", "blob_id")? {
            self.migrate_corpus_assets_table()?;
        }
        self.ensure_column(
            "external_streams",
            "selected_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "reject_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "survive_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "import_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "win_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "loss_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_streams",
            "updated_at",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column("external_streams", "blocked", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("external_items", "stream_title", "TEXT NOT NULL DEFAULT ''")?;
        self.ensure_column("external_items", "cached_path", "TEXT")?;
        self.ensure_column("external_items", "blob_id", "TEXT")?;
        self.ensure_column("external_items", "visual_key", "TEXT")?;
        self.ensure_column("external_items", "embedding_model", "TEXT")?;
        self.ensure_column("external_items", "embedding_dim", "INTEGER")?;
        self.ensure_column("external_items", "embedding", "BLOB")?;
        self.ensure_column(
            "external_items",
            "rotation_quarters",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column("external_items", "hidden", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("external_items", "imported_asset_id", "TEXT")?;
        self.ensure_column(
            "external_items",
            "selected_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_items",
            "reject_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_items",
            "survive_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "external_items",
            "import_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column("external_items", "win_count", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("external_items", "loss_count", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("external_items", "updated_at", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("external_items", "clip_embedding_model", "TEXT")?;
        self.ensure_column("external_items", "clip_embedding_dim", "INTEGER")?;
        self.ensure_column("external_items", "clip_embedding", "BLOB")?;
        self.ensure_column("external_items", "face_embedding_model", "TEXT")?;
        self.ensure_column("external_items", "face_embedding_dim", "INTEGER")?;
        self.ensure_column("external_items", "face_embedding", "BLOB")?;
        self.conn
            .execute("DROP TABLE IF EXISTS session_asset_biases", [])?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_assets_asset ON corpus_assets(corpus_id, asset_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_assets_blob ON corpus_assets(blob_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_visual_key ON assets(visual_key)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_asset_domain_labels_label ON asset_domain_labels(label)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_asset_quality_features_revision ON asset_quality_features(extractor_revision)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_item_quality_features_revision ON external_item_quality_features(extractor_revision)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_streams_source_active ON external_streams(source_key, active, last_modified DESC)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_streams_source_active_blocked ON external_streams(source_key, active, blocked, last_modified DESC)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_items_source_visibility ON external_items(source_key, hidden, imported_asset_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_items_md5 ON external_items(md5) WHERE md5 IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_items_visual_key ON external_items(visual_key) WHERE visual_key IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_external_events_session_recent ON external_events(session_id, id DESC)",
            [],
        )?;
        self.conn.execute(
            r"
            UPDATE sessions
            SET last_touched_at = started_at
            WHERE last_touched_at = 0
            ",
            [],
        )?;
        self.conn.execute_batch(
            r"
            UPDATE assets
            SET hearted = 1
            WHERE hearted = 0
              AND (
                heart_count > 0
                OR EXISTS (
                    SELECT 1
                    FROM session_asset_hearts
                    WHERE session_asset_hearts.asset_id = assets.id
                )
                OR EXISTS (
                    SELECT 1
                    FROM heart_events
                    WHERE heart_events.asset_id = assets.id
                      AND heart_events.active != 0
                )
              );
            ",
        )?;

        self.conn.execute_batch(
            r"
            CREATE TABLE IF NOT EXISTS face_identities (
                id              INTEGER PRIMARY KEY,
                name            TEXT,
                rating_mu       REAL NOT NULL DEFAULT 1500.0,
                rating_sigma    REAL NOT NULL DEFAULT 350.0,
                compare_count   INTEGER NOT NULL DEFAULT 0,
                created_at      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS faces (
                id              INTEGER PRIMARY KEY,
                asset_id        TEXT REFERENCES assets(id) ON DELETE CASCADE,
                remote_item_id  INTEGER,
                identity_id     INTEGER REFERENCES face_identities(id) ON DELETE SET NULL,
                hidden          INTEGER NOT NULL DEFAULT 0,
                geometry_key    TEXT NOT NULL DEFAULT '',
                detector_model  TEXT NOT NULL,
                bbox_x          REAL NOT NULL,
                bbox_y          REAL NOT NULL,
                bbox_w          REAL NOT NULL,
                bbox_h          REAL NOT NULL,
                confidence      REAL NOT NULL,
                landmarks       BLOB NOT NULL,
                aligned_path    TEXT,
                embedding_model TEXT,
                embedding_dim   INTEGER,
                embedding       BLOB,
                recognition_model TEXT,
                recognition_dim INTEGER,
                recognition_embedding BLOB,
                rating_mu       REAL NOT NULL DEFAULT 1500.0,
                rating_sigma    REAL NOT NULL DEFAULT 350.0,
                compare_count   INTEGER NOT NULL DEFAULT 0,
                created_at      INTEGER NOT NULL,
                CHECK ((asset_id IS NOT NULL) != (remote_item_id IS NOT NULL))
            );

            CREATE TABLE IF NOT EXISTS face_comparisons (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                winner_face_id  INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
                loser_face_id   INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
                created_at      INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_scans (
                id              INTEGER PRIMARY KEY,
                asset_id        TEXT REFERENCES assets(id) ON DELETE CASCADE,
                remote_item_id  INTEGER REFERENCES external_items(id) ON DELETE CASCADE,
                detector_model  TEXT NOT NULL,
                created_at      INTEGER NOT NULL,
                CHECK ((asset_id IS NOT NULL) != (remote_item_id IS NOT NULL))
            );

            CREATE TABLE IF NOT EXISTS face_tombstones (
                id              INTEGER PRIMARY KEY,
                asset_id        TEXT REFERENCES assets(id) ON DELETE CASCADE,
                remote_item_id  INTEGER REFERENCES external_items(id) ON DELETE CASCADE,
                geometry_key    TEXT NOT NULL,
                created_at      INTEGER NOT NULL,
                CHECK ((asset_id IS NOT NULL) != (remote_item_id IS NOT NULL))
            );

            CREATE TABLE IF NOT EXISTS face_identity_bindings (
                id              INTEGER PRIMARY KEY,
                asset_id        TEXT REFERENCES assets(id) ON DELETE CASCADE,
                remote_item_id  INTEGER REFERENCES external_items(id) ON DELETE CASCADE,
                geometry_key    TEXT NOT NULL,
                identity_id     INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
                created_at      INTEGER NOT NULL,
                CHECK ((asset_id IS NOT NULL) != (remote_item_id IS NOT NULL))
            );

            CREATE TABLE IF NOT EXISTS face_identity_vetoes (
                id                  INTEGER PRIMARY KEY,
                identity_lo         INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
                identity_hi         INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
                recognition_model   TEXT NOT NULL,
                created_at          INTEGER NOT NULL
            );
            ",
        )?;
        self.ensure_column(
            "faces",
            "detector_model",
            "TEXT NOT NULL DEFAULT 'scrfd-10g@onnx-v1'",
        )?;
        self.ensure_column(
            "faces",
            "identity_id",
            "INTEGER REFERENCES face_identities(id) ON DELETE SET NULL",
        )?;
        self.ensure_column("faces", "hidden", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("faces", "geometry_key", "TEXT NOT NULL DEFAULT ''")?;
        self.ensure_column(
            "face_identities",
            "rating_mu",
            "REAL NOT NULL DEFAULT 1500.0",
        )?;
        self.ensure_column(
            "face_identities",
            "rating_sigma",
            "REAL NOT NULL DEFAULT 350.0",
        )?;
        self.ensure_column(
            "face_identities",
            "compare_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "face_identities",
            "created_at",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column("faces", "rating_mu", "REAL NOT NULL DEFAULT 1500.0")?;
        self.ensure_column("faces", "rating_sigma", "REAL NOT NULL DEFAULT 350.0")?;
        self.ensure_column("faces", "recognition_model", "TEXT")?;
        self.ensure_column("faces", "recognition_dim", "INTEGER")?;
        self.ensure_column("faces", "recognition_embedding", "BLOB")?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_asset ON faces(asset_id) WHERE asset_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_identity ON faces(identity_id) WHERE identity_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_remote ON faces(remote_item_id) WHERE remote_item_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_visible ON faces(hidden, detector_model, compare_count)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_geometry ON faces(asset_id, geometry_key)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_detector_model ON faces(detector_model)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_recognition_model ON faces(recognition_model)",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_identities_name ON face_identities(name) WHERE name IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_identities_duels ON face_identities(compare_count, rating_mu)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_rating_mu ON faces(rating_mu)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_comparisons_session ON face_comparisons(session_id, id DESC)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_comparisons_created ON face_comparisons(created_at, id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_scans_asset ON face_scans(asset_id) WHERE asset_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_scans_remote ON face_scans(remote_item_id) WHERE remote_item_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_scans_asset_model ON face_scans(asset_id, detector_model) WHERE asset_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_scans_remote_model ON face_scans(remote_item_id, detector_model) WHERE remote_item_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_tombstones_asset_geometry ON face_tombstones(asset_id, geometry_key) WHERE asset_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_tombstones_remote_geometry ON face_tombstones(remote_item_id, geometry_key) WHERE remote_item_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_identity_bindings_asset_geometry ON face_identity_bindings(asset_id, geometry_key) WHERE asset_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_identity_bindings_remote_geometry ON face_identity_bindings(remote_item_id, geometry_key) WHERE remote_item_id IS NOT NULL",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_identity_bindings_identity ON face_identity_bindings(identity_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_face_identity_vetoes_pair_model ON face_identity_vetoes(identity_lo, identity_hi, recognition_model)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_asset_cache_updated ON quality_asset_cache(updated_at)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_session_cache_updated ON quality_session_cache(updated_at)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_subject_cache_updated ON quality_subject_cache(updated_at)",
            [],
        )?;

        Ok(())
    }

    pub(super) fn run_bootstrap_maintenance(&self) -> anyhow::Result<()> {
        self.backfill_asset_identity_metadata()?;
        self.backfill_hidden_external_item_tombstones()?;
        self.backfill_face_identity_rows()?;
        self.backfill_face_identity_bindings()?;
        Ok(())
    }

    fn ensure_column(
        &self,
        table_name: &str,
        column_name: &str,
        column_spec: &str,
    ) -> anyhow::Result<()> {
        if self.has_column(table_name, column_name)? {
            return Ok(());
        }
        self.conn.execute(
            &format!("ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}"),
            [],
        )?;
        Ok(())
    }

    pub(super) fn has_column(&self, table_name: &str, column_name: &str) -> anyhow::Result<bool> {
        let mut stmt = self
            .conn
            .prepare(&format!("PRAGMA table_info({table_name})"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let current_name: String = row.get(1)?;
            if current_name == column_name {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn migrate_corpus_assets_table(&self) -> anyhow::Result<()> {
        let legacy_rows = {
            let mut stmt = self.conn.prepare(
                r"
                SELECT corpus_id, asset_id, path, hidden
                FROM corpus_assets
                ORDER BY corpus_id, path
                ",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };

        self.conn.execute_batch(
            r"
            DROP INDEX IF EXISTS idx_corpus_assets_path;
            ALTER TABLE corpus_assets RENAME TO corpus_assets_legacy;
            CREATE TABLE corpus_assets (
                corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                blob_id TEXT,
                blob_width INTEGER NOT NULL DEFAULT 0,
                blob_height INTEGER NOT NULL DEFAULT 0,
                blob_bytes INTEGER NOT NULL DEFAULT 0,
                hidden INTEGER NOT NULL DEFAULT 0,
                last_seen_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (corpus_id, path)
            );
            ",
        )?;

        for (corpus_id, asset_id, path, hidden) in legacy_rows {
            let inspection = fs::read(&path)
                .ok()
                .and_then(|bytes| inspect_image_bytes(&bytes).ok());
            self.conn.execute(
                r"
                INSERT INTO corpus_assets (
                    corpus_id,
                    path,
                    asset_id,
                    blob_id,
                    blob_width,
                    blob_height,
                    blob_bytes,
                    hidden,
                    last_seen_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ",
                params![
                    corpus_id,
                    path,
                    asset_id,
                    inspection
                        .as_ref()
                        .map(|identity| identity.blob_id.0.clone()),
                    inspection
                        .as_ref()
                        .map_or(0_i64, |identity| i64::from(identity.width)),
                    inspection
                        .as_ref()
                        .map_or(0_i64, |identity| i64::from(identity.height)),
                    inspection.as_ref().map_or(0_i64, |identity| {
                        i64::try_from(identity.byte_len).unwrap_or_default()
                    }),
                    hidden,
                    now_ts(),
                ],
            )?;
        }
        self.conn
            .execute("DROP TABLE corpus_assets_legacy", [])
            .context("dropping legacy corpus_assets table")?;
        Ok(())
    }

    fn backfill_asset_identity_metadata(&self) -> anyhow::Result<()> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT a.id
            FROM assets a
            WHERE a.visual_key IS NULL
               OR a.visual_key = ''
               OR a.preferred_blob_id IS NULL
               OR a.pixel_width = 0
               OR a.pixel_height = 0
            ORDER BY a.created_at ASC, a.id ASC
            ",
        )?;
        let asset_ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        for asset_id in asset_ids {
            let Some(identity) = self.best_asset_identity_source(&AssetId(asset_id.clone()))?
            else {
                continue;
            };
            self.conn.execute(
                r"
                UPDATE assets
                SET preferred_blob_id = COALESCE(preferred_blob_id, ?2),
                    visual_key = COALESCE(NULLIF(visual_key, ''), ?3),
                    pixel_width = CASE WHEN pixel_width = 0 THEN ?4 ELSE pixel_width END,
                    pixel_height = CASE WHEN pixel_height = 0 THEN ?5 ELSE pixel_height END
                WHERE id = ?1
                ",
                params![
                    asset_id,
                    identity.blob_id.0,
                    identity.visual_key.0,
                    i64::from(identity.width),
                    i64::from(identity.height),
                ],
            )?;
        }
        Ok(())
    }

    fn best_asset_identity_source(
        &self,
        asset_id: &AssetId,
    ) -> anyhow::Result<Option<ImageIdentity>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT path
            FROM corpus_assets
            WHERE asset_id = ?1
            ORDER BY blob_width * blob_height DESC, path ASC
            ",
        )?;
        let paths = stmt
            .query_map(params![asset_id.0], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for path in paths {
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            if let Ok(identity) = inspect_image_bytes(&bytes) {
                return Ok(Some(identity));
            }
        }
        Ok(None)
    }

    fn backfill_face_identity_rows(&self) -> anyhow::Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("opening face identity backfill transaction")?;

        let missing_faces = tx
            .prepare("SELECT id FROM faces WHERE identity_id IS NULL ORDER BY id ASC")?
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for face_id in missing_faces {
            let identity_id = create_face_identity_tx(&tx, None)?;
            assign_face_identity_tx(&tx, FaceId(face_id), identity_id)?;
        }

        tx.execute(
            r"
            DELETE FROM face_identities
            WHERE NOT EXISTS (
                SELECT 1 FROM faces WHERE faces.identity_id = face_identities.id
            )
            ",
            [],
        )?;

        tx.commit()
            .context("committing face identity backfill transaction")?;
        Ok(())
    }

    fn backfill_face_identity_bindings(&self) -> anyhow::Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("opening face identity binding backfill transaction")?;

        let mut stmt = tx.prepare(
            r"
            SELECT asset_id, remote_item_id, geometry_key, identity_id
            FROM faces
            WHERE identity_id IS NOT NULL
              AND geometry_key IS NOT NULL
              AND geometry_key <> ''
            ORDER BY id ASC
            ",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);

        for (asset_id, remote_item_id, geometry_key, identity_id) in rows {
            match (asset_id, remote_item_id) {
                (Some(asset_id), None) => {
                    tx.execute(
                        r"
                        INSERT OR IGNORE INTO face_identity_bindings (
                            asset_id,
                            remote_item_id,
                            geometry_key,
                            identity_id,
                            created_at
                        ) VALUES (?1, NULL, ?2, ?3, ?4)
                        ",
                        params![asset_id, geometry_key, identity_id, now_ts()],
                    )?;
                }
                (None, Some(remote_item_id)) => {
                    tx.execute(
                        r"
                        INSERT OR IGNORE INTO face_identity_bindings (
                            asset_id,
                            remote_item_id,
                            geometry_key,
                            identity_id,
                            created_at
                        ) VALUES (NULL, ?1, ?2, ?3, ?4)
                        ",
                        params![remote_item_id, geometry_key, identity_id, now_ts()],
                    )?;
                }
                _ => {}
            }
        }

        tx.commit()
            .context("committing face identity binding backfill transaction")?;
        Ok(())
    }

    fn backfill_hidden_external_item_tombstones(&self) -> anyhow::Result<()> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, cached_path, blob_id, visual_key
            FROM external_items ei
            WHERE ei.hidden = 1
              AND ei.cached_path IS NOT NULL
              AND (
                    ei.blob_id IS NULL
                 OR ei.visual_key IS NULL
                 OR NOT EXISTS (
                        SELECT 1
                        FROM external_item_tombstones t
                        WHERE t.blob_id = ei.blob_id
                    )
                 OR NOT EXISTS (
                        SELECT 1
                        FROM external_item_tombstones t
                        WHERE t.visual_key = ei.visual_key
                    )
              )
            ",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                RemoteItemId(row.get(0)?),
                PathBuf::from(row.get::<_, String>(1)?),
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;

        let mut recovered = 0usize;
        let mut propagated = 0usize;
        for row in rows {
            let (item_id, cached_path, blob_id, visual_key) = row?;
            let identity = if let (Some(blob_id), Some(visual_key)) = (blob_id, visual_key) {
                ImageIdentity {
                    blob_id: BlobId(blob_id),
                    visual_key: crate::identity::VisualKey(visual_key),
                    width: 0,
                    height: 0,
                    byte_len: 0,
                }
            } else {
                let bytes = match fs::read(&cached_path) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        warn!(
                            item_id = item_id.0,
                            path = %cached_path.display(),
                            error = %error,
                            "failed to read hidden remote for tombstone backfill"
                        );
                        continue;
                    }
                };
                let identity = match inspect_image_bytes(&bytes) {
                    Ok(identity) => identity,
                    Err(error) => {
                        warn!(
                            item_id = item_id.0,
                            path = %cached_path.display(),
                            error = %format!("{error:#}"),
                            "failed to inspect hidden remote for tombstone backfill"
                        );
                        continue;
                    }
                };
                self.conn.execute(
                    r"
                    UPDATE external_items
                    SET blob_id = ?2,
                        visual_key = ?3,
                        updated_at = ?4
                    WHERE id = ?1
                    ",
                    params![
                        item_id.0,
                        identity.blob_id.0.as_str(),
                        identity.visual_key.0.as_str(),
                        now_ts()
                    ],
                )?;
                recovered += 1;
                identity
            };
            let blob_id = identity.blob_id.0.clone();
            let visual_key = identity.visual_key.0.clone();

            self.conn.execute(
                r"
                INSERT OR IGNORE INTO external_item_tombstones (blob_id, visual_key, created_at)
                VALUES (?1, ?2, ?3)
                ",
                params![blob_id.as_str(), visual_key.as_str(), now_ts()],
            )?;
            propagated += self.conn.execute(
                r"
                UPDATE external_items
                SET hidden = 1,
                    updated_at = ?3
                WHERE hidden = 0
                  AND imported_asset_id IS NULL
                  AND (
                    blob_id = ?1
                    OR visual_key = ?2
                  )
                ",
                params![blob_id.as_str(), visual_key.as_str(), now_ts()],
            )?;
        }

        if recovered > 0 || propagated > 0 {
            info!(
                recovered,
                propagated, "backfilled hidden remote identity tombstones"
            );
        }
        Ok(())
    }
}
