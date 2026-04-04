use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Duration as StdDuration,
};

use anyhow::{Context, bail};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use time::{Duration, OffsetDateTime};
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::{
    asset_domain::{AssetDomainLabel, AssetDomainTrainingRow},
    face::{DetectedFace, FaceLandmarks},
    facemash::{
        FaceBeauty, FaceOracleCalibrationSample, FaceOracleDuelSample, FaceOracleTrainingData,
        pool_embeddings, rate_face_win,
    },
    identity::{BlobId, ImageIdentity, inspect_image_bytes, mint_asset_id},
    model::{
        AssetId, AssetRecord, CorpusId, EmbeddingRecord, ExternalEventKind, FaceId, FaceIdentityId,
        ProjectionModel, RemoteCandidate, RemoteItemId, RemoteItemRecord, SessionEmbeddingHead,
        SessionId, SessionRecord, SimilarityChoice, SimilarityModel, SimilarityObservation,
    },
    onnx::OnnxEngine,
    quality::{
        HierarchicalAssetQualityCacheV1, HierarchicalSessionQualityCacheV1, QualityFormalVersion,
        StoredAssetQualityCache, StoredSessionQualityCache, legacy_asset_quality_payload,
        legacy_session_quality_payload,
    },
    sources::{RemoteItemSnapshot, RemoteStreamSnapshot},
};

mod external;
mod faces;
mod maintenance;
mod quality;
mod schema;
#[cfg(test)]
mod tests;

pub use faces::{FaceIdentityRecord, FaceRecord, FacemashIdentityCandidate};

pub struct Store {
    conn: Connection,
}

#[derive(Debug, Clone)]
struct IngestSkipSummary {
    count: usize,
    sample_path: PathBuf,
}

#[derive(Debug, Clone)]
struct CorpusAssetRow {
    asset: AssetRecord,
    preferred_blob_id: Option<BlobId>,
    variant_blob_id: Option<BlobId>,
    blob_width: u32,
    blob_height: u32,
}

pub const FACE_DETECTION_MIN_FACE_SIDE: f32 = 24.0;
const FACEMASH_MIN_CONFIDENCE: f32 = 0.65;
const FACEMASH_MIN_EYE_SPAN_FRACTION: f32 = 0.2;
const FACEMASH_MIN_MOUTH_SPAN_FRACTION: f32 = 0.18;
const FACEMASH_MIN_FEATURE_HEIGHT_FRACTION: f32 = 0.22;

impl Store {
    pub fn open(db_path: &Path) -> anyhow::Result<Self> {
        Self::open_internal(db_path, true, false)
    }

    pub fn open_hot(db_path: &Path) -> anyhow::Result<Self> {
        Self::open_internal(db_path, false, false)
    }

    fn open_internal(
        db_path: &Path,
        initialize_schema: bool,
        run_bootstrap_maintenance: bool,
    ) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating database directory {}", parent.display()))?;
        }

        let conn = Connection::open(db_path)
            .with_context(|| format!("opening SQLite database {}", db_path.display()))?;
        conn.busy_timeout(StdDuration::from_secs(15))
            .context("configuring SQLite busy timeout")?;
        conn.pragma_update(None, "journal_mode", "wal")
            .context("enabling WAL mode")?;
        conn.pragma_update(None, "foreign_keys", "on")
            .context("enabling foreign keys")?;

        let store = Self { conn };
        if initialize_schema {
            store.init_schema()?;
        }
        if run_bootstrap_maintenance {
            store.run_bootstrap_maintenance()?;
        }
        Ok(store)
    }

    pub fn devour_bootstrap_maintenance(&self) -> anyhow::Result<()> {
        self.run_bootstrap_maintenance()
    }

    /// Ensure the corpus exists in the DB without scanning images.
    /// Returns the corpus ID immediately so the app can start serving.
    pub fn ensure_corpus_id(&self, root_path: &Path) -> anyhow::Result<CorpusId> {
        let root_path = root_path
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root_path.display()))?;
        self.ensure_corpus(&root_path)
    }

    fn ensure_corpus(&self, root_path: &Path) -> anyhow::Result<CorpusId> {
        let root = root_path.to_string_lossy();
        self.conn.execute(
            r"
            INSERT INTO corpora (root_path, created_at)
            VALUES (?1, ?2)
            ON CONFLICT(root_path) DO NOTHING
            ",
            params![root.as_ref(), now_ts()],
        )?;
        self.conn
            .query_row(
                "SELECT id FROM corpora WHERE root_path = ?1",
                params![root.as_ref()],
                |row| row.get::<_, i64>(0).map(CorpusId),
            )
            .map_err(Into::into)
    }

    /// Full corpus ingest: walk disk, hash, decode new images, embed.
    /// Safe to call on an already-populated corpus — unchanged images are
    /// detected by BLAKE3 and skipped without decoding.
    pub fn ingest_corpus(
        &mut self,
        root_path: &Path,
        corpus_id: CorpusId,
        embedder: &OnnxEngine,
    ) -> anyhow::Result<()> {
        if !root_path.exists() {
            bail!("corpus root does not exist: {}", root_path.display());
        }

        let root_path = root_path
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root_path.display()))?;
        let model_name = embedder.model_name().to_owned();
        let mut dino_available = embedder.enabled();
        let mut skipped_identity = BTreeMap::<String, IngestSkipSummary>::new();

        let image_paths: Vec<PathBuf> = WalkDir::new(&root_path)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .filter(|path| is_supported_image(path))
            .collect();
        let total = image_paths.len();
        info!(total, "corpus walk complete, ingesting images");

        let mut ingested = 0usize;
        let mut unchanged = 0usize;
        let mut embed_count = 0usize;
        let log_interval = (total / 20).max(50);

        for path in image_paths {
            let bytes = fs::read(&path)
                .with_context(|| format!("reading image bytes from {}", path.display()))?;
            let blob_id = BlobId(blake3::hash(&bytes).to_hex().to_string());
            let path_string = path.to_string_lossy().into_owned();

            // Fast path: blob unchanged at this path → skip decode and all ingest writes.
            if let Some(asset_id) = self.corpus_blob_match(corpus_id, &path_string, &blob_id)? {
                if dino_available && !self.embedding_exists(&asset_id, &model_name)? {
                    match embedder.embed(&path) {
                        Ok(Some(embedding)) => {
                            embed_count += 1;
                            self.save_embedding(&asset_id, &embedding)?;
                            let mut projection = self
                                .projection_model(&embedding.model_name)?
                                .unwrap_or_else(|| {
                                    ProjectionModel::zero(
                                        embedding.model_name.clone(),
                                        embedding.vector.len(),
                                    )
                                });
                            if projection.dim != embedding.vector.len() {
                                projection = ProjectionModel::zero(
                                    embedding.model_name.clone(),
                                    embedding.vector.len(),
                                );
                            }
                            self.save_projection(&projection)?;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            dino_available = false;
                            warn!(
                                "disabling ONNX embedding after runtime failure on {}: {error:#}",
                                path.display(),
                            );
                        }
                    }
                }
                unchanged += 1;
                ingested += 1;
                if ingested.is_multiple_of(log_interval) {
                    info!(
                        ingested,
                        total,
                        unchanged,
                        embedded = embed_count,
                        "corpus ingest progress"
                    );
                }
                continue;
            }

            let identity = match inspect_image_bytes(&bytes) {
                Ok(identity) => identity,
                Err(error) => {
                    note_ingest_skip(&mut skipped_identity, compact_ingest_error(&error), &path);
                    continue;
                }
            };

            let tx = self
                .conn
                .transaction()
                .context("opening ingest transaction")?;
            let asset_id =
                resolve_asset_id_for_identity(&tx, &identity)?.unwrap_or_else(mint_asset_id);
            let hidden = preserved_hidden_state_tx(&tx, corpus_id, &path_string, &asset_id)?;
            upsert_asset_identity_tx(&tx, &asset_id, &identity, 0)?;
            upsert_corpus_variant_tx(&tx, corpus_id, &path_string, &asset_id, &identity, hidden)?;
            let needs_embedding = !embedding_exists_tx(&tx, &asset_id, &model_name)?;
            tx.commit().context("committing ingest transaction")?;

            if dino_available && needs_embedding {
                match embedder.embed(&path) {
                    Ok(Some(embedding)) => {
                        embed_count += 1;
                        self.save_embedding(&asset_id, &embedding)?;
                        let mut projection = self
                            .projection_model(&embedding.model_name)?
                            .unwrap_or_else(|| {
                                ProjectionModel::zero(
                                    embedding.model_name.clone(),
                                    embedding.vector.len(),
                                )
                            });
                        if projection.dim != embedding.vector.len() {
                            projection = ProjectionModel::zero(
                                embedding.model_name.clone(),
                                embedding.vector.len(),
                            );
                        }
                        self.save_projection(&projection)?;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        dino_available = false;
                        warn!(
                            "disabling ONNX embedding after runtime failure on {}: {error:#}",
                            path.display(),
                        );
                    }
                }
            }

            ingested += 1;
            if ingested.is_multiple_of(log_interval) {
                info!(
                    ingested,
                    total,
                    unchanged,
                    embedded = embed_count,
                    "corpus ingest progress"
                );
            }
        }

        for (cause, summary) in skipped_identity {
            warn!(
                skipped = summary.count,
                sample = %summary.sample_path.display(),
                error = %cause,
                "skipped images during ingest"
            );
        }

        info!(
            ingested,
            unchanged,
            embedded = embed_count,
            "corpus ingest complete"
        );
        Ok(())
    }

    /// Check whether the blob at this path is already known and unchanged.
    fn corpus_blob_match(
        &self,
        corpus_id: CorpusId,
        path: &str,
        blob_id: &BlobId,
    ) -> anyhow::Result<Option<AssetId>> {
        let matched = self
            .conn
            .query_row(
                r"
                SELECT asset_id
                FROM corpus_assets
                WHERE corpus_id = ?1 AND path = ?2 AND blob_id = ?3
                LIMIT 1
                ",
                params![corpus_id.0, path, blob_id.0],
                |row| row.get::<_, String>(0).map(AssetId),
            )
            .optional()?;
        Ok(matched)
    }

    fn embedding_exists(&self, asset_id: &AssetId, model_name: &str) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM embeddings
                WHERE asset_id = ?1 AND model_name = ?2
                LIMIT 1
                ",
                params![asset_id.0, model_name],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn create_session(&self, corpus_id: CorpusId) -> anyhow::Result<SessionRecord> {
        let started_at = now();
        self.conn.execute(
            r"
            INSERT INTO sessions (
                corpus_id,
                started_at,
                last_touched_at,
                ended_at,
                z0,
                z1,
                z2,
                frontier,
                comparisons,
                nudges,
                hearts
            )
            VALUES (?1, ?2, ?2, NULL, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
            ",
            params![corpus_id.0, started_at.unix_timestamp()],
        )?;
        let session_id = SessionId(self.conn.last_insert_rowid());
        self.session(session_id)
    }

    pub fn resume_or_create_session(
        &self,
        corpus_id: CorpusId,
        snap_window: Duration,
    ) -> anyhow::Result<SessionRecord> {
        let snap_threshold = now_ts() - snap_window.whole_seconds();
        let resumable = self
            .conn
            .query_row(
                r#"
                SELECT id
                FROM sessions
                WHERE corpus_id = ?1
                  AND COALESCE(ended_at, last_touched_at, started_at) >= ?2
                ORDER BY COALESCE(ended_at, last_touched_at, started_at) DESC, id DESC
                LIMIT 1
                "#,
                params![corpus_id.0, snap_threshold],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;

        if let Some(session_id) = resumable {
            let session_id = SessionId(session_id);
            self.touch_session(session_id)?;
            return self.session(session_id);
        }

        self.create_session(corpus_id)
    }

    pub fn session(&self, session_id: SessionId) -> anyhow::Result<SessionRecord> {
        self.conn
            .query_row(
                r#"
                SELECT id, corpus_id, z0, z1, z2, frontier, comparisons, nudges, hearts
                FROM sessions
                WHERE id = ?1
                "#,
                params![session_id.0],
                |row| {
                    Ok(SessionRecord {
                        id: SessionId(row.get(0)?),
                        corpus_id: CorpusId(row.get(1)?),
                        mood: [row.get(2)?, row.get(3)?, row.get(4)?],
                        frontier: row.get(5)?,
                        comparisons: row.get(6)?,
                        nudges: row.get(7)?,
                        hearts: row.get(8)?,
                    })
                },
            )
            .with_context(|| format!("loading session {}", session_id.0))
    }

    pub fn touch_session(&self, session_id: SessionId) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE sessions
            SET last_touched_at = ?2,
                ended_at = NULL
            WHERE id = ?1
            ",
            params![session_id.0, now_ts()],
        )?;
        Ok(())
    }

    pub fn close_session(&self, session_id: SessionId) -> anyhow::Result<()> {
        let closed_at = now_ts();
        self.conn.execute(
            r"
            UPDATE sessions
            SET last_touched_at = ?2,
                ended_at = ?2
            WHERE id = ?1
            ",
            params![session_id.0, closed_at],
        )?;
        Ok(())
    }

    pub fn recent_arena_asset_ids(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> anyhow::Result<Vec<AssetId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_id
            FROM (
                SELECT
                    created_at,
                    id,
                    0 AS source_ord,
                    0 AS slot_ord,
                    left_asset_id AS asset_id
                FROM comparisons
                WHERE session_id = ?1
                UNION ALL
                SELECT
                    created_at,
                    id,
                    0 AS source_ord,
                    1 AS slot_ord,
                    right_asset_id AS asset_id
                FROM comparisons
                WHERE session_id = ?1
                UNION ALL
                SELECT
                    created_at,
                    id,
                    1 AS source_ord,
                    0 AS slot_ord,
                    local_asset_id AS asset_id
                FROM external_events
                WHERE session_id = ?1
                  AND event_kind = ?2
                  AND local_asset_id IS NOT NULL
            )
            ORDER BY created_at DESC, source_ord ASC, id DESC, slot_ord ASC
            LIMIT ?3
            ",
        )?;
        let mut rows = stmt.query(params![
            session_id.0,
            ExternalEventKind::Selected.as_str(),
            i64::try_from(limit)?,
        ])?;
        let mut ids = Vec::with_capacity(limit);
        while let Some(row) = rows.next()? {
            ids.push(AssetId(row.get(0)?));
        }
        Ok(ids)
    }

    pub fn corpus_assets(&self, corpus_id: CorpusId) -> anyhow::Result<Vec<AssetRecord>> {
        let mut assets = choose_corpus_asset_rows(self.load_corpus_asset_rows(corpus_id, None)?);
        assets.sort_by(|left, right| {
            right
                .alpha
                .total_cmp(&left.alpha)
                .then_with(|| right.win_count.cmp(&left.win_count))
                .then_with(|| right.compare_count.cmp(&left.compare_count))
                .then_with(|| left.id.0.cmp(&right.id.0))
        });
        Ok(assets)
    }

    pub fn corpus_asset(
        &self,
        corpus_id: CorpusId,
        asset_id: &AssetId,
    ) -> anyhow::Result<Option<AssetRecord>> {
        Ok(
            choose_corpus_asset_rows(self.load_corpus_asset_rows(corpus_id, Some(asset_id))?)
                .into_iter()
                .next(),
        )
    }

    fn load_corpus_asset_rows(
        &self,
        corpus_id: CorpusId,
        asset_id: Option<&AssetId>,
    ) -> anyhow::Result<Vec<CorpusAssetRow>> {
        let sql = if asset_id.is_some() {
            r"
            SELECT a.id,
                   ca.path,
                   a.alpha,
                   a.c0,
                   a.c1,
                   a.c2,
                   a.rotation_quarters,
                   a.compare_count,
                   a.win_count,
                   a.heart_count,
                   a.hearted,
                   ca.hidden,
                   a.preferred_blob_id,
                   ca.blob_id,
                   ca.blob_width,
                   ca.blob_height
            FROM corpus_assets ca
            JOIN assets a ON a.id = ca.asset_id
            WHERE ca.corpus_id = ?1
              AND ca.asset_id = ?2
            "
        } else {
            r"
            SELECT a.id,
                   ca.path,
                   a.alpha,
                   a.c0,
                   a.c1,
                   a.c2,
                   a.rotation_quarters,
                   a.compare_count,
                   a.win_count,
                   a.heart_count,
                   a.hearted,
                   ca.hidden,
                   a.preferred_blob_id,
                   ca.blob_id,
                   ca.blob_width,
                   ca.blob_height
            FROM corpus_assets ca
            JOIN assets a ON a.id = ca.asset_id
            WHERE ca.corpus_id = ?1
            "
        };
        let mut stmt = self.conn.prepare(sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            Ok(CorpusAssetRow {
                asset: AssetRecord {
                    id: AssetId(row.get(0)?),
                    path: PathBuf::from(row.get::<_, String>(1)?),
                    width: u32::try_from(row.get::<_, i64>(14)?).unwrap_or_default(),
                    height: u32::try_from(row.get::<_, i64>(15)?).unwrap_or_default(),
                    alpha: row.get(2)?,
                    coords: [row.get(3)?, row.get(4)?, row.get(5)?],
                    rotation_quarters: row.get(6)?,
                    compare_count: u32::try_from(row.get::<_, i64>(7)?).unwrap_or_default(),
                    win_count: u32::try_from(row.get::<_, i64>(8)?).unwrap_or_default(),
                    heart_count: u32::try_from(row.get::<_, i64>(9)?).unwrap_or_default(),
                    is_hearted: row.get::<_, i64>(10)? != 0,
                    hidden: row.get::<_, i64>(11)? != 0,
                },
                preferred_blob_id: row.get::<_, Option<String>>(12)?.map(BlobId),
                variant_blob_id: row.get::<_, Option<String>>(13)?.map(BlobId),
                blob_width: u32::try_from(row.get::<_, i64>(14)?).unwrap_or_default(),
                blob_height: u32::try_from(row.get::<_, i64>(15)?).unwrap_or_default(),
            })
        };
        let rows = match asset_id {
            Some(asset_id) => stmt.query_map(params![corpus_id.0, asset_id.0], map_row)?,
            None => stmt.query_map(params![corpus_id.0], map_row)?,
        };
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn session_asset_offsets(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<HashMap<AssetId, f32>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_id, offset
            FROM session_asset_offsets
            WHERE session_id = ?1
            ",
        )?;
        let rows = stmt.query_map(params![session_id.0], |row| {
            Ok((AssetId(row.get(0)?), row.get::<_, f32>(1)?))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn session_hearted_assets(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<HashSet<AssetId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_id
            FROM session_asset_hearts
            WHERE session_id = ?1
            ",
        )?;
        let rows = stmt.query_map(params![session_id.0], |row| Ok(AssetId(row.get(0)?)))?;
        rows.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
    }

    pub fn session_subsource_lock(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<Option<crate::model::SessionSubsourceLock>> {
        self.conn
            .query_row(
                r"
                SELECT source_key, stream_id
                FROM session_subsource_locks
                WHERE session_id = ?1
                ",
                params![session_id.0],
                |row| {
                    Ok(crate::model::SessionSubsourceLock {
                        source_key: row.get(0)?,
                        stream_id: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn set_session_subsource_lock(
        &self,
        session_id: SessionId,
        source_key: &str,
        stream_id: i64,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO session_subsource_locks (session_id, source_key, stream_id, updated_at)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(session_id) DO UPDATE SET
                source_key = excluded.source_key,
                stream_id = excluded.stream_id,
                updated_at = excluded.updated_at
            ",
            params![session_id.0, source_key, stream_id, now_ts()],
        )?;
        Ok(())
    }

    pub fn clear_session_subsource_lock(&self, session_id: SessionId) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            DELETE FROM session_subsource_locks
            WHERE session_id = ?1
            ",
            params![session_id.0],
        )?;
        Ok(())
    }

    pub fn hearted_assets(&self) -> anyhow::Result<HashSet<AssetId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id
            FROM assets
            WHERE hearted != 0
            ",
        )?;
        let rows = stmt.query_map([], |row| Ok(AssetId(row.get(0)?)))?;
        rows.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
    }

    pub fn corpus_embeddings(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<HashMap<AssetId, Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT DISTINCT e.asset_id, e.vector
            FROM embeddings e
            JOIN corpus_assets ca ON ca.asset_id = e.asset_id
            WHERE ca.corpus_id = ?1 AND e.model_name = ?2
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0, model_name], |row| {
            Ok((
                AssetId(row.get(0)?),
                decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn embeddings_for_assets(
        &self,
        asset_ids: &[AssetId],
        model_name: &str,
    ) -> anyhow::Result<HashMap<AssetId, Vec<f32>>> {
        if asset_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = (0..asset_ids.len())
            .map(|_| "?".to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r"
            SELECT asset_id, vector
            FROM embeddings
            WHERE model_name = ?1
              AND asset_id IN ({placeholders})
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params = rusqlite::params_from_iter(
            std::iter::once(model_name).chain(asset_ids.iter().map(|asset_id| asset_id.0.as_str())),
        );
        let rows = stmt.query_map(params, |row| {
            Ok((
                AssetId(row.get(0)?),
                decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn session_embedding_head(
        &self,
        session_id: SessionId,
        model_name: &str,
    ) -> anyhow::Result<Option<SessionEmbeddingHead>> {
        self.conn
            .query_row(
                r"
                SELECT dim, weights
                FROM session_embedding_heads
                WHERE session_id = ?1 AND model_name = ?2
                ",
                params![session_id.0, model_name],
                |row| {
                    Ok(SessionEmbeddingHead {
                        model_name: model_name.to_owned(),
                        dim: usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        weights: decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn embedding(
        &self,
        asset_id: &AssetId,
        model_name: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        self.conn
            .query_row(
                r"
                SELECT vector
                FROM embeddings
                WHERE asset_id = ?1 AND model_name = ?2
                ",
                params![asset_id.0, model_name],
                |row| {
                    let bytes: Vec<u8> = row.get(0)?;
                    Ok(decode_vec_f32(&bytes))
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_embedding(
        &self,
        asset_id: &AssetId,
        embedding: &EmbeddingRecord,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO embeddings (asset_id, model_name, dim, vector)
            VALUES (?1, ?2, ?3, ?4)
            ",
            params![
                asset_id.0,
                embedding.model_name,
                i64::try_from(embedding.vector.len())?,
                encode_vec_f32(&embedding.vector),
            ],
        )?;
        Ok(())
    }

    pub fn asset_domain_labels(
        &self,
        asset_ids: &[AssetId],
    ) -> anyhow::Result<HashMap<AssetId, AssetDomainLabel>> {
        if asset_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = (0..asset_ids.len())
            .map(|_| "?".to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r"
            SELECT asset_id, label
            FROM asset_domain_labels
            WHERE asset_id IN ({placeholders})
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params =
            rusqlite::params_from_iter(asset_ids.iter().map(|asset_id| asset_id.0.as_str()));
        let rows = stmt.query_map(params, |row| {
            let raw: String = row.get(1)?;
            let label = raw.parse::<AssetDomainLabel>().map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
                )
            })?;
            Ok((AssetId(row.get(0)?), label))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn set_asset_domain_label(
        &self,
        asset_id: &AssetId,
        label: AssetDomainLabel,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO asset_domain_labels (asset_id, label, updated_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(asset_id) DO UPDATE
            SET label = excluded.label,
                updated_at = excluded.updated_at
            ",
            params![asset_id.0, label.as_str(), now_ts()],
        )?;
        Ok(())
    }

    pub fn asset_domain_label_counts(&self) -> anyhow::Result<(usize, usize)> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT label, COUNT(*)
            FROM asset_domain_labels
            GROUP BY label
            ",
        )?;
        let mut real = 0usize;
        let mut anime = 0usize;
        for row in stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })? {
            let (label, count) = row?;
            let count = usize::try_from(count).unwrap_or(0);
            match label.parse::<AssetDomainLabel>() {
                Ok(AssetDomainLabel::Real) => real = count,
                Ok(AssetDomainLabel::Anime) => anime = count,
                Err(_) => {}
            }
        }
        Ok((real, anime))
    }

    pub fn asset_domain_training_rows(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<Vec<AssetDomainTrainingRow>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT ad.asset_id, ad.label, e.vector
            FROM asset_domain_labels ad
            JOIN corpus_assets ca
              ON ca.asset_id = ad.asset_id
             AND ca.corpus_id = ?1
            JOIN embeddings e
              ON e.asset_id = ad.asset_id
             AND e.model_name = ?2
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0, model_name], |row| {
            let raw_label: String = row.get(1)?;
            let label = raw_label.parse::<AssetDomainLabel>().map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
                )
            })?;
            Ok(AssetDomainTrainingRow {
                asset_id: AssetId(row.get(0)?),
                label,
                embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(2)?),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn ingest_external_import(
        &mut self,
        corpus_id: CorpusId,
        import_path: &Path,
        bytes: &[u8],
        rotation_quarters: i32,
        embedding: Option<&EmbeddingRecord>,
    ) -> anyhow::Result<AssetId> {
        let identity = inspect_image_bytes(bytes).context("inspecting external import identity")?;
        self.ingest_external_import_precomputed(
            corpus_id,
            import_path,
            &identity,
            rotation_quarters,
            embedding,
        )
    }

    pub fn ingest_external_import_precomputed(
        &mut self,
        corpus_id: CorpusId,
        import_path: &Path,
        identity: &ImageIdentity,
        rotation_quarters: i32,
        embedding: Option<&EmbeddingRecord>,
    ) -> anyhow::Result<AssetId> {
        let path_string = import_path.to_string_lossy().into_owned();
        let tx = self
            .conn
            .transaction()
            .context("opening external import transaction")?;
        let asset_id = resolve_asset_id_for_identity(&tx, identity)?.unwrap_or_else(mint_asset_id);
        let hidden = preserved_hidden_state_tx(&tx, corpus_id, &path_string, &asset_id)?;
        upsert_asset_identity_tx(&tx, &asset_id, identity, rotation_quarters.rem_euclid(4))?;
        upsert_corpus_variant_tx(&tx, corpus_id, &path_string, &asset_id, identity, hidden)?;
        if let Some(embedding) = embedding {
            upsert_embedding_tx(&tx, &asset_id, embedding)?;
        }
        tx.commit()
            .context("committing external import transaction")?;
        Ok(asset_id)
    }

    pub fn projection_model(&self, model_name: &str) -> anyhow::Result<Option<ProjectionModel>> {
        self.conn
            .query_row(
                r"
                SELECT dim, bias0, bias1, bias2, weights
                FROM projection_models
                WHERE model_name = ?1
                ",
                params![model_name],
                |row| {
                    Ok(ProjectionModel {
                        model_name: model_name.to_owned(),
                        dim: usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        bias: [row.get(1)?, row.get(2)?, row.get(3)?],
                        weights: decode_vec_f32(&row.get::<_, Vec<u8>>(4)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_projection(&self, model: &ProjectionModel) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO projection_models (
                model_name,
                dim,
                bias0,
                bias1,
                bias2,
                weights,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                model.model_name,
                i64::try_from(model.dim)?,
                model.bias[0],
                model.bias[1],
                model.bias[2],
                encode_vec_f32(&model.weights),
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn similarity_model(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<Option<SimilarityModel>> {
        let row = self
            .conn
            .query_row(
                r"
                SELECT dim, mean, weights, kind, payload
                FROM similarity_models
                WHERE corpus_id = ?1 AND model_name = ?2
                ",
                params![corpus_id.0, model_name],
                |row| {
                    Ok((
                        usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                    ))
                },
            )
            .optional()
            .map_err(anyhow::Error::from)?;
        let Some((dim, mean, weights, payload)) = row else {
            return Ok(None);
        };
        if let Some(payload) = payload {
            return serde_json::from_slice::<SimilarityModel>(&payload)
                .map(Some)
                .map_err(Into::into);
        }
        Ok(Some(SimilarityModel::Linear(
            crate::model::LinearSimilarityModel {
                model_name: model_name.to_owned(),
                dim,
                mean: decode_vec_f32(&mean),
                weights: decode_vec_f32(&weights),
            },
        )))
    }

    pub fn save_similarity_model(
        &self,
        corpus_id: CorpusId,
        model: &SimilarityModel,
    ) -> anyhow::Result<()> {
        let (dim, mean, weights) = match model {
            SimilarityModel::Linear(model) => (model.dim, &model.mean, &model.weights),
            SimilarityModel::Ordinal(model) => {
                (model.prior.dim, &model.prior.mean, &model.prior.weights)
            }
        };
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO similarity_models (
                corpus_id,
                model_name,
                dim,
                mean,
                weights,
                kind,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                corpus_id.0,
                model.model_name(),
                i64::try_from(dim)?,
                encode_vec_f32(mean),
                encode_vec_f32(weights),
                match model {
                    SimilarityModel::Linear(_) => "linear",
                    SimilarityModel::Ordinal(_) => "ordinal",
                },
                serde_json::to_vec(model)?,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn similarity_history(
        &self,
        corpus_id: CorpusId,
    ) -> anyhow::Result<Vec<SimilarityObservation>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id, chosen_pair
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id ASC
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0], |row| {
            Ok((
                AssetId(row.get(0)?),
                AssetId(row.get(1)?),
                AssetId(row.get(2)?),
                row.get::<_, String>(3)?,
            ))
        })?;
        let raw = rows.collect::<Result<Vec<_>, _>>()?;
        raw.into_iter()
            .map(|(asset_a, asset_b, asset_c, choice)| {
                Ok(SimilarityObservation {
                    asset_a,
                    asset_b,
                    asset_c,
                    choice: choice.parse().map_err(anyhow::Error::msg)?,
                })
            })
            .collect()
    }

    pub fn recent_similarity_asset_ids(
        &self,
        corpus_id: CorpusId,
        limit: usize,
    ) -> anyhow::Result<Vec<AssetId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id DESC
            LIMIT ?2
            ",
        )?;
        let mut rows = stmt.query(params![corpus_id.0, i64::try_from(limit)?])?;
        let mut ids = Vec::with_capacity(limit * 3);
        while let Some(row) = rows.next()? {
            ids.push(AssetId(row.get(0)?));
            ids.push(AssetId(row.get(1)?));
            ids.push(AssetId(row.get(2)?));
        }
        Ok(ids)
    }

    pub fn recent_similarity_triads(
        &self,
        corpus_id: CorpusId,
        limit: usize,
    ) -> anyhow::Result<Vec<[AssetId; 3]>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id DESC
            LIMIT ?2
            ",
        )?;
        let mut rows = stmt.query(params![corpus_id.0, i64::try_from(limit)?])?;
        let mut triads = Vec::with_capacity(limit);
        while let Some(row) = rows.next()? {
            triads.push([
                AssetId(row.get(0)?),
                AssetId(row.get(1)?),
                AssetId(row.get(2)?),
            ]);
        }
        Ok(triads)
    }

    pub fn rotate_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE assets
            SET rotation_quarters = ((rotation_quarters + ?2) % 4 + 4) % 4
            WHERE id = ?1
            ",
            params![asset_id.0, direction],
        )?;
        Ok(())
    }

    pub fn set_hidden(
        &self,
        corpus_id: CorpusId,
        asset_id: &AssetId,
        hidden: bool,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE corpus_assets
            SET hidden = ?3
            WHERE corpus_id = ?1 AND asset_id = ?2
            ",
            params![corpus_id.0, asset_id.0, i64::from(hidden)],
        )?;
        Ok(())
    }

    pub fn persist_duel_step(
        &mut self,
        session: &SessionRecord,
        left_before: &AssetRecord,
        right_before: &AssetRecord,
        left_after: &AssetRecord,
        right_after: &AssetRecord,
        winner: &AssetId,
        left_utility: f32,
        right_utility: f32,
        projection: Option<&ProjectionModel>,
        embedding_head: Option<&SessionEmbeddingHead>,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening comparison transaction")?;

        update_asset(&tx, left_after)?;
        update_asset(&tx, right_after)?;
        update_session(&tx, session)?;
        tx.execute(
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
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                left_before.id.0,
                right_before.id.0,
                winner.0,
                now_ts(),
                left_utility,
                right_utility,
            ],
        )?;
        upsert_projection(&tx, projection)?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit().context("committing comparison transaction")?;
        self.mirror_legacy_quality_cache(session, [left_after, right_after]);
        Ok(())
    }

    pub fn persist_hierarchical_duel_step(
        &mut self,
        session: &SessionRecord,
        left_before: &AssetRecord,
        right_before: &AssetRecord,
        left_after: &AssetRecord,
        right_after: &AssetRecord,
        winner: &AssetId,
        left_utility: f32,
        right_utility: f32,
        embedding_head: Option<&SessionEmbeddingHead>,
        left_cache: &HierarchicalAssetQualityCacheV1,
        right_cache: &HierarchicalAssetQualityCacheV1,
        session_cache: &HierarchicalSessionQualityCacheV1,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening hierarchical comparison transaction")?;
        update_asset(&tx, left_after)?;
        update_asset(&tx, right_after)?;
        update_session(&tx, session)?;
        tx.execute(
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
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                left_before.id.0,
                right_before.id.0,
                winner.0,
                now_ts(),
                left_utility,
                right_utility,
            ],
        )?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit()
            .context("committing hierarchical comparison transaction")?;
        let updated_at = now();
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: left_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                left_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: right_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                right_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: crate::quality::SessionQualityCachePayload::HierarchicalGaussianV1(
                session_cache.clone(),
            ),
            updated_at,
        })?;
        Ok(())
    }

    pub fn persist_nudge_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        direction: f32,
        utility: f32,
        frontier: f32,
        signal: f32,
        exact_offset: f32,
        projection: Option<&ProjectionModel>,
        embedding_head: Option<&SessionEmbeddingHead>,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening nudge transaction")?;

        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        write_session_offset(&tx, session.id, asset_id, exact_offset)?;
        tx.execute(
            r"
            INSERT INTO nudge_events (
                session_id,
                corpus_id,
                asset_id,
                direction,
                created_at,
                utility,
                frontier,
                signal
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                asset_id.0,
                direction,
                now_ts(),
                utility,
                frontier,
                signal,
            ],
        )?;
        upsert_projection(&tx, projection)?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit().context("committing nudge transaction")?;
        self.mirror_legacy_quality_cache(session, [asset_after]);
        Ok(())
    }

    pub fn persist_hierarchical_nudge_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        direction: f32,
        utility: f32,
        frontier: f32,
        signal: f32,
        exact_offset: f32,
        embedding_head: Option<&SessionEmbeddingHead>,
        asset_cache: &HierarchicalAssetQualityCacheV1,
        session_cache: &HierarchicalSessionQualityCacheV1,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening hierarchical nudge transaction")?;
        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        write_session_offset(&tx, session.id, asset_id, exact_offset)?;
        tx.execute(
            r"
            INSERT INTO nudge_events (
                session_id,
                corpus_id,
                asset_id,
                direction,
                created_at,
                utility,
                frontier,
                signal
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                asset_id.0,
                direction,
                now_ts(),
                utility,
                frontier,
                signal,
            ],
        )?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit()
            .context("committing hierarchical nudge transaction")?;
        let updated_at = now();
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: asset_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                asset_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: crate::quality::SessionQualityCachePayload::HierarchicalGaussianV1(
                session_cache.clone(),
            ),
            updated_at,
        })?;
        Ok(())
    }

    pub fn persist_heart_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        active: bool,
    ) -> anyhow::Result<()> {
        if !active {
            return Ok(());
        }
        let has_utility = self.has_column("heart_events", "utility")?;
        let has_signal = self.has_column("heart_events", "signal")?;
        let tx = self
            .conn
            .transaction()
            .context("opening heart transaction")?;

        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        enshrine_session_heart(&tx, session.id, asset_id)?;
        let legacy_signal = 1.0;
        match (has_utility, has_signal) {
            (true, true) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    utility,
                    signal,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (true, false) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    utility,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (false, true) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    signal,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (false, false) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    i64::from(active),
                ],
            )?,
        };
        tx.commit().context("committing heart transaction")?;
        self.mirror_legacy_quality_cache(session, [asset_after]);
        Ok(())
    }

    fn mirror_legacy_quality_cache<const N: usize>(
        &self,
        session: &SessionRecord,
        assets: [&AssetRecord; N],
    ) {
        let Ok(model) = self.active_quality_model() else {
            return;
        };
        if model.formal_version != QualityFormalVersion::LegacyIndependentV1 {
            return;
        }

        let updated_at = now();
        for asset in assets {
            if let Err(error) = self.save_asset_quality_cache(&StoredAssetQualityCache {
                asset_id: asset.id.clone(),
                payload: legacy_asset_quality_payload(asset),
                updated_at,
            }) {
                warn!(
                    error = %format!("{error:#}"),
                    asset_id = %asset.id.0,
                    "failed to mirror legacy asset quality cache"
                );
            }
        }
        if let Err(error) = self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: legacy_session_quality_payload(session),
            updated_at,
        }) {
            warn!(
                error = %format!("{error:#}"),
                session_id = session.id.0,
                "failed to mirror legacy session quality cache"
            );
        }
    }

    pub fn persist_similarity_step(
        &mut self,
        corpus_id: CorpusId,
        model: &SimilarityModel,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        choice: SimilarityChoice,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening similarity transaction")?;
        let (dim, mean, weights) = match model {
            SimilarityModel::Linear(model) => (model.dim, &model.mean, &model.weights),
            SimilarityModel::Ordinal(model) => {
                (model.prior.dim, &model.prior.mean, &model.prior.weights)
            }
        };

        tx.execute(
            r"
            INSERT OR REPLACE INTO similarity_models (
                corpus_id,
                model_name,
                dim,
                mean,
                weights,
                kind,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                corpus_id.0,
                model.model_name(),
                i64::try_from(dim)?,
                encode_vec_f32(mean),
                encode_vec_f32(weights),
                match model {
                    SimilarityModel::Linear(_) => "linear",
                    SimilarityModel::Ordinal(_) => "ordinal",
                },
                serde_json::to_vec(model)?,
                now_ts(),
            ],
        )?;
        tx.execute(
            r"
            INSERT INTO similarity_triads (
                corpus_id,
                model_name,
                asset_a_id,
                asset_b_id,
                asset_c_id,
                chosen_pair,
                created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                corpus_id.0,
                model.model_name(),
                asset_a.0,
                asset_b.0,
                asset_c.0,
                choice.as_str(),
                now_ts(),
            ],
        )?;
        tx.commit().context("committing similarity transaction")?;
        Ok(())
    }
}

fn resolve_asset_id_for_identity(
    tx: &Transaction<'_>,
    identity: &ImageIdentity,
) -> anyhow::Result<Option<AssetId>> {
    if let Some(asset_id) = tx
        .query_row(
            r"
            SELECT asset_id
            FROM corpus_assets
            WHERE blob_id = ?1
            ORDER BY blob_width * blob_height DESC, last_seen_at DESC, path ASC
            LIMIT 1
            ",
            params![identity.blob_id.0],
            |row| row.get::<_, String>(0),
        )
        .optional()?
    {
        return Ok(Some(AssetId(asset_id)));
    }

    tx.query_row(
        r"
        SELECT id
        FROM assets
        WHERE visual_key = ?1
        ORDER BY pixel_width * pixel_height DESC, created_at ASC, id ASC
        LIMIT 1
        ",
        params![identity.visual_key.0],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map(|asset_id| asset_id.map(AssetId))
    .map_err(Into::into)
}

fn preserved_hidden_state_tx(
    tx: &Transaction<'_>,
    corpus_id: CorpusId,
    path: &str,
    asset_id: &AssetId,
) -> anyhow::Result<bool> {
    if let Some(hidden) = tx
        .query_row(
            r"
            SELECT hidden
            FROM corpus_assets
            WHERE corpus_id = ?1 AND path = ?2
            LIMIT 1
            ",
            params![corpus_id.0, path],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
    {
        return Ok(hidden != 0);
    }

    let hidden = tx
        .query_row(
            r"
            SELECT MIN(hidden)
            FROM corpus_assets
            WHERE corpus_id = ?1 AND asset_id = ?2
            ",
            params![corpus_id.0, asset_id.0],
            |row| row.get::<_, Option<i64>>(0),
        )
        .optional()?
        .flatten()
        .unwrap_or(0);
    Ok(hidden != 0)
}

fn upsert_asset_identity_tx(
    tx: &Transaction<'_>,
    asset_id: &AssetId,
    identity: &ImageIdentity,
    rotation_quarters: i32,
) -> anyhow::Result<()> {
    tx.execute(
        r"
        INSERT INTO assets (
            id,
            created_at,
            preferred_blob_id,
            visual_key,
            pixel_width,
            pixel_height,
            rotation_quarters,
            alpha,
            c0,
            c1,
            c2,
            heart_count,
            hearted,
            compare_count,
            win_count
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
        ON CONFLICT(id) DO UPDATE SET
            visual_key = COALESCE(NULLIF(assets.visual_key, ''), excluded.visual_key),
            preferred_blob_id = CASE
                WHEN excluded.pixel_width * excluded.pixel_height
                    > assets.pixel_width * assets.pixel_height
                    THEN excluded.preferred_blob_id
                ELSE COALESCE(assets.preferred_blob_id, excluded.preferred_blob_id)
            END,
            pixel_width = CASE
                WHEN excluded.pixel_width * excluded.pixel_height
                    > assets.pixel_width * assets.pixel_height
                    THEN excluded.pixel_width
                ELSE assets.pixel_width
            END,
            pixel_height = CASE
                WHEN excluded.pixel_width * excluded.pixel_height
                    > assets.pixel_width * assets.pixel_height
                    THEN excluded.pixel_height
                ELSE assets.pixel_height
            END
        ",
        params![
            asset_id.0,
            now_ts(),
            identity.blob_id.0,
            identity.visual_key.0,
            i64::from(identity.width),
            i64::from(identity.height),
            rotation_quarters,
        ],
    )?;
    Ok(())
}

fn upsert_corpus_variant_tx(
    tx: &Transaction<'_>,
    corpus_id: CorpusId,
    path: &str,
    asset_id: &AssetId,
    identity: &ImageIdentity,
    hidden: bool,
) -> anyhow::Result<()> {
    tx.execute(
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
        ON CONFLICT(corpus_id, path) DO UPDATE SET
            asset_id = excluded.asset_id,
            blob_id = excluded.blob_id,
            blob_width = excluded.blob_width,
            blob_height = excluded.blob_height,
            blob_bytes = excluded.blob_bytes,
            hidden = excluded.hidden,
            last_seen_at = excluded.last_seen_at
        ",
        params![
            corpus_id.0,
            path,
            asset_id.0,
            identity.blob_id.0,
            i64::from(identity.width),
            i64::from(identity.height),
            i64::try_from(identity.byte_len).unwrap_or_default(),
            i64::from(hidden),
            now_ts(),
        ],
    )?;
    Ok(())
}

fn embedding_exists_tx(
    tx: &Transaction<'_>,
    asset_id: &AssetId,
    model_name: &str,
) -> anyhow::Result<bool> {
    tx.query_row(
        r"
        SELECT 1
        FROM embeddings
        WHERE asset_id = ?1 AND model_name = ?2
        LIMIT 1
        ",
        params![asset_id.0, model_name],
        |_row| Ok(()),
    )
    .optional()
    .map(|present| present.is_some())
    .map_err(Into::into)
}

fn upsert_embedding_tx(
    tx: &Transaction<'_>,
    asset_id: &AssetId,
    embedding: &EmbeddingRecord,
) -> anyhow::Result<()> {
    tx.execute(
        r"
        INSERT OR REPLACE INTO embeddings (asset_id, model_name, dim, vector)
        VALUES (?1, ?2, ?3, ?4)
        ",
        params![
            asset_id.0,
            embedding.model_name,
            i64::try_from(embedding.vector.len())?,
            encode_vec_f32(&embedding.vector),
        ],
    )?;
    Ok(())
}

fn choose_corpus_asset_rows(rows: Vec<CorpusAssetRow>) -> Vec<AssetRecord> {
    let mut best_by_asset: HashMap<AssetId, CorpusAssetRow> = HashMap::new();
    let mut visible_by_asset: HashMap<AssetId, bool> = HashMap::new();

    for row in rows {
        let asset_id = row.asset.id.clone();
        let visible = !row.asset.hidden;
        visible_by_asset
            .entry(asset_id.clone())
            .and_modify(|current| *current |= visible)
            .or_insert(visible);
        match best_by_asset.get(&asset_id) {
            Some(current) if !corpus_variant_beats(&row, current) => {}
            _ => {
                best_by_asset.insert(asset_id, row);
            }
        }
    }

    best_by_asset
        .into_iter()
        .map(|(asset_id, row)| {
            let mut asset = row.asset;
            let any_visible = visible_by_asset.get(&asset_id).copied().unwrap_or_default();
            asset.hidden = !any_visible;
            asset
        })
        .collect()
}

fn corpus_variant_beats(left: &CorpusAssetRow, right: &CorpusAssetRow) -> bool {
    let left_pref = variant_matches_preferred(left);
    let right_pref = variant_matches_preferred(right);
    if left_pref != right_pref {
        return left_pref;
    }

    let left_area = u64::from(left.blob_width) * u64::from(left.blob_height);
    let right_area = u64::from(right.blob_width) * u64::from(right.blob_height);
    if left_area != right_area {
        return left_area > right_area;
    }

    left.asset.path < right.asset.path
}

fn variant_matches_preferred(row: &CorpusAssetRow) -> bool {
    row.preferred_blob_id.is_some() && row.preferred_blob_id == row.variant_blob_id
}

fn update_asset(tx: &Transaction<'_>, asset: &AssetRecord) -> anyhow::Result<()> {
    tx.execute(
        r"
        UPDATE assets
        SET alpha = ?2,
            c0 = ?3,
            c1 = ?4,
            c2 = ?5,
            rotation_quarters = ?6,
            heart_count = ?7,
            hearted = ?8,
            compare_count = ?9,
            win_count = ?10
        WHERE id = ?1
        ",
        params![
            asset.id.0,
            asset.alpha,
            asset.coords[0],
            asset.coords[1],
            asset.coords[2],
            asset.rotation_quarters,
            asset.heart_count,
            i64::from(asset.is_hearted),
            asset.compare_count,
            asset.win_count,
        ],
    )?;
    Ok(())
}

fn update_session(tx: &Transaction<'_>, session: &SessionRecord) -> anyhow::Result<()> {
    tx.execute(
        r"
        UPDATE sessions
        SET z0 = ?2,
            z1 = ?3,
            z2 = ?4,
            frontier = ?5,
            comparisons = ?6,
            nudges = ?7,
            hearts = ?8,
            last_touched_at = ?9,
            ended_at = NULL
        WHERE id = ?1
        ",
        params![
            session.id.0,
            session.mood[0],
            session.mood[1],
            session.mood[2],
            session.frontier,
            session.comparisons,
            session.nudges,
            session.hearts,
            now_ts(),
        ],
    )?;
    Ok(())
}

fn write_session_offset(
    tx: &Transaction<'_>,
    session_id: SessionId,
    asset_id: &AssetId,
    offset: f32,
) -> anyhow::Result<()> {
    if offset.abs() < 0.0005 {
        tx.execute(
            r"
            DELETE FROM session_asset_offsets
            WHERE session_id = ?1 AND asset_id = ?2
            ",
            params![session_id.0, asset_id.0],
        )?;
    } else {
        tx.execute(
            r"
            INSERT OR REPLACE INTO session_asset_offsets (
                session_id,
                asset_id,
                offset,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
            params![session_id.0, asset_id.0, offset, now_ts()],
        )?;
    }
    Ok(())
}

fn enshrine_session_heart(
    tx: &Transaction<'_>,
    session_id: SessionId,
    asset_id: &AssetId,
) -> anyhow::Result<()> {
    tx.execute(
        r"
        INSERT OR REPLACE INTO session_asset_hearts (session_id, asset_id, updated_at)
        VALUES (?1, ?2, ?3)
        ",
        params![session_id.0, asset_id.0, now_ts()],
    )?;
    Ok(())
}

fn upsert_projection(
    tx: &Transaction<'_>,
    projection: Option<&ProjectionModel>,
) -> anyhow::Result<()> {
    let Some(model) = projection else {
        return Ok(());
    };

    tx.execute(
        r"
        INSERT OR REPLACE INTO projection_models (
            model_name,
            dim,
            bias0,
            bias1,
            bias2,
            weights,
            updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ",
        params![
            model.model_name,
            i64::try_from(model.dim)?,
            model.bias[0],
            model.bias[1],
            model.bias[2],
            encode_vec_f32(&model.weights),
            now_ts(),
        ],
    )?;
    Ok(())
}

fn upsert_session_embedding_head(
    tx: &Transaction<'_>,
    session_id: SessionId,
    embedding_head: Option<&SessionEmbeddingHead>,
) -> anyhow::Result<()> {
    let Some(head) = embedding_head else {
        return Ok(());
    };

    tx.execute(
        r"
        INSERT OR REPLACE INTO session_embedding_heads (
            session_id,
            model_name,
            dim,
            weights,
            updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5)
        ",
        params![
            session_id.0,
            head.model_name,
            i64::try_from(head.dim)?,
            encode_vec_f32(&head.weights),
            now_ts(),
        ],
    )?;
    Ok(())
}

fn encode_vec_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_vec_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase),
        Some(ext)
            if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "jxl")
    )
}

fn now() -> OffsetDateTime {
    OffsetDateTime::now_utc()
}

fn now_ts() -> i64 {
    now().unix_timestamp()
}

fn note_ingest_skip(skips: &mut BTreeMap<String, IngestSkipSummary>, cause: String, path: &Path) {
    skips
        .entry(cause)
        .and_modify(|summary| summary.count += 1)
        .or_insert_with(|| IngestSkipSummary {
            count: 1,
            sample_path: path.to_path_buf(),
        });
}

fn compact_ingest_error(error: &anyhow::Error) -> String {
    error.root_cause().to_string()
}
