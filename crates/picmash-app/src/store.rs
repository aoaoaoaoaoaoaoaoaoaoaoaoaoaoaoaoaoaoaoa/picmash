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

mod corpus;
mod events;
mod external;
mod external_frontier;
mod faces;
mod maintenance;
mod quality;
mod schema;
mod session;
#[cfg(test)]
mod tests;
mod vectors;

pub use external::ExternalStreamWarmCandidate;
pub use external_frontier::{
    ExternalIdentityDisposition, ExternalReadyCachePath, ExternalStreamFrontierCounts,
    UpsertedExternalStreamBatchEntry,
};
pub use faces::{FaceIdentityRecord, FaceRecord, FacemashIdentityCandidate};
pub use schema::{
    BOOTSTRAP_PHASE_FACE_IDENTITY_BINDINGS, BOOTSTRAP_PHASE_INITIAL, BootstrapMaintenanceProgress,
};

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

    pub fn devour_bootstrap_maintenance_batch(
        &self,
        phase: &str,
        limit: usize,
    ) -> anyhow::Result<BootstrapMaintenanceProgress> {
        self.run_bootstrap_maintenance_batch(phase, limit)
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

    if let Some(asset_id) = tx
        .query_row(
            r"
            SELECT id
            FROM assets
            WHERE render_hash = ?1
            ORDER BY pixel_width * pixel_height DESC, created_at ASC, id ASC
            LIMIT 1
            ",
            params![identity.render_hash.0],
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
            render_hash,
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
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
        ON CONFLICT(id) DO UPDATE SET
            render_hash = COALESCE(NULLIF(assets.render_hash, ''), excluded.render_hash),
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
            identity.render_hash.0,
            identity.visual_key.0,
            i64::from(identity.width),
            i64::from(identity.height),
            rotation_quarters,
        ],
    )?;
    Ok(())
}

fn resolve_external_aliases_for_asset_identity_tx(
    tx: &Transaction<'_>,
    asset_id: &AssetId,
    identity: &ImageIdentity,
) -> anyhow::Result<usize> {
    tx.execute(
        r"
        UPDATE external_items
        SET resolved_asset_id = ?1,
            updated_at = ?5
        WHERE imported_asset_id IS NULL
          AND (
            blob_id = ?2
            OR render_hash = ?3
            OR visual_key = ?4
          )
        ",
        params![
            asset_id.0,
            identity.blob_id.0,
            identity.render_hash.0,
            identity.visual_key.0,
            now_ts(),
        ],
    )
    .map_err(Into::into)
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
