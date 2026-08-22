use super::*;

const EXTERNAL_FRONTIER_IDENTITY_PREDICATE: &str = r"
      AND i.blob_id IS NOT NULL
      AND i.blob_id <> ''
      AND i.render_hash IS NOT NULL
      AND i.render_hash <> ''
      AND i.visual_key IS NOT NULL
      AND i.visual_key <> ''
";

#[derive(Debug, Clone, Copy, Default)]
pub struct ExternalItemWarmState {
    pub(crate) needs_materialization: bool,
    pub(crate) needs_identity: bool,
    pub(crate) needs_quality_features: bool,
    pub(crate) needs_embedding: bool,
    pub(crate) needs_clip_embedding: bool,
    pub(crate) needs_face_embedding: bool,
}

impl ExternalItemWarmState {
    pub(crate) fn needs_inline_work(self) -> bool {
        self.needs_materialization
            || self.needs_identity
            || self.needs_quality_features
            || self.needs_embedding
            || self.needs_clip_embedding
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalIdentityDisposition {
    Active,
    Tombstoned,
    Resolved(AssetId),
}

#[derive(Debug, Clone)]
pub struct UpsertedExternalStreamBatchEntry {
    pub stream_id: i64,
    pub blocked: bool,
    pub item_ids: HashMap<i64, RemoteItemId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExternalStreamFrontierCounts {
    pub ready_items: usize,
    pub live_items: usize,
    pub image_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalReadyCachePath {
    pub stream_id: i64,
    pub path: PathBuf,
}

impl Store {
    pub fn external_scan_due(&self, source_key: &str, interval: Duration) -> anyhow::Result<bool> {
        let last_scanned = self
            .conn
            .query_row(
                r"
                SELECT last_scanned_at
                FROM external_sources
                WHERE source_key = ?1
                ",
                params![source_key],
                |row| row.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten();
        Ok(last_scanned
            .is_none_or(|last_scanned| now_ts() - last_scanned >= interval.whole_seconds()))
    }

    pub fn upsert_external_source(
        &self,
        source_key: &str,
        display_name: &str,
        kind: &str,
        board: &str,
        last_error: Option<&str>,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO external_sources (
                source_key,
                display_name,
                kind,
                board,
                last_scanned_at,
                last_error,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?5)
            ON CONFLICT(source_key) DO UPDATE SET
                display_name = excluded.display_name,
                kind = excluded.kind,
                board = excluded.board,
                last_scanned_at = excluded.last_scanned_at,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            ",
            params![source_key, display_name, kind, board, now_ts(), last_error],
        )?;
        Ok(())
    }

    pub fn external_scan_fault(&self, source_key: &str, error: &str) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO external_sources (
                source_key,
                display_name,
                kind,
                board,
                last_scanned_at,
                last_error,
                updated_at
            ) VALUES (?1, ?1, 'unknown', '', ?2, ?3, ?2)
            ON CONFLICT(source_key) DO UPDATE SET
                last_scanned_at = excluded.last_scanned_at,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            ",
            params![source_key, now_ts(), error],
        )?;
        Ok(())
    }

    pub fn external_source_counts(
        &self,
        source_key: &str,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let active_streams = self
            .conn
            .query_row(
                r"
                SELECT COUNT(*)
                FROM external_streams
                WHERE source_key = ?1 AND active = 1 AND blocked = 0
                ",
                params![source_key],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or(0);
        let blocked_streams = self
            .conn
            .query_row(
                r"
                SELECT COUNT(*)
                FROM external_streams
                WHERE source_key = ?1 AND blocked = 1
                ",
                params![source_key],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or(0);
        let mut cached_stmt = self.conn.prepare(
            r"
            SELECT i.cached_path
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            WHERE i.source_key = ?1
              AND s.active = 1
              AND s.blocked = 0
              AND i.hidden = 0
              AND i.import_pending = 0
              AND i.resolved_asset_id IS NULL
              AND i.imported_asset_id IS NULL
              AND i.cached_path IS NOT NULL
              AND i.embedding IS NOT NULL
            ",
        )?;
        let cached_items = cached_stmt
            .query_map(params![source_key], |row| row.get::<_, String>(0))?
            .filter_map(Result::ok)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .count();
        Ok((
            usize::try_from(active_streams).unwrap_or_default(),
            usize::try_from(blocked_streams).unwrap_or_default(),
            cached_items,
        ))
    }

    pub fn external_source_ready_profile(
        &self,
        source_key: &str,
        model_name: &str,
    ) -> anyhow::Result<(usize, HashMap<i64, usize>)> {
        let query = format!(
            r"
            SELECT i.stream_id, i.cached_path
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            WHERE i.source_key = ?1
              AND s.active = 1
              AND s.blocked = 0
              AND i.hidden = 0
              AND i.import_pending = 0
              AND i.resolved_asset_id IS NULL
              AND i.imported_asset_id IS NULL
              AND i.cached_path IS NOT NULL
              AND i.embedding IS NOT NULL
              AND i.embedding_model = ?2
              {EXTERNAL_FRONTIER_IDENTITY_PREDICATE}
            "
        );
        let mut stmt = self.conn.prepare(&query)?;
        let mut by_stream = HashMap::new();
        let mut total = 0usize;
        let rows = stmt.query_map(params![source_key, model_name], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (stream_id, cached_path) = row?;
            if !PathBuf::from(cached_path).exists() {
                continue;
            }
            total += 1;
            *by_stream.entry(stream_id).or_default() += 1;
        }
        Ok((total, by_stream))
    }

    pub fn external_stream_frontier_counts(
        &self,
        source_key: &str,
        stream_id: i64,
        model_name: &str,
    ) -> anyhow::Result<Option<ExternalStreamFrontierCounts>> {
        let image_count = self
            .conn
            .query_row(
                r"
                SELECT image_count
                FROM external_streams
                WHERE source_key = ?1
                  AND id = ?2
                  AND active = 1
                  AND blocked = 0
                ",
                params![source_key, stream_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        let Some(image_count) = image_count else {
            return Ok(None);
        };
        let live_items = self
            .conn
            .query_row(
                r"
                SELECT COUNT(*)
                FROM external_items i
                JOIN external_streams s ON s.id = i.stream_id
                WHERE i.source_key = ?1
                  AND i.stream_id = ?2
                  AND s.active = 1
                  AND s.blocked = 0
                  AND i.hidden = 0
                  AND i.import_pending = 0
                  AND i.resolved_asset_id IS NULL
                  AND i.imported_asset_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM external_item_tombstones tombstone
                      WHERE tombstone.visual_key = i.visual_key
                  )
                ",
                params![source_key, stream_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or_default();
        let query = format!(
            r"
            SELECT i.cached_path
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            WHERE i.source_key = ?1
              AND i.stream_id = ?2
              AND s.active = 1
              AND s.blocked = 0
              AND i.hidden = 0
              AND i.import_pending = 0
              AND i.resolved_asset_id IS NULL
              AND i.imported_asset_id IS NULL
              AND i.cached_path IS NOT NULL
              AND i.embedding IS NOT NULL
              AND i.embedding_model = ?3
              {EXTERNAL_FRONTIER_IDENTITY_PREDICATE}
              AND NOT EXISTS (
                  SELECT 1
                  FROM external_item_tombstones tombstone
                  WHERE tombstone.visual_key = i.visual_key
              )
            "
        );
        let mut ready_stmt = self.conn.prepare(&query)?;
        let ready_items = ready_stmt
            .query_map(params![source_key, stream_id, model_name], |row| {
                row.get::<_, String>(0)
            })?
            .filter_map(Result::ok)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .count();
        Ok(Some(ExternalStreamFrontierCounts {
            ready_items,
            live_items: usize::try_from(live_items).unwrap_or_default(),
            image_count: usize::try_from(image_count).unwrap_or_default(),
        }))
    }

    pub fn external_source_ready_paths(
        &self,
        source_key: &str,
        model_name: &str,
    ) -> anyhow::Result<Vec<ExternalReadyCachePath>> {
        let query = format!(
            r"
            SELECT i.stream_id, i.cached_path
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            WHERE i.source_key = ?1
              AND s.active = 1
              AND s.blocked = 0
              AND i.hidden = 0
              AND i.import_pending = 0
              AND i.resolved_asset_id IS NULL
              AND i.imported_asset_id IS NULL
              AND i.cached_path IS NOT NULL
              AND i.embedding IS NOT NULL
              AND i.embedding_model = ?2
              {EXTERNAL_FRONTIER_IDENTITY_PREDICATE}
              AND NOT EXISTS (
                  SELECT 1
                  FROM external_item_tombstones tombstone
                  WHERE tombstone.visual_key = i.visual_key
              )
            ORDER BY s.last_modified DESC, i.post_no DESC
            "
        );
        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![source_key, model_name], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut paths = Vec::new();
        for row in rows {
            let (stream_id, cached_path) = row?;
            let path = PathBuf::from(cached_path);
            if !path.exists() {
                continue;
            }
            paths.push(ExternalReadyCachePath { stream_id, path });
        }
        Ok(paths)
    }
}
