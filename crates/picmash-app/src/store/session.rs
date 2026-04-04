use super::*;

impl Store {
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
}
