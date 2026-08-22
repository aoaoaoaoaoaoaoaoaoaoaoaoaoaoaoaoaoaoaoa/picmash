use super::external_frontier::ExternalItemWarmState;
use super::*;
use std::collections::{HashMap, HashSet};

const CACHE_PATH_WITHDRAW_CHUNK: usize = 256;

macro_rules! external_frontier_identity_predicate {
    () => {
        r"
              AND i.blob_id IS NOT NULL
              AND i.blob_id <> ''
              AND i.render_hash IS NOT NULL
              AND i.render_hash <> ''
              AND i.visual_key IS NOT NULL
              AND i.visual_key <> ''
        "
    };
}

#[derive(Debug, Clone)]
pub struct ExternalStreamWarmCandidate {
    pub item_id: RemoteItemId,
    pub snapshot: RemoteItemSnapshot,
}

impl Store {
    pub fn retire_missing_external_streams(
        &self,
        source_key: &str,
        live_thread_nos: &[i64],
    ) -> anyhow::Result<()> {
        let placeholders = live_thread_nos
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        if live_thread_nos.is_empty() {
            self.conn.execute(
                r"
                UPDATE external_streams
                SET active = 0, updated_at = ?2
                WHERE source_key = ?1
                ",
                params![source_key, now_ts()],
            )?;
            return Ok(());
        }

        let sql = format!(
            "
            UPDATE external_streams
            SET active = 0, updated_at = ?2
            WHERE source_key = ?1
              AND thread_no NOT IN ({placeholders})
            "
        );
        let params = rusqlite::params_from_iter(
            std::iter::once(rusqlite::types::Value::from(source_key.to_owned()))
                .chain(std::iter::once(rusqlite::types::Value::from(now_ts())))
                .chain(
                    live_thread_nos
                        .iter()
                        .copied()
                        .map(rusqlite::types::Value::from),
                ),
        );
        self.conn.execute(&sql, params)?;
        Ok(())
    }

    pub fn withdraw_missing_external_items(
        &self,
        source_key: &str,
        live_post_nos: &[i64],
    ) -> anyhow::Result<usize> {
        if live_post_nos.is_empty() {
            return self
                .conn
                .execute(
                    r"
                    UPDATE external_items
                    SET cached_path = NULL, updated_at = ?2
                    WHERE source_key = ?1
                      AND import_pending = 0
                      AND resolved_asset_id IS NULL
                      AND imported_asset_id IS NULL
                      AND cached_path IS NOT NULL
                    ",
                    params![source_key, now_ts()],
                )
                .map_err(Into::into);
        }

        let placeholders = live_post_nos
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "
            UPDATE external_items
            SET cached_path = NULL, updated_at = ?2
            WHERE source_key = ?1
              AND import_pending = 0
              AND resolved_asset_id IS NULL
              AND imported_asset_id IS NULL
              AND cached_path IS NOT NULL
              AND post_no NOT IN ({placeholders})
            "
        );
        let params = rusqlite::params_from_iter(
            std::iter::once(rusqlite::types::Value::from(source_key.to_owned()))
                .chain(std::iter::once(rusqlite::types::Value::from(now_ts())))
                .chain(
                    live_post_nos
                        .iter()
                        .copied()
                        .map(rusqlite::types::Value::from),
                ),
        );
        self.conn.execute(&sql, params).map_err(Into::into)
    }

    pub fn withdraw_external_items_from_frontier(
        &self,
        item_ids: &[RemoteItemId],
    ) -> anyhow::Result<usize> {
        if item_ids.is_empty() {
            return Ok(0);
        }

        let placeholders = item_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "
            UPDATE external_items
            SET cached_path = NULL, updated_at = ?1
            WHERE cached_path IS NOT NULL
              AND import_pending = 0
              AND resolved_asset_id IS NULL
              AND imported_asset_id IS NULL
              AND id IN ({placeholders})
            "
        );
        let params = rusqlite::params_from_iter(
            std::iter::once(rusqlite::types::Value::from(now_ts())).chain(
                item_ids
                    .iter()
                    .map(|item_id| rusqlite::types::Value::from(item_id.0)),
            ),
        );
        self.conn.execute(&sql, params).map_err(Into::into)
    }

    pub fn withdraw_external_items_by_cached_paths(
        &self,
        cached_paths: &[PathBuf],
    ) -> anyhow::Result<usize> {
        let now = now_ts();
        let mut withdrawn = 0;
        for paths in cached_paths.chunks(CACHE_PATH_WITHDRAW_CHUNK) {
            if paths.is_empty() {
                continue;
            }
            let placeholders = paths.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let sql = format!(
                "
                UPDATE external_items
                SET cached_path = NULL, updated_at = ?1
                WHERE cached_path IS NOT NULL
                  AND import_pending = 0
                  AND resolved_asset_id IS NULL
                  AND imported_asset_id IS NULL
                  AND cached_path IN ({placeholders})
                "
            );
            let params =
                rusqlite::params_from_iter(
                    std::iter::once(rusqlite::types::Value::from(now)).chain(paths.iter().map(
                        |path| rusqlite::types::Value::from(path.to_string_lossy().into_owned()),
                    )),
                );
            withdrawn += self.conn.execute(&sql, params)?;
        }
        Ok(withdrawn)
    }

    pub fn upsert_external_stream(
        &self,
        source_key: &str,
        stream: &RemoteStreamSnapshot,
    ) -> anyhow::Result<(i64, bool)> {
        self.conn.execute(
            r"
            INSERT INTO external_streams (
                source_key,
                thread_no,
                title,
                semantic_slug,
                last_modified,
                reply_count,
                image_count,
                active,
                blocked,
                last_seen_at,
                last_scanned_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, 0, ?8, ?8, ?8)
            ON CONFLICT(source_key, thread_no) DO UPDATE SET
                title = excluded.title,
                semantic_slug = excluded.semantic_slug,
                last_modified = excluded.last_modified,
                reply_count = excluded.reply_count,
                image_count = excluded.image_count,
                active = 1,
                last_seen_at = excluded.last_seen_at,
                last_scanned_at = excluded.last_scanned_at,
                updated_at = excluded.updated_at
            ",
            params![
                source_key,
                stream.thread_no,
                stream.title,
                stream.semantic_slug,
                stream.last_modified,
                i64::from(stream.reply_count),
                i64::from(stream.image_count),
                now_ts(),
            ],
        )?;
        self.conn
            .query_row(
                r"
                SELECT id, blocked
                FROM external_streams
                WHERE source_key = ?1 AND thread_no = ?2
                ",
                params![source_key, stream.thread_no],
                |row| Ok((row.get(0)?, row.get::<_, i64>(1)? != 0)),
            )
            .map_err(Into::into)
    }

    pub fn upsert_external_item(
        &self,
        source_key: &str,
        stream_id: i64,
        stream_title: &str,
        item: &RemoteItemSnapshot,
        cached_path: Option<&Path>,
    ) -> anyhow::Result<RemoteItemId> {
        self.conn.execute(
            r"
            INSERT INTO external_items (
                source_key,
                stream_id,
                thread_no,
                post_no,
                stream_title,
                title,
                image_url,
                thumb_url,
                ext,
                md5,
                width,
                height,
                cached_path,
                rotation_quarters,
                hidden,
                imported_asset_id,
                last_seen_at,
                created_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 0, 0, NULL, ?14, ?14, ?14)
            ON CONFLICT(source_key, post_no) DO UPDATE SET
                stream_id = excluded.stream_id,
                thread_no = excluded.thread_no,
                stream_title = excluded.stream_title,
                title = excluded.title,
                image_url = excluded.image_url,
                thumb_url = excluded.thumb_url,
                ext = excluded.ext,
                md5 = excluded.md5,
                width = excluded.width,
                height = excluded.height,
                cached_path = COALESCE(excluded.cached_path, external_items.cached_path),
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at
            ",
            params![
                source_key,
                stream_id,
                item.thread_no,
                item.post_no,
                stream_title,
                item.title,
                item.image_url,
                item.thumb_url,
                item.ext,
                item.md5,
                i64::from(item.width),
                i64::from(item.height),
                cached_path.map(|path| path.to_string_lossy().into_owned()),
                now_ts(),
            ],
        )?;
        self.conn
            .query_row(
                r"
                SELECT id
                FROM external_items
                WHERE source_key = ?1 AND post_no = ?2
                ",
                params![source_key, item.post_no],
                |row| row.get::<_, i64>(0),
            )
            .map(RemoteItemId)
            .map_err(Into::into)
    }

    pub fn upsert_external_items(
        &mut self,
        source_key: &str,
        stream_id: i64,
        stream_title: &str,
        items: &[RemoteItemSnapshot],
    ) -> anyhow::Result<HashMap<i64, RemoteItemId>> {
        let tx = self.conn.transaction()?;
        let mut upsert = tx.prepare(
            r"
            INSERT INTO external_items (
                source_key,
                stream_id,
                thread_no,
                post_no,
                stream_title,
                title,
                image_url,
                thumb_url,
                ext,
                md5,
                width,
                height,
                cached_path,
                rotation_quarters,
                hidden,
                imported_asset_id,
                last_seen_at,
                created_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 0, 0, NULL, ?14, ?14, ?14)
            ON CONFLICT(source_key, post_no) DO UPDATE SET
                stream_id = excluded.stream_id,
                thread_no = excluded.thread_no,
                stream_title = excluded.stream_title,
                title = excluded.title,
                image_url = excluded.image_url,
                thumb_url = excluded.thumb_url,
                ext = excluded.ext,
                md5 = excluded.md5,
                width = excluded.width,
                height = excluded.height,
                cached_path = COALESCE(excluded.cached_path, external_items.cached_path),
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at
            ",
        )?;
        let mut fetch_id = tx.prepare(
            r"
            SELECT id
            FROM external_items
            WHERE source_key = ?1 AND post_no = ?2
            ",
        )?;
        let mut item_ids = HashMap::with_capacity(items.len());
        for item in items {
            upsert.execute(params![
                source_key,
                stream_id,
                item.thread_no,
                item.post_no,
                stream_title,
                item.title,
                item.image_url,
                item.thumb_url,
                item.ext,
                item.md5,
                i64::from(item.width),
                i64::from(item.height),
                item.materialized_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                now_ts(),
            ])?;
            let item_id = fetch_id.query_row(params![source_key, item.post_no], |row| {
                row.get::<_, i64>(0)
            })?;
            item_ids.insert(item.post_no, RemoteItemId(item_id));
        }
        drop(fetch_id);
        drop(upsert);
        tx.commit()?;
        Ok(item_ids)
    }

    pub fn upsert_external_streams_batch(
        &mut self,
        source_key: &str,
        streams: &[RemoteStreamSnapshot],
    ) -> anyhow::Result<Vec<UpsertedExternalStreamBatchEntry>> {
        let tx = self.conn.transaction()?;
        let mut upsert_stream = tx.prepare(
            r"
            INSERT INTO external_streams (
                source_key,
                thread_no,
                title,
                semantic_slug,
                last_modified,
                reply_count,
                image_count,
                active,
                blocked,
                last_seen_at,
                last_scanned_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, 0, ?8, ?8, ?8)
            ON CONFLICT(source_key, thread_no) DO UPDATE SET
                title = excluded.title,
                semantic_slug = excluded.semantic_slug,
                last_modified = excluded.last_modified,
                reply_count = excluded.reply_count,
                image_count = excluded.image_count,
                active = 1,
                last_seen_at = excluded.last_seen_at,
                last_scanned_at = excluded.last_scanned_at,
                updated_at = excluded.updated_at
            ",
        )?;
        let mut fetch_stream = tx.prepare(
            r"
            SELECT id, blocked
            FROM external_streams
            WHERE source_key = ?1 AND thread_no = ?2
            ",
        )?;
        let mut upsert_item = tx.prepare(
            r"
            INSERT INTO external_items (
                source_key,
                stream_id,
                thread_no,
                post_no,
                stream_title,
                title,
                image_url,
                thumb_url,
                ext,
                md5,
                width,
                height,
                cached_path,
                rotation_quarters,
                hidden,
                imported_asset_id,
                last_seen_at,
                created_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 0, 0, NULL, ?14, ?14, ?14)
            ON CONFLICT(source_key, post_no) DO UPDATE SET
                stream_id = excluded.stream_id,
                thread_no = excluded.thread_no,
                stream_title = excluded.stream_title,
                title = excluded.title,
                image_url = excluded.image_url,
                thumb_url = excluded.thumb_url,
                ext = excluded.ext,
                md5 = excluded.md5,
                width = excluded.width,
                height = excluded.height,
                cached_path = COALESCE(excluded.cached_path, external_items.cached_path),
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at
            ",
        )?;
        let mut fetch_stream_items = tx.prepare(
            r"
            SELECT id, post_no
            FROM external_items
            WHERE source_key = ?1 AND stream_id = ?2
            ",
        )?;
        let now = now_ts();
        let mut outcomes = Vec::with_capacity(streams.len());
        for stream in streams {
            upsert_stream.execute(params![
                source_key,
                stream.thread_no,
                stream.title,
                stream.semantic_slug,
                stream.last_modified,
                i64::from(stream.reply_count),
                i64::from(stream.image_count),
                now,
            ])?;
            let (stream_id, blocked) = fetch_stream
                .query_row(params![source_key, stream.thread_no], |row| {
                    Ok((row.get(0)?, row.get::<_, i64>(1)? != 0))
                })?;
            if blocked {
                outcomes.push(UpsertedExternalStreamBatchEntry {
                    stream_id,
                    blocked: true,
                    item_ids: HashMap::new(),
                });
                continue;
            }
            for item in &stream.items {
                upsert_item.execute(params![
                    source_key,
                    stream_id,
                    item.thread_no,
                    item.post_no,
                    stream.title,
                    item.title,
                    item.image_url,
                    item.thumb_url,
                    item.ext,
                    item.md5,
                    i64::from(item.width),
                    i64::from(item.height),
                    item.materialized_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    now,
                ])?;
            }
            let live_post_nos = stream
                .items
                .iter()
                .map(|item| item.post_no)
                .collect::<HashSet<_>>();
            let item_ids = fetch_stream_items
                .query_map(params![source_key, stream_id], |row| {
                    Ok((row.get::<_, i64>(1)?, RemoteItemId(row.get::<_, i64>(0)?)))
                })?
                .filter_map(Result::ok)
                .filter(|(post_no, _)| live_post_nos.contains(post_no))
                .collect::<HashMap<_, _>>();
            outcomes.push(UpsertedExternalStreamBatchEntry {
                stream_id,
                blocked: false,
                item_ids,
            });
        }
        drop(fetch_stream_items);
        drop(upsert_item);
        drop(fetch_stream);
        drop(upsert_stream);
        tx.commit()?;
        Ok(outcomes)
    }

    pub fn external_item_embedding(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        self.conn
            .query_row(
                r"
                SELECT embedding
                FROM external_items
                WHERE id = ?1 AND embedding_model = ?2
                ",
                params![item_id.0, model_name],
                |row| Ok(decode_vec_f32(&row.get::<_, Vec<u8>>(0)?)),
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn external_item_warm_state(
        &self,
        item_id: RemoteItemId,
        embedding_model_name: &str,
        clip_model_name: Option<&str>,
        face_model_name: &str,
        extractor_revision: &str,
    ) -> anyhow::Result<ExternalItemWarmState> {
        self.conn
            .query_row(
                r"
                SELECT
                    CASE
                        WHEN i.cached_path IS NULL OR i.cached_path = ''
                        THEN 1 ELSE 0
                    END,
                    CASE
                        WHEN i.blob_id IS NULL OR i.blob_id = ''
                          OR i.render_hash IS NULL OR i.render_hash = ''
                          OR i.visual_key IS NULL OR i.visual_key = ''
                        THEN 1 ELSE 0
                    END,
                    CASE WHEN qf.item_id IS NULL THEN 1 ELSE 0 END,
                    CASE
                        WHEN i.embedding IS NULL OR i.embedding_model != ?2
                        THEN 1 ELSE 0
                    END,
                    CASE
                        WHEN ?3 IS NOT NULL
                         AND (i.clip_embedding IS NULL OR i.clip_embedding_model != ?3)
                        THEN 1 ELSE 0
                    END,
                    CASE
                        WHEN i.face_embedding_model IS NULL OR i.face_embedding_model != ?4
                        THEN 1 ELSE 0
                    END
                FROM external_items i
                LEFT JOIN external_item_quality_features qf
                  ON qf.item_id = i.id
                 AND qf.extractor_revision = ?5
                WHERE i.id = ?1
                ",
                params![
                    item_id.0,
                    embedding_model_name,
                    clip_model_name,
                    face_model_name,
                    extractor_revision,
                ],
                |row| {
                    Ok(ExternalItemWarmState {
                        needs_materialization: row.get::<_, i64>(0)? != 0,
                        needs_identity: row.get::<_, i64>(1)? != 0,
                        needs_quality_features: row.get::<_, i64>(2)? != 0,
                        needs_embedding: row.get::<_, i64>(3)? != 0,
                        needs_clip_embedding: row.get::<_, i64>(4)? != 0,
                        needs_face_embedding: row.get::<_, i64>(5)? != 0,
                    })
                },
            )
            .optional()
            .map(|state| state.unwrap_or_default())
            .map_err(Into::into)
    }

    pub fn external_item_face_embedding_missing(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM external_items
                WHERE id = ?1
                  AND (
                    face_embedding_model IS NULL
                    OR face_embedding_model != ?2
                  )
                LIMIT 1
                ",
                params![item_id.0, model_name],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn external_item_face_embedding(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        self.conn
            .query_row(
                r"
                SELECT face_embedding
                FROM external_items
                WHERE id = ?1
                  AND face_embedding_model = ?2
                ",
                params![item_id.0, model_name],
                |row| row.get::<_, Option<Vec<u8>>>(0),
            )
            .optional()
            .map(|blob| {
                blob.flatten()
                    .and_then(|blob| (blob.len() % 4 == 0).then(|| decode_vec_f32(&blob)))
            })
            .map_err(Into::into)
    }

    pub fn external_items_missing_face_embedding_batch(
        &self,
        source_key: &str,
        embedding_model_name: &str,
        face_model_name: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(RemoteItemId, PathBuf)>> {
        let mut stmt = self.conn.prepare(concat!(
            r"
            SELECT i.id, i.cached_path
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
            ",
            external_frontier_identity_predicate!(),
            r"
              AND (
                    i.face_embedding_model IS NULL
                    OR i.face_embedding_model != ?3
                  )
            ORDER BY s.last_modified DESC, i.post_no DESC
            LIMIT ?4
            ",
        ))?;
        let rows = stmt.query_map(
            params![
                source_key,
                embedding_model_name,
                face_model_name,
                i64::try_from(limit)?
            ],
            |row| {
                Ok((
                    RemoteItemId(row.get(0)?),
                    PathBuf::from(row.get::<_, String>(1)?),
                ))
            },
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn external_item_quality_features_missing(
        &self,
        item_id: RemoteItemId,
        extractor_revision: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM external_item_quality_features
                WHERE item_id = ?1
                  AND extractor_revision = ?2
                LIMIT 1
                ",
                params![item_id.0, extractor_revision],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_none())
            .map_err(Into::into)
    }

    pub fn external_item_quality_features(
        &self,
        item_id: RemoteItemId,
        extractor_revision: &str,
    ) -> anyhow::Result<Option<crate::quality_features::AssetQualityFeatures>> {
        self.conn
            .query_row(
                r"
                SELECT technical_payload, vibe_payload
                FROM external_item_quality_features
                WHERE item_id = ?1
                  AND extractor_revision = ?2
                ",
                params![item_id.0, extractor_revision],
                |row| {
                    Ok(crate::quality_features::AssetQualityFeatures {
                        technical: crate::quality::decode_quality_payload(
                            &row.get::<_, String>(0)?,
                        )
                        .map_err(quality_payload_into_rusqlite)?,
                        vibe: crate::quality::decode_quality_payload(&row.get::<_, String>(1)?)
                            .map_err(quality_payload_into_rusqlite)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_external_item_quality_features(
        &self,
        item_id: RemoteItemId,
        extractor_revision: &str,
        features: &crate::quality_features::AssetQualityFeatures,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO external_item_quality_features (
                item_id,
                extractor_revision,
                technical_payload,
                vibe_payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(item_id) DO UPDATE SET
                extractor_revision = excluded.extractor_revision,
                technical_payload = excluded.technical_payload,
                vibe_payload = excluded.vibe_payload,
                updated_at = excluded.updated_at
            ",
            params![
                item_id.0,
                extractor_revision,
                crate::quality::encode_quality_payload(&features.technical)?,
                crate::quality::encode_quality_payload(&features.vibe)?,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn external_item_frontier_ready(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                concat!(
                    r"
                SELECT 1
                FROM external_items i
                WHERE i.id = ?1
                  AND i.hidden = 0
                  AND i.import_pending = 0
                  AND i.resolved_asset_id IS NULL
                  AND i.imported_asset_id IS NULL
                  AND i.cached_path IS NOT NULL
                  AND i.embedding IS NOT NULL
                  AND i.embedding_model = ?2
                ",
                    external_frontier_identity_predicate!(),
                    r"
                LIMIT 1
                ",
                ),
                params![item_id.0, model_name],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn external_item_identity_missing(&self, item_id: RemoteItemId) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM external_items
                WHERE id = ?1
                  AND (
                    blob_id IS NULL OR blob_id = ''
                    OR render_hash IS NULL OR render_hash = ''
                    OR visual_key IS NULL OR visual_key = ''
                  )
                LIMIT 1
                ",
                params![item_id.0],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn external_item_hidden(&self, item_id: RemoteItemId) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT hidden FROM external_items WHERE id = ?1",
                params![item_id.0],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map(|hidden| hidden.unwrap_or_default() != 0)
            .map_err(Into::into)
    }

    pub fn external_item_import_pending(&self, item_id: RemoteItemId) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT import_pending FROM external_items WHERE id = ?1",
                params![item_id.0],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map(|pending| pending.unwrap_or_default() != 0)
            .map_err(Into::into)
    }

    pub fn external_item_resolved_asset_id(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<AssetId>> {
        self.conn
            .query_row(
                "SELECT resolved_asset_id FROM external_items WHERE id = ?1",
                params![item_id.0],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .map(|asset_id| asset_id.flatten().map(AssetId))
            .map_err(Into::into)
    }

    pub fn pending_external_import_outcome(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<crate::model::PendingExternalImportOutcome>> {
        self.conn
            .query_row(
                r"
                SELECT session_id, corpus_id, outcome_kind
                FROM pending_external_import_outcomes
                WHERE item_id = ?1
                ",
                params![item_id.0],
                |row| {
                    Ok(crate::model::PendingExternalImportOutcome {
                        item_id,
                        session_id: SessionId(row.get(0)?),
                        corpus_id: CorpusId(row.get(1)?),
                        outcome_kind: row
                            .get::<_, String>(2)?
                            .parse()
                            .map_err(quality_payload_into_rusqlite)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn queue_external_import_outcome(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        outcome_kind: ExternalEventKind,
    ) -> anyhow::Result<bool> {
        let tx = self
            .conn
            .transaction()
            .context("opening pending external import transaction")?;
        let queued = tx.execute(
            r"
            UPDATE external_items
            SET import_pending = 1,
                updated_at = ?2
            WHERE id = ?1
              AND hidden = 0
              AND import_pending = 0
              AND resolved_asset_id IS NULL
              AND imported_asset_id IS NULL
            ",
            params![item_id.0, now_ts()],
        )?;
        if queued > 0 {
            tx.execute(
                r"
                INSERT OR IGNORE INTO pending_external_import_outcomes (
                    item_id,
                    session_id,
                    corpus_id,
                    outcome_kind,
                    queued_at,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?5)
                ",
                params![
                    item_id.0,
                    session_id.0,
                    corpus_id.0,
                    outcome_kind.as_str(),
                    now_ts(),
                ],
            )?;
            tx.execute(
                r"
                INSERT INTO maintenance_jobs (
                    kind,
                    job_key,
                    priority,
                    next_run_at,
                    generation,
                    running_generation,
                    attempts,
                    last_error,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, 1, NULL, 0, NULL, ?5)
                ON CONFLICT(kind, job_key) DO UPDATE SET
                    priority = MIN(maintenance_jobs.priority, excluded.priority),
                    next_run_at = MIN(maintenance_jobs.next_run_at, excluded.next_run_at),
                    generation = maintenance_jobs.generation + 1,
                    last_error = NULL,
                    updated_at = excluded.updated_at
                ",
                params![
                    crate::maintenance::MaintenanceJobKind::ExternalOutcomeSeal.as_str(),
                    item_id.0.to_string(),
                    crate::maintenance::MaintenancePriority::Hot.as_i64(),
                    now_ts(),
                    now_ts(),
                ],
            )?;
            touch_session_tx(&tx, session_id)?;
        }
        tx.commit()
            .context("committing pending external import transaction")?;
        Ok(queued > 0)
    }

    pub fn external_item_cached_path(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<PathBuf>> {
        self.conn
            .query_row(
                "SELECT cached_path FROM external_items WHERE id = ?1",
                params![item_id.0],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .map(|path| path.flatten().map(PathBuf::from))
            .map_err(Into::into)
    }

    pub fn save_external_item_cached_path(
        &self,
        item_id: RemoteItemId,
        cached_path: &Path,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE external_items
            SET cached_path = ?2,
                updated_at = ?3
            WHERE id = ?1
            ",
            params![
                item_id.0,
                cached_path.to_string_lossy().into_owned(),
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn save_external_item_identity(
        &self,
        item_id: RemoteItemId,
        identity: &ImageIdentity,
        cached_path: &Path,
    ) -> anyhow::Result<ExternalIdentityDisposition> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("opening external identity transaction")?;
        let tombstoned = tx
            .query_row(
                r"
                SELECT 1
                FROM external_item_tombstones
                WHERE blob_id = ?1 OR visual_key = ?2
                LIMIT 1
                ",
                params![identity.blob_id.0.as_str(), identity.visual_key.0.as_str()],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        let resolved_asset_id = resolve_asset_id_for_identity(&tx, identity)?;
        tx.execute(
            r"
            UPDATE external_items
            SET blob_id = ?2,
                render_hash = ?3,
                visual_key = ?4,
                cached_path = ?5,
                resolved_asset_id = COALESCE(?6, resolved_asset_id),
                hidden = CASE WHEN hidden = 1 OR ?7 = 1 THEN 1 ELSE 0 END,
                updated_at = ?8
            WHERE id = ?1
            ",
            params![
                item_id.0,
                identity.blob_id.0.as_str(),
                identity.render_hash.0.as_str(),
                identity.visual_key.0.as_str(),
                cached_path.to_string_lossy().into_owned(),
                resolved_asset_id
                    .as_ref()
                    .map(|asset_id| asset_id.0.as_str()),
                i64::from(tombstoned),
                now_ts(),
            ],
        )?;
        tx.commit()
            .context("committing external identity transaction")?;
        if tombstoned {
            return Ok(ExternalIdentityDisposition::Tombstoned);
        }
        Ok(match resolved_asset_id {
            Some(asset_id) => ExternalIdentityDisposition::Resolved(asset_id),
            None => ExternalIdentityDisposition::Active,
        })
    }

    pub fn save_external_embedding(
        &self,
        item_id: RemoteItemId,
        embedding: &EmbeddingRecord,
        cached_path: &Path,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE external_items
            SET embedding_model = ?2,
                embedding_dim = ?3,
                embedding = ?4,
                cached_path = ?5,
                updated_at = ?6
            WHERE id = ?1
            ",
            params![
                item_id.0,
                embedding.model_name,
                i64::try_from(embedding.vector.len())?,
                encode_vec_f32(&embedding.vector),
                cached_path.to_string_lossy().into_owned(),
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn external_item_clip_embedding(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        self.conn
            .query_row(
                r"
                SELECT clip_embedding
                FROM external_items
                WHERE id = ?1 AND clip_embedding_model = ?2
                ",
                params![item_id.0, model_name],
                |row| Ok(decode_vec_f32(&row.get::<_, Vec<u8>>(0)?)),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_external_clip_embedding(
        &self,
        item_id: RemoteItemId,
        embedding: &EmbeddingRecord,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE external_items
            SET clip_embedding_model = ?2,
                clip_embedding_dim = ?3,
                clip_embedding = ?4,
                updated_at = ?5
            WHERE id = ?1
            ",
            params![
                item_id.0,
                embedding.model_name,
                i64::try_from(embedding.vector.len())?,
                encode_vec_f32(&embedding.vector),
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn save_external_face_embedding(
        &self,
        item_id: RemoteItemId,
        model_name: &str,
        embedding: Option<&EmbeddingRecord>,
    ) -> anyhow::Result<()> {
        let (dim, blob) = match embedding {
            Some(embedding) => (
                Some(i64::try_from(embedding.vector.len())?),
                Some(encode_vec_f32(&embedding.vector)),
            ),
            None => (Some(0_i64), None),
        };
        self.conn.execute(
            r"
            UPDATE external_items
            SET face_embedding_model = ?2,
                face_embedding_dim = ?3,
                face_embedding = ?4,
                updated_at = ?5
            WHERE id = ?1
            ",
            params![item_id.0, model_name, dim, blob, now_ts()],
        )?;
        Ok(())
    }

    pub fn remote_item(&self, item_id: RemoteItemId) -> anyhow::Result<Option<RemoteItemRecord>> {
        self.conn
            .query_row(
                r"
                SELECT
                    id,
                    source_key,
                    stream_id,
                    stream_title,
                    thread_no,
                    post_no,
                    title,
                    cached_path,
                    image_url,
                    thumb_url,
                    visual_key,
                    rotation_quarters
                FROM external_items
                WHERE id = ?1
                ",
                params![item_id.0],
                |row| {
                    let Some(path) = row.get::<_, Option<String>>(7)? else {
                        return Ok(None);
                    };
                    Ok(Some(RemoteItemRecord {
                        id: RemoteItemId(row.get(0)?),
                        source_key: row.get(1)?,
                        stream_id: row.get(2)?,
                        stream_title: row.get(3)?,
                        thread_no: row.get(4)?,
                        post_no: row.get(5)?,
                        title: row.get(6)?,
                        path: PathBuf::from(path),
                        image_url: row.get(8)?,
                        thumb_url: row.get(9)?,
                        visual_key: row
                            .get::<_, Option<String>>(10)?
                            .map(crate::identity::VisualKey),
                        rotation_quarters: row.get(11)?,
                    }))
                },
            )
            .optional()
            .map(|item| item.flatten())
            .map_err(Into::into)
    }

    pub fn remote_item_snapshot(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<crate::sources::RemoteItemSnapshot>> {
        self.conn
            .query_row(
                r"
                SELECT
                    thread_no,
                    post_no,
                    title,
                    image_url,
                    thumb_url,
                    ext,
                    md5,
                    width,
                    height
                FROM external_items
                WHERE id = ?1
                ",
                params![item_id.0],
                |row| {
                    Ok(crate::sources::RemoteItemSnapshot {
                        thread_no: row.get(0)?,
                        post_no: row.get(1)?,
                        title: row.get(2)?,
                        image_url: row.get(3)?,
                        thumb_url: row.get(4)?,
                        ext: row.get(5)?,
                        md5: row.get(6)?,
                        width: u32::try_from(row.get::<_, i64>(7)?).unwrap_or_default(),
                        height: u32::try_from(row.get::<_, i64>(8)?).unwrap_or_default(),
                        file_size: 0,
                        materialized_path: None,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn arena_remote_item(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<RemoteItemRecord>> {
        self.conn
            .query_row(
                concat!(
                    r"
                SELECT
                    i.id,
                    i.source_key,
                    i.stream_id,
                    i.stream_title,
                    i.thread_no,
                    i.post_no,
                    i.title,
                    i.cached_path,
                    i.image_url,
                    i.thumb_url,
                    i.visual_key,
                    i.rotation_quarters
                FROM external_items i
                JOIN external_streams s ON s.id = i.stream_id
                WHERE i.id = ?1
                  AND i.hidden = 0
                  AND i.import_pending = 0
                  AND i.resolved_asset_id IS NULL
                  AND i.imported_asset_id IS NULL
                  AND s.active = 1
                  AND s.blocked = 0
                ",
                    external_frontier_identity_predicate!(),
                ),
                params![item_id.0],
                |row| {
                    let Some(path) = row.get::<_, Option<String>>(7)? else {
                        return Ok(None);
                    };
                    Ok(Some(RemoteItemRecord {
                        id: RemoteItemId(row.get(0)?),
                        source_key: row.get(1)?,
                        stream_id: row.get(2)?,
                        stream_title: row.get(3)?,
                        thread_no: row.get(4)?,
                        post_no: row.get(5)?,
                        title: row.get(6)?,
                        path: PathBuf::from(path),
                        image_url: row.get(8)?,
                        thumb_url: row.get(9)?,
                        visual_key: row
                            .get::<_, Option<String>>(10)?
                            .map(crate::identity::VisualKey),
                        rotation_quarters: row.get(11)?,
                    }))
                },
            )
            .optional()
            .map(|item| item.flatten())
            .map_err(Into::into)
    }

    pub fn remote_candidates(
        &self,
        source_key: &str,
        embedding_model_name: &str,
        face_model_name: &str,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<Vec<RemoteCandidate>> {
        let mut stmt = self.conn.prepare(concat!(
            r"
            SELECT
                i.id,
                i.source_key,
                i.stream_id,
                i.stream_title,
                i.thread_no,
                i.post_no,
                i.title,
                i.cached_path,
                i.image_url,
                i.thumb_url,
                i.visual_key,
                i.rotation_quarters,
                i.embedding,
                i.face_embedding_model,
                i.face_embedding,
                qf.technical_payload,
                qf.vibe_payload,
                qec.payload,
                i.selected_count,
                i.reject_count,
                i.survive_count,
                i.import_count,
                i.win_count,
                i.loss_count,
                s.selected_count,
                s.reject_count,
                s.survive_count,
                s.import_count,
                s.win_count,
                s.loss_count,
                s.image_count,
                s.last_modified
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            LEFT JOIN external_item_quality_features qf
              ON qf.item_id = i.id
             AND qf.extractor_revision = ?3
            LEFT JOIN quality_external_item_cache qec
              ON qec.item_id = i.id
             AND qec.formal_version = ?4
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
            ",
            external_frontier_identity_predicate!(),
            r"
              AND NOT EXISTS (
                  SELECT 1
                  FROM external_item_tombstones tombstone
                  WHERE tombstone.visual_key = i.visual_key
              )
            ORDER BY s.last_modified DESC, i.post_no DESC
            ",
        ))?;
        let rows = stmt.query_map(
            params![
                source_key,
                embedding_model_name,
                crate::quality_features::QUALITY_FEATURE_REVISION,
                formal_version.as_str(),
            ],
            |row| {
                Ok(RemoteCandidate {
                    item: RemoteItemRecord {
                        id: RemoteItemId(row.get(0)?),
                        source_key: row.get(1)?,
                        stream_id: row.get(2)?,
                        stream_title: row.get(3)?,
                        thread_no: row.get(4)?,
                        post_no: row.get(5)?,
                        title: row.get(6)?,
                        path: PathBuf::from(row.get::<_, String>(7)?),
                        image_url: row.get(8)?,
                        thumb_url: row.get(9)?,
                        visual_key: row
                            .get::<_, Option<String>>(10)?
                            .map(crate::identity::VisualKey),
                        rotation_quarters: row.get(11)?,
                    },
                    embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(12)?),
                    face_embedding: match (
                        row.get::<_, Option<String>>(13)?,
                        row.get::<_, Option<Vec<u8>>>(14)?,
                    ) {
                        (Some(face_model), Some(blob))
                            if face_model == face_model_name && blob.len() % 4 == 0 =>
                        {
                            Some(decode_vec_f32(&blob))
                        }
                        _ => None,
                    },
                    quality_features: match (
                        row.get::<_, Option<String>>(15)?,
                        row.get::<_, Option<String>>(16)?,
                    ) {
                        (Some(technical), Some(vibe)) => {
                            Some(crate::quality_features::AssetQualityFeatures {
                                technical: crate::quality::decode_quality_payload(&technical)
                                    .map_err(quality_payload_into_rusqlite)?,
                                vibe: crate::quality::decode_quality_payload(&vibe)
                                    .map_err(quality_payload_into_rusqlite)?,
                            })
                        }
                        _ => None,
                    },
                    quality_cache: row
                        .get::<_, Option<String>>(17)?
                        .map(|payload| crate::quality::decode_quality_payload(&payload))
                        .transpose()
                        .map_err(quality_payload_into_rusqlite)?,
                    selected_count: row.get(18)?,
                    reject_count: row.get(19)?,
                    survive_count: row.get(20)?,
                    import_count: row.get(21)?,
                    win_count: row.get(22)?,
                    loss_count: row.get(23)?,
                    stream_selected_count: row.get(24)?,
                    stream_reject_count: row.get(25)?,
                    stream_survive_count: row.get(26)?,
                    stream_import_count: row.get(27)?,
                    stream_win_count: row.get(28)?,
                    stream_loss_count: row.get(29)?,
                    stream_image_count: row.get(30)?,
                    stream_last_modified: row.get(31)?,
                })
            },
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn external_stream_warm_candidates(
        &self,
        source_key: &str,
        stream_id: i64,
    ) -> anyhow::Result<Vec<ExternalStreamWarmCandidate>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT
                i.id,
                i.thread_no,
                i.post_no,
                i.title,
                i.image_url,
                i.thumb_url,
                i.ext,
                i.md5,
                i.width,
                i.height,
                i.cached_path
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
            ORDER BY i.post_no DESC
            ",
        )?;
        let rows = stmt.query_map(params![source_key, stream_id], |row| {
            let image_url = row.get::<_, String>(4)?;
            let cached_path = row.get::<_, Option<String>>(10)?.map(PathBuf::from);
            Ok(ExternalStreamWarmCandidate {
                item_id: RemoteItemId(row.get(0)?),
                snapshot: RemoteItemSnapshot {
                    thread_no: row.get(1)?,
                    post_no: row.get(2)?,
                    title: row.get(3)?,
                    image_url: image_url.clone(),
                    thumb_url: row.get(5)?,
                    ext: row.get(6)?,
                    md5: row.get(7)?,
                    width: u32::try_from(row.get::<_, i64>(8)?).unwrap_or_default(),
                    height: u32::try_from(row.get::<_, i64>(9)?).unwrap_or_default(),
                    file_size: 0,
                    materialized_path: cached_path.or_else(|| file_url_to_path(&image_url)),
                },
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// All frontier embeddings across every source, L2-normalized.
    pub fn all_frontier_embeddings(
        &self,
        model_name: &str,
    ) -> anyhow::Result<Vec<(RemoteItemId, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(concat!(
            r"
            SELECT i.id, i.embedding
            FROM external_items i
            JOIN external_streams s ON s.id = i.stream_id
            WHERE s.active = 1
              AND s.blocked = 0
              AND i.hidden = 0
              AND i.import_pending = 0
              AND i.resolved_asset_id IS NULL
              AND i.imported_asset_id IS NULL
              AND i.cached_path IS NOT NULL
              AND i.embedding IS NOT NULL
              AND i.embedding_model = ?1
            ",
            external_frontier_identity_predicate!(),
        ))?;
        let rows = stmt.query_map(params![model_name], |row| {
            let id = RemoteItemId(row.get(0)?);
            let raw = decode_vec_f32(&row.get::<_, Vec<u8>>(1)?);
            Ok((id, raw))
        })?;
        let mut result = Vec::new();
        for row in rows {
            let (id, embedding) = row?;
            let norm = embedding
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            let normalized = embedding.iter().map(|v| v / norm).collect();
            result.push((id, normalized));
        }
        Ok(result)
    }

    pub fn recent_selected_external_item_ids(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> anyhow::Result<Vec<RemoteItemId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT item_id
            FROM external_events
            WHERE session_id = ?1
              AND event_kind <> ?2
            ORDER BY id DESC
            LIMIT ?3
            ",
        )?;
        let rows = stmt.query_map(
            params![
                session_id.0,
                ExternalEventKind::Imported.as_str(),
                i64::try_from(limit)?,
            ],
            |row| Ok(RemoteItemId(row.get(0)?)),
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn recent_selected_external_stream_ids(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> anyhow::Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT stream_id
            FROM external_events
            WHERE session_id = ?1
              AND event_kind <> ?2
            ORDER BY id DESC
            LIMIT ?3
            ",
        )?;
        let rows = stmt.query_map(
            params![
                session_id.0,
                ExternalEventKind::Imported.as_str(),
                i64::try_from(limit)?,
            ],
            |row| row.get(0),
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn recent_selected_external_source_keys(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> anyhow::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT source_key
            FROM external_events
            WHERE session_id = ?1
              AND event_kind <> ?2
            ORDER BY id DESC
            LIMIT ?3
            ",
        )?;
        let rows = stmt.query_map(
            params![
                session_id.0,
                ExternalEventKind::Imported.as_str(),
                i64::try_from(limit)?,
            ],
            |row| row.get(0),
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn external_source_recently_selected(
        &self,
        session_id: SessionId,
        source_key: &str,
        since_ts: i64,
    ) -> anyhow::Result<bool> {
        let seen = self.conn.query_row(
            r"
            SELECT EXISTS(
                SELECT 1
                FROM external_events
                WHERE session_id = ?1
                  AND source_key = ?2
                  AND event_kind <> ?3
                  AND created_at >= ?4
            )
            ",
            params![
                session_id.0,
                source_key,
                ExternalEventKind::Imported.as_str(),
                since_ts
            ],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(seen != 0)
    }

    pub fn note_external_selected(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        local_asset_id: &AssetId,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external selection transaction")?;
        tick_external_item_counter(&tx, item_id, "selected_count", 1)?;
        let stream_id = external_item_stream_id(&tx, item_id)?;
        tick_external_stream_counter(&tx, stream_id, "selected_count", 1)?;
        tx.execute(
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
            )
            SELECT ?1, ?2, source_key, stream_id, id, ?3, ?4, ?5
            FROM external_items
            WHERE id = ?6
            ",
            params![
                session_id.0,
                corpus_id.0,
                local_asset_id.0,
                ExternalEventKind::Selected.as_str(),
                now_ts(),
                item_id.0
            ],
        )?;
        tx.commit()
            .context("committing external selection transaction")?;
        Ok(())
    }

    pub fn reject_external_item(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        local_asset_id: Option<&AssetId>,
        event_kind: ExternalEventKind,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external reject transaction")?;
        let now = now_ts();
        let (blob_id, visual_key, md5) = tx
            .query_row(
                r"
                SELECT blob_id, visual_key, md5
                FROM external_items
                WHERE id = ?1
                ",
                params![item_id.0],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .with_context(|| format!("loading external identity for reject {}", item_id.0))?;
        tx.execute(
            r"
            UPDATE external_items
            SET hidden = 1,
                reject_count = reject_count + 1,
                updated_at = ?2
            WHERE id = ?1
            ",
            params![item_id.0, now],
        )?;
        if let (Some(blob_id), Some(visual_key)) = (blob_id.as_deref(), visual_key.as_deref()) {
            tx.execute(
                r"
                INSERT OR IGNORE INTO external_item_tombstones (blob_id, visual_key, created_at)
                VALUES (?1, ?2, ?3)
                ",
                params![blob_id, visual_key, now],
            )?;
            let identity_siblings = tx.execute(
                r"
                UPDATE external_items
                SET hidden = 1,
                    reject_count = reject_count + 1,
                    updated_at = ?4
                WHERE hidden = 0
                  AND import_pending = 0
                  AND resolved_asset_id IS NULL
                  AND imported_asset_id IS NULL
                  AND id != ?1
                  AND (
                    blob_id = ?2
                    OR visual_key = ?3
                  )
                ",
                params![item_id.0, blob_id, visual_key, now],
            )?;
            if identity_siblings > 0 {
                info!(
                    item_id = item_id.0,
                    identity_siblings, "propagated rejection by canonical remote identity"
                );
            }
        }
        if let Some(md5) = md5.as_deref() {
            let md5_siblings = tx.execute(
                r"
                UPDATE external_items
                SET hidden = 1,
                    reject_count = reject_count + 1,
                    updated_at = ?3
                WHERE hidden = 0
                  AND import_pending = 0
                  AND resolved_asset_id IS NULL
                  AND imported_asset_id IS NULL
                  AND md5 = ?2
                  AND id != ?1
                ",
                params![item_id.0, md5, now],
            )?;
            if md5_siblings > 0 {
                info!(
                    item_id = item_id.0,
                    md5_siblings, "propagated rejection by md5"
                );
            }
        }
        let stream_id = external_item_stream_id(&tx, item_id)?;
        tick_external_stream_counter(&tx, stream_id, "reject_count", 1)?;
        tx.execute(
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
            )
            SELECT ?1, ?2, source_key, stream_id, id, ?3, ?4, ?5
            FROM external_items
            WHERE id = ?6
            ",
            params![
                session_id.0,
                corpus_id.0,
                local_asset_id.map(|asset_id| asset_id.0.clone()),
                event_kind.as_str(),
                now_ts(),
                item_id.0
            ],
        )?;
        tx.commit()
            .context("committing external reject transaction")?;
        Ok(())
    }

    pub fn block_external_stream(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        local_asset_id: Option<&AssetId>,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external stream block transaction")?;
        let stream_id = external_item_stream_id(&tx, item_id)?;
        tx.execute(
            r"
            UPDATE external_streams
            SET blocked = 1,
                updated_at = ?2
            WHERE id = ?1
            ",
            params![stream_id, now_ts()],
        )?;
        tick_external_stream_counter(&tx, stream_id, "reject_count", 1)?;
        tx.execute(
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
            )
            SELECT ?1, ?2, source_key, stream_id, id, ?3, ?4, ?5
            FROM external_items
            WHERE id = ?6
            ",
            params![
                session_id.0,
                corpus_id.0,
                local_asset_id.map(|asset_id| asset_id.0.clone()),
                ExternalEventKind::StreamBlocked.as_str(),
                now_ts(),
                item_id.0
            ],
        )?;
        tx.commit()
            .context("committing external stream block transaction")?;
        Ok(())
    }

    pub fn rotate_external_item(
        &self,
        item_id: RemoteItemId,
        direction: i32,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE external_items
            SET rotation_quarters = ((rotation_quarters + ?2) % 4 + 4) % 4,
                updated_at = ?3
            WHERE id = ?1
            ",
            params![item_id.0, direction, now_ts()],
        )?;
        Ok(())
    }

    pub fn record_external_result(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        local_asset_id: &AssetId,
        event_kind: ExternalEventKind,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external result transaction")?;
        record_external_result_tx(
            &tx,
            session_id,
            corpus_id,
            item_id,
            local_asset_id,
            event_kind,
        )?;
        tx.commit()
            .context("committing external result transaction")?;
        Ok(())
    }

    pub fn link_external_import(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        asset_id: &AssetId,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external import transaction")?;
        link_external_import_tx(&tx, session_id, corpus_id, item_id, asset_id)?;
        tx.commit()
            .context("committing external import transaction")?;
        Ok(())
    }

    pub fn seal_imported_external_outcome_precomputed(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        import_path: &Path,
        identity: &ImageIdentity,
        rotation_quarters: i32,
        embedding: Option<&EmbeddingRecord>,
        outcome_kind: ExternalEventKind,
    ) -> anyhow::Result<AssetId> {
        let path_string = import_path.to_string_lossy().into_owned();
        let tx = self
            .conn
            .transaction()
            .context("opening imported external outcome transaction")?;
        let asset_id = resolve_asset_id_for_identity(&tx, identity)?.unwrap_or_else(mint_asset_id);
        let hidden = preserved_hidden_state_tx(&tx, corpus_id, &path_string, &asset_id)?;
        upsert_asset_identity_tx(&tx, &asset_id, identity, rotation_quarters.rem_euclid(4))?;
        resolve_external_aliases_for_asset_identity_tx(&tx, &asset_id, identity)?;
        upsert_corpus_variant_tx(&tx, corpus_id, &path_string, &asset_id, identity, hidden)?;
        if let Some(embedding) = embedding {
            upsert_embedding_tx(&tx, &asset_id, embedding)?;
        }
        link_external_import_tx(&tx, session_id, corpus_id, item_id, &asset_id)?;
        tx.execute(
            "DELETE FROM pending_external_import_outcomes WHERE item_id = ?1",
            params![item_id.0],
        )?;
        record_external_result_tx(&tx, session_id, corpus_id, item_id, &asset_id, outcome_kind)?;
        touch_session_tx(&tx, session_id)?;
        tx.commit()
            .context("committing imported external outcome transaction")?;
        Ok(asset_id)
    }

    pub fn record_external_result_and_touch_session(
        &mut self,
        session_id: SessionId,
        corpus_id: CorpusId,
        item_id: RemoteItemId,
        local_asset_id: &AssetId,
        event_kind: ExternalEventKind,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening external result+touch transaction")?;
        record_external_result_tx(
            &tx,
            session_id,
            corpus_id,
            item_id,
            local_asset_id,
            event_kind,
        )?;
        touch_session_tx(&tx, session_id)?;
        tx.commit()
            .context("committing external result+touch transaction")?;
        Ok(())
    }
}

fn external_item_stream_id(tx: &Transaction<'_>, item_id: RemoteItemId) -> anyhow::Result<i64> {
    tx.query_row(
        "SELECT stream_id FROM external_items WHERE id = ?1",
        params![item_id.0],
        |row| row.get(0),
    )
    .with_context(|| format!("loading external stream for item {}", item_id.0))
}

fn tick_external_item_counter(
    tx: &Transaction<'_>,
    item_id: RemoteItemId,
    column: &str,
    delta: i64,
) -> anyhow::Result<()> {
    tx.execute(
        &format!(
            "
            UPDATE external_items
            SET {column} = {column} + ?2,
                updated_at = ?3
            WHERE id = ?1
            "
        ),
        params![item_id.0, delta, now_ts()],
    )?;
    Ok(())
}

fn tick_external_stream_counter(
    tx: &Transaction<'_>,
    stream_id: i64,
    column: &str,
    delta: i64,
) -> anyhow::Result<()> {
    tx.execute(
        &format!(
            "
            UPDATE external_streams
            SET {column} = {column} + ?2,
                updated_at = ?3
            WHERE id = ?1
            "
        ),
        params![stream_id, delta, now_ts()],
    )?;
    Ok(())
}

fn record_external_result_tx(
    tx: &Transaction<'_>,
    session_id: SessionId,
    corpus_id: CorpusId,
    item_id: RemoteItemId,
    local_asset_id: &AssetId,
    event_kind: ExternalEventKind,
) -> anyhow::Result<()> {
    let stream_id = external_item_stream_id(tx, item_id)?;
    match event_kind {
        ExternalEventKind::LocalWin => {
            tick_external_item_counter(tx, item_id, "loss_count", 1)?;
            tick_external_item_counter(tx, item_id, "survive_count", 1)?;
            tick_external_stream_counter(tx, stream_id, "loss_count", 1)?;
            tick_external_stream_counter(tx, stream_id, "survive_count", 1)?;
        }
        ExternalEventKind::RemoteWin => {
            tick_external_item_counter(tx, item_id, "win_count", 1)?;
            tick_external_item_counter(tx, item_id, "survive_count", 1)?;
            tick_external_stream_counter(tx, stream_id, "win_count", 1)?;
            tick_external_stream_counter(tx, stream_id, "survive_count", 1)?;
        }
        ExternalEventKind::Hearted | ExternalEventKind::Kept => {
            tick_external_item_counter(tx, item_id, "survive_count", 1)?;
            tick_external_stream_counter(tx, stream_id, "survive_count", 1)?;
        }
        other => bail!("unsupported external result kind: {}", other.as_str()),
    }
    tx.execute(
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
        )
        SELECT ?1, ?2, source_key, stream_id, id, ?3, ?4, ?5
        FROM external_items
        WHERE id = ?6
        ",
        params![
            session_id.0,
            corpus_id.0,
            local_asset_id.0,
            event_kind.as_str(),
            now_ts(),
            item_id.0
        ],
    )?;
    Ok(())
}

fn link_external_import_tx(
    tx: &Transaction<'_>,
    session_id: SessionId,
    corpus_id: CorpusId,
    item_id: RemoteItemId,
    asset_id: &AssetId,
) -> anyhow::Result<()> {
    tx.execute(
        r"
        UPDATE external_items
        SET imported_asset_id = ?2,
            resolved_asset_id = ?2,
            import_pending = 0,
            import_count = import_count + 1,
            updated_at = ?3
        WHERE id = ?1
        ",
        params![item_id.0, asset_id.0, now_ts()],
    )?;
    let stream_id = external_item_stream_id(tx, item_id)?;
    tick_external_stream_counter(tx, stream_id, "import_count", 1)?;
    tx.execute(
        r"
        INSERT OR REPLACE INTO asset_external_provenance (
            asset_id,
            source_key,
            stream_id,
            item_id,
            imported_at
        )
        SELECT ?2, source_key, stream_id, id, ?3
        FROM external_items
        WHERE id = ?1
        ",
        params![item_id.0, asset_id.0, now_ts()],
    )?;
    tx.execute(
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
        )
        SELECT ?1, ?2, source_key, stream_id, id, ?3, ?4, ?5
        FROM external_items
        WHERE id = ?6
        ",
        params![
            session_id.0,
            corpus_id.0,
            asset_id.0,
            ExternalEventKind::Imported.as_str(),
            now_ts(),
            item_id.0
        ],
    )?;
    Ok(())
}

fn quality_payload_into_rusqlite(error: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(error.to_string())),
    )
}

fn file_url_to_path(image_url: &str) -> Option<PathBuf> {
    image_url.strip_prefix("file://").map(PathBuf::from)
}

fn touch_session_tx(tx: &Transaction<'_>, session_id: SessionId) -> anyhow::Result<()> {
    tx.execute(
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
