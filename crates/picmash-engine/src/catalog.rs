use std::{
    ffi::OsString,
    os::unix::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
};

use rusqlite::{OptionalExtension, Transaction, params};
use walkdir::WalkDir;

use crate::{
    Engine,
    engine::now_ns,
    fault::{Fault, IoResultExt, Result},
    ids::{AssetId, CollectionId, OccurrenceId},
    media::{BlobDigest, ImageIdentity, RenderDigest, inspect_path, supported_path},
    model::{AssetOccurrence, AssetView, Collection, ScanFailure, ScanReport},
};

#[derive(Debug)]
struct Candidate {
    path: PathBuf,
    identity: ImageIdentity,
}

impl Engine {
    pub fn scan(&self, root: impl AsRef<Path>) -> Result<ScanReport> {
        let root = root.as_ref().canonicalize().at(root.as_ref())?;
        let now = now_ns()?;
        let collection_id = self.ensure_collection(&root, now)?;
        let mut candidates = Vec::new();
        let mut failures = Vec::new();
        let mut complete = true;

        for entry in WalkDir::new(&root).follow_links(false) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    complete = false;
                    failures.push(ScanFailure {
                        path: error.path().unwrap_or(&root).to_path_buf(),
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            if !entry.file_type().is_file() || !supported_path(entry.path()) {
                continue;
            }
            match inspect_path(entry.path()) {
                Ok(identity) => candidates.push(Candidate {
                    path: entry.path().to_path_buf(),
                    identity,
                }),
                Err(error) => failures.push(ScanFailure {
                    path: entry.path().to_path_buf(),
                    error: error.to_string(),
                }),
            }
        }

        self.apply_scan(collection_id, candidates, failures, complete, now)
    }

    pub fn collection(&self, id: CollectionId) -> Result<Collection> {
        let connection = self.connection.lock();
        connection
            .query_row(
                "SELECT root, catalog_revision FROM pm_collections WHERE id = ?1",
                [id.get()],
                |row| {
                    let root: Vec<u8> = row.get(0)?;
                    let revision: i64 = row.get(1)?;
                    Ok((root, revision))
                },
            )
            .map_err(Fault::from)
            .and_then(|(root, revision)| {
                Ok(Collection {
                    id,
                    root: decode_path(root),
                    catalog_revision: nonnegative_u64(revision, "catalog revision")?,
                })
            })
    }

    pub fn assets(&self, collection_id: CollectionId) -> Result<Vec<AssetView>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            r"
            WITH ranked AS (
                SELECT o.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY o.asset_id
                           ORDER BY o.width * o.height DESC, o.byte_len DESC, o.path ASC
                       ) AS rank
                FROM pm_occurrences o
                WHERE o.collection_id = ?1 AND o.present = 1
            ), latest_snapshot AS (
                SELECT id FROM pm_preference_snapshots
                WHERE collection_id = ?1
                ORDER BY id DESC LIMIT 1
            ), duel_counts AS (
                SELECT asset_id, COUNT(*) AS duel_count
                FROM (
                    SELECT d.left_asset_id AS asset_id
                    FROM pm_asset_duels d
                    JOIN pm_observations e ON e.id = d.observation_id
                    JOIN pm_sessions s ON s.id = e.session_id
                    WHERE s.collection_id = ?1
                    UNION ALL
                    SELECT d.right_asset_id AS asset_id
                    FROM pm_asset_duels d
                    JOIN pm_observations e ON e.id = d.observation_id
                    JOIN pm_sessions s ON s.id = e.session_id
                    WHERE s.collection_id = ?1
                )
                GROUP BY asset_id
            )
            SELECT a.id, r.id, r.path, r.blob_digest, a.render_digest,
                   r.width, r.height, r.byte_len, r.rotation_quarters,
                   ca.favorite, COALESCE(dc.duel_count, 0), ps.score
            FROM pm_collection_assets ca
            JOIN pm_assets a ON a.id = ca.asset_id
            JOIN ranked r ON r.asset_id = a.id AND r.rank = 1
            LEFT JOIN latest_snapshot ls
            LEFT JOIN pm_preference_scores ps ON ps.snapshot_id = ls.id AND ps.asset_id = a.id
            LEFT JOIN duel_counts dc ON dc.asset_id = a.id
            WHERE ca.collection_id = ?1 AND ca.hidden = 0
            ORDER BY ca.favorite DESC, ps.score DESC NULLS LAST, a.id ASC
            ",
        )?;
        let rows = statement.query_map([collection_id.get()], |row| {
            Ok(AssetRow {
                asset_id: row.get(0)?,
                occurrence_id: row.get(1)?,
                path: row.get(2)?,
                blob: row.get(3)?,
                render: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                byte_len: row.get(7)?,
                rotation_quarters: row.get(8)?,
                favorite: row.get(9)?,
                duel_count: row.get(10)?,
                preference_score: row.get(11)?,
            })
        })?;
        rows.map(|row| row.map_err(Fault::from).and_then(AssetRow::finish))
            .collect()
    }

    pub fn set_hidden(
        &self,
        collection_id: CollectionId,
        asset_id: &AssetId,
        hidden: bool,
    ) -> Result<()> {
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let changed = tx.execute(
            "UPDATE pm_collection_assets
             SET hidden = ?3, updated_at_ns = ?4
             WHERE collection_id = ?1 AND asset_id = ?2 AND hidden != ?3",
            params![collection_id.get(), asset_id.as_str(), hidden, now],
        )?;
        if changed == 0 && !collection_asset_exists(&tx, collection_id, asset_id)? {
            return Err(Fault::InvalidInput(format!(
                "asset {asset_id} does not belong to collection {collection_id}"
            )));
        }
        if changed != 0 {
            tx.execute(
                "UPDATE pm_collections SET catalog_revision = catalog_revision + 1 WHERE id = ?1",
                [collection_id.get()],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn rotate_occurrence(&self, id: OccurrenceId, clockwise_quarters: i8) -> Result<u8> {
        let turns = i64::from(clockwise_quarters).rem_euclid(4);
        if turns == 0 {
            let connection = self.connection.lock();
            let rotation = connection
                .query_row(
                    "SELECT rotation_quarters FROM pm_occurrences WHERE id = ?1",
                    [id.get()],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .ok_or_else(|| Fault::InvalidInput(format!("unknown occurrence {id}")))?;
            return u8::try_from(rotation)
                .map_err(|_| Fault::Corrupt("invalid stored rotation".to_owned()));
        }
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        tx.query_row(
            "SELECT id FROM pm_occurrences WHERE id = ?1",
            [id.get()],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .ok_or_else(|| Fault::InvalidInput(format!("unknown occurrence {id}")))?;
        tx.execute(
            "UPDATE pm_occurrences
             SET rotation_quarters = (rotation_quarters + ?2) % 4
             WHERE id = ?1",
            params![id.get(), turns],
        )?;
        let rotation = tx.query_row(
            "SELECT rotation_quarters FROM pm_occurrences WHERE id = ?1",
            [id.get()],
            |row| row.get::<_, i64>(0),
        )?;
        tx.commit()?;
        u8::try_from(rotation).map_err(|_| Fault::Corrupt("invalid stored rotation".to_owned()))
    }

    fn ensure_collection(&self, root: &Path, now: i64) -> Result<CollectionId> {
        let connection = self.connection.lock();
        connection.execute(
            "INSERT INTO pm_collections(root, created_at_ns) VALUES (?1, ?2)
             ON CONFLICT(root) DO NOTHING",
            params![encode_path(root), now],
        )?;
        let id = connection.query_row(
            "SELECT id FROM pm_collections WHERE root = ?1",
            [encode_path(root)],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(CollectionId::from_raw(id))
    }

    fn apply_scan(
        &self,
        collection_id: CollectionId,
        candidates: Vec<Candidate>,
        failures: Vec<ScanFailure>,
        complete: bool,
        now: i64,
    ) -> Result<ScanReport> {
        let discovered_paths = candidates.len() + failures.len();
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let generation = tx.query_row(
            "UPDATE pm_collections SET scan_generation = scan_generation + 1
             WHERE id = ?1 RETURNING scan_generation",
            [collection_id.get()],
            |row| row.get::<_, i64>(0),
        )?;
        let mut changed = false;
        for candidate in candidates {
            changed |= upsert_candidate(&tx, collection_id, generation, &candidate, now)?;
            tx.execute(
                "DELETE FROM pm_scan_failures WHERE collection_id = ?1 AND path = ?2",
                params![collection_id.get(), encode_path(&candidate.path)],
            )?;
        }
        for failure in &failures {
            let unavailable = tx.execute(
                "UPDATE pm_occurrences SET last_seen_generation = ?3, present = 0
                 WHERE collection_id = ?1 AND path = ?2 AND present = 1",
                params![collection_id.get(), encode_path(&failure.path), generation],
            )?;
            changed |= unavailable != 0;
            tx.execute(
                "INSERT INTO pm_scan_failures(collection_id, path, generation, error)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(collection_id, path) DO UPDATE SET
                     generation = excluded.generation, error = excluded.error",
                params![
                    collection_id.get(),
                    encode_path(&failure.path),
                    generation,
                    failure.error
                ],
            )?;
        }
        let retired = if complete {
            tx.execute(
                "UPDATE pm_occurrences SET present = 0
                 WHERE collection_id = ?1 AND present = 1 AND last_seen_generation != ?2",
                params![collection_id.get(), generation],
            )?
        } else {
            0
        };
        changed |= retired != 0;
        if changed {
            tx.execute(
                "UPDATE pm_collections SET catalog_revision = catalog_revision + 1 WHERE id = ?1",
                [collection_id.get()],
            )?;
        }
        let visible_assets = tx.query_row(
            "SELECT COUNT(*) FROM pm_collection_assets ca
             WHERE ca.collection_id = ?1 AND ca.hidden = 0
               AND EXISTS (
                   SELECT 1 FROM pm_occurrences o
                   WHERE o.collection_id = ca.collection_id AND o.asset_id = ca.asset_id
                     AND o.present = 1
               )",
            [collection_id.get()],
            |row| row.get::<_, usize>(0),
        )?;
        tx.commit()?;
        Ok(ScanReport {
            collection_id,
            generation: nonnegative_u64(generation, "scan generation")?,
            discovered_paths,
            visible_assets,
            retired_occurrences: retired,
            failures,
        })
    }
}

fn upsert_candidate(
    tx: &Transaction<'_>,
    collection_id: CollectionId,
    generation: i64,
    candidate: &Candidate,
    now: i64,
) -> Result<bool> {
    let existing = tx
        .query_row(
            "SELECT asset_id, blob_digest, width, height, byte_len, present
             FROM pm_occurrences WHERE collection_id = ?1 AND path = ?2",
            params![collection_id.get(), encode_path(&candidate.path)],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, bool>(5)?,
                ))
            },
        )
        .optional()?;
    let asset_id = tx
        .query_row(
            "SELECT id FROM pm_assets WHERE render_digest = ?1",
            [candidate.identity.render.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .map_or_else(
            || -> Result<AssetId> {
                let id = AssetId::fresh();
                tx.execute(
                    "INSERT INTO pm_assets(id, render_digest, identity_authority, created_at_ns)
                     VALUES (?1, ?2, 'exact', ?3)",
                    params![id.as_str(), candidate.identity.render.as_str(), now],
                )?;
                Ok(id)
            },
            |id| AssetId::parse(id).map_err(Fault::from),
        )?;
    let changed = existing.as_ref().is_none_or(|old| {
        old.0 != asset_id.as_str()
            || old.1 != candidate.identity.blob.as_str()
            || old.2 != i64::from(candidate.identity.width)
            || old.3 != i64::from(candidate.identity.height)
            || old.4 != i64::try_from(candidate.identity.byte_len).unwrap_or(i64::MAX)
            || !old.5
    });
    tx.execute(
        "INSERT INTO pm_occurrences(
             collection_id, path, asset_id, blob_digest, width, height, byte_len,
             present, last_seen_generation
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, ?8)
         ON CONFLICT(collection_id, path) DO UPDATE SET
             asset_id = excluded.asset_id,
             blob_digest = excluded.blob_digest,
             width = excluded.width,
             height = excluded.height,
             byte_len = excluded.byte_len,
             present = 1,
             last_seen_generation = excluded.last_seen_generation",
        params![
            collection_id.get(),
            encode_path(&candidate.path),
            asset_id.as_str(),
            candidate.identity.blob.as_str(),
            candidate.identity.width,
            candidate.identity.height,
            candidate.identity.byte_len,
            generation,
        ],
    )?;
    tx.execute(
        "INSERT INTO pm_collection_assets(collection_id, asset_id, updated_at_ns)
         VALUES (?1, ?2, ?3) ON CONFLICT(collection_id, asset_id) DO NOTHING",
        params![collection_id.get(), asset_id.as_str(), now],
    )?;
    Ok(changed)
}

fn collection_asset_exists(
    tx: &Transaction<'_>,
    collection_id: CollectionId,
    asset_id: &AssetId,
) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM pm_collection_assets WHERE collection_id = ?1 AND asset_id = ?2
         )",
        params![collection_id.get(), asset_id.as_str()],
        |row| row.get(0),
    )?)
}

pub fn encode_path(path: &Path) -> Vec<u8> {
    path.as_os_str().as_bytes().to_vec()
}

pub fn decode_path(bytes: Vec<u8>) -> PathBuf {
    PathBuf::from(OsString::from_vec(bytes))
}

fn nonnegative_u64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Fault::Corrupt(format!("negative {field}")))
}

struct AssetRow {
    asset_id: String,
    occurrence_id: i64,
    path: Vec<u8>,
    blob: String,
    render: String,
    width: i64,
    height: i64,
    byte_len: i64,
    rotation_quarters: i64,
    favorite: bool,
    duel_count: i64,
    preference_score: Option<f64>,
}

impl AssetRow {
    fn finish(self) -> Result<AssetView> {
        let asset_id = AssetId::parse(self.asset_id)?;
        Ok(AssetView {
            id: asset_id.clone(),
            occurrence: AssetOccurrence {
                id: OccurrenceId::from_raw(self.occurrence_id),
                asset_id,
                path: decode_path(self.path),
                blob: BlobDigest::parse(self.blob)?,
                render: RenderDigest::parse(self.render)?,
                width: u32::try_from(self.width)
                    .map_err(|_| Fault::Corrupt("invalid occurrence width".to_owned()))?,
                height: u32::try_from(self.height)
                    .map_err(|_| Fault::Corrupt("invalid occurrence height".to_owned()))?,
                byte_len: u64::try_from(self.byte_len)
                    .map_err(|_| Fault::Corrupt("invalid occurrence byte length".to_owned()))?,
                rotation_quarters: u8::try_from(self.rotation_quarters)
                    .map_err(|_| Fault::Corrupt("invalid occurrence rotation".to_owned()))?,
            },
            favorite: self.favorite,
            duel_count: u32::try_from(self.duel_count)
                .map_err(|_| Fault::Corrupt("invalid duel count".to_owned()))?,
            preference_score: self.preference_score,
        })
    }
}
