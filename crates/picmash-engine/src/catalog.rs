use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    os::unix::{
        ffi::{OsStrExt, OsStringExt},
        fs::MetadataExt as _,
    },
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::anyhow;
use rayon::{ThreadPoolBuilder, prelude::*};
use rusqlite::{OptionalExtension, Transaction, params};
use walkdir::WalkDir;

use crate::{
    Engine,
    engine::now_ns,
    fault::{Fault, IoResultExt, Result},
    ids::{AssetId, CollectionId, OccurrenceId},
    media::{
        BlobDigest, ImageIdentity, RenderDigest, blob_digest, inspect_bytes_with_blob,
        supported_path,
    },
    model::{AssetOccurrence, AssetView, Collection, ScanFailure, ScanProgress, ScanReport},
};

const SCAN_THREADS: usize = 2;
const PROGRESS_STEPS: usize = 100;

#[derive(Debug)]
struct Candidate {
    path: PathBuf,
    identity: ImageIdentity,
    seal: FileSeal,
}

#[derive(Clone, Debug)]
struct CachedIdentity {
    seal: Option<FileSeal>,
    identity: ImageIdentity,
}

#[derive(Debug, Default)]
struct ScanCache {
    by_path: HashMap<PathBuf, CachedIdentity>,
    by_blob: HashMap<BlobDigest, ImageIdentity>,
}

#[derive(Debug)]
enum Inspection {
    Candidate { candidate: Candidate, reused: bool },
    Failure(ScanFailure),
    Halted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileSeal([u8; 56]);

impl FileSeal {
    fn read(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path).at(path)?;
        let components = [
            metadata.dev().to_le_bytes(),
            metadata.ino().to_le_bytes(),
            metadata.len().to_le_bytes(),
            metadata.mtime().to_le_bytes(),
            metadata.mtime_nsec().to_le_bytes(),
            metadata.ctime().to_le_bytes(),
            metadata.ctime_nsec().to_le_bytes(),
        ];
        let mut bytes = [0; 56];
        for (index, component) in components.into_iter().enumerate() {
            bytes[index * 8..][..8].copy_from_slice(&component);
        }
        Ok(Self(bytes))
    }

    fn parse(bytes: Vec<u8>) -> Result<Self> {
        bytes.try_into().map(Self).map_err(|bytes: Vec<u8>| {
            Fault::Corrupt(format!("invalid file seal length {}", bytes.len()))
        })
    }

    const fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Engine {
    pub fn scan(&self, root: impl AsRef<Path>) -> Result<ScanReport> {
        self.scan_with_control(root, || false, |_| {})?
            .ok_or_else(|| Fault::Corrupt("an unconditional collection scan halted".to_owned()))
    }

    pub fn scan_with_halt(
        &self,
        root: impl AsRef<Path>,
        halt: impl Fn() -> bool + Sync,
    ) -> Result<Option<ScanReport>> {
        self.scan_with_control(root, halt, |_| {})
    }

    pub fn scan_with_control(
        &self,
        root: impl AsRef<Path>,
        halt: impl Fn() -> bool + Sync,
        progress: impl Fn(ScanProgress) + Sync,
    ) -> Result<Option<ScanReport>> {
        let root = root.as_ref().canonicalize().at(root.as_ref())?;
        let cache = self.scan_cache(&root)?;
        let mut paths = Vec::new();
        let mut failures = Vec::new();
        let mut complete = true;

        for entry in WalkDir::new(&root).follow_links(false) {
            if halt() {
                return Ok(None);
            }
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
            paths.push(entry.path().to_path_buf());
        }

        let total = paths.len();
        progress(ScanProgress {
            inspected_paths: 0,
            total_paths: total,
            reused_paths: 0,
        });
        let completed = AtomicUsize::new(0);
        let reused = AtomicUsize::new(0);
        let stride = (total / PROGRESS_STEPS).max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(SCAN_THREADS)
            .thread_name(|index| format!("picmash-scan-{index}"))
            .build()
            .map_err(|error| Fault::Runtime(format!("raise bounded scan pool: {error}")))?;
        let inspections = pool.install(|| {
            paths
                .par_iter()
                .map(|path| {
                    if halt() {
                        return Inspection::Halted;
                    }
                    let inspected = inspect_candidate(path, &cache);
                    let was_reused = inspected.as_ref().is_ok_and(|(_candidate, reused)| *reused);
                    let inspected_paths = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if was_reused {
                        let _prior = reused.fetch_add(1, Ordering::Relaxed);
                    }
                    if inspected_paths == total || inspected_paths.is_multiple_of(stride) {
                        progress(ScanProgress {
                            inspected_paths,
                            total_paths: total,
                            reused_paths: reused.load(Ordering::Relaxed),
                        });
                    }
                    match inspected {
                        Ok((candidate, reused)) => Inspection::Candidate { candidate, reused },
                        Err(error) => Inspection::Failure(ScanFailure {
                            path: path.clone(),
                            error: error.to_string(),
                        }),
                    }
                })
                .collect::<Vec<_>>()
        });
        if halt()
            || inspections
                .iter()
                .any(|inspection| matches!(inspection, Inspection::Halted))
        {
            return Ok(None);
        }
        let mut candidates = Vec::with_capacity(inspections.len());
        let mut reused_paths = 0;
        for inspection in inspections {
            match inspection {
                Inspection::Candidate { candidate, reused } => {
                    candidates.push(candidate);
                    reused_paths += usize::from(reused);
                }
                Inspection::Failure(failure) => failures.push(failure),
                Inspection::Halted => return Ok(None),
            }
        }

        if halt() {
            return Ok(None);
        }
        let now = now_ns()?;
        let collection_id = self.ensure_collection(&root, now)?;
        self.apply_scan(
            collection_id,
            candidates,
            failures,
            complete,
            reused_paths,
            now,
        )
        .map(Some)
    }

    fn scan_cache(&self, root: &Path) -> Result<ScanCache> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT o.path, o.file_seal, o.blob_digest, a.render_digest,
                    o.width, o.height, o.byte_len
             FROM pm_occurrences o
             JOIN pm_collections c ON c.id = o.collection_id
             JOIN pm_assets a ON a.id = o.asset_id
             WHERE c.root = ?1 AND a.identity_authority = 'exact'",
        )?;
        let rows = statement.query_map([encode_path(root)], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Option<Vec<u8>>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
            ))
        })?;
        let mut cache = ScanCache::default();
        for row in rows {
            let (path, seal, blob, render, width, height, byte_len) = row?;
            let blob = BlobDigest::parse(blob)?;
            let identity = ImageIdentity {
                blob: blob.clone(),
                render: RenderDigest::parse(render)?,
                width: u32::try_from(width)
                    .map_err(|_| Fault::Corrupt("invalid cached width".to_owned()))?,
                height: u32::try_from(height)
                    .map_err(|_| Fault::Corrupt("invalid cached height".to_owned()))?,
                byte_len: u64::try_from(byte_len)
                    .map_err(|_| Fault::Corrupt("invalid cached byte length".to_owned()))?,
            };
            let _prior = cache.by_blob.insert(blob, identity.clone());
            let _prior = cache.by_path.insert(
                decode_path(path),
                CachedIdentity {
                    seal: seal.map(FileSeal::parse).transpose()?,
                    identity,
                },
            );
        }
        Ok(cache)
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
                       COUNT(*) OVER (PARTITION BY o.asset_id) AS occurrence_count,
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
            )
            SELECT a.id, r.id, r.path, r.blob_digest, a.render_digest,
                   r.width, r.height, r.byte_len, r.rotation_quarters,
                   r.occurrence_count, ca.favorite, COALESCE(ps.duel_count, 0), ps.score
            FROM pm_collection_assets ca
            JOIN pm_assets a ON a.id = ca.asset_id
            JOIN ranked r ON r.asset_id = a.id AND r.rank = 1
            LEFT JOIN latest_snapshot ls
            LEFT JOIN pm_preference_scores ps ON ps.snapshot_id = ls.id AND ps.asset_id = a.id
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
                occurrence_count: row.get(9)?,
                favorite: row.get(10)?,
                duel_count: row.get(11)?,
                preference_score: row.get(12)?,
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
        reused_paths: usize,
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
            reused_paths,
            visible_assets,
            retired_occurrences: retired,
            failures,
        })
    }
}

fn inspect_candidate(path: &Path, cache: &ScanCache) -> Result<(Candidate, bool)> {
    let seal = FileSeal::read(path)?;
    if let Some(cached) = cache
        .by_path
        .get(path)
        .filter(|cached| cached.seal == Some(seal))
    {
        return Ok((
            Candidate {
                path: path.to_path_buf(),
                identity: cached.identity.clone(),
                seal,
            },
            true,
        ));
    }

    let bytes = fs::read(path).at(path)?;
    if FileSeal::read(path)? != seal {
        return Err(Fault::Image {
            path: path.to_path_buf(),
            source: anyhow!("file changed while it was being inspected"),
        });
    }
    let blob = blob_digest(&bytes);
    let (identity, reused) = match cache.by_blob.get(&blob) {
        Some(identity) => (identity.clone(), true),
        None => (
            inspect_bytes_with_blob(&bytes, blob).map_err(|source| Fault::Image {
                path: path.to_path_buf(),
                source,
            })?,
            false,
        ),
    };
    Ok((
        Candidate {
            path: path.to_path_buf(),
            identity,
            seal,
        },
        reused,
    ))
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
             present, last_seen_generation, file_seal
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, ?8, ?9)
         ON CONFLICT(collection_id, path) DO UPDATE SET
             asset_id = excluded.asset_id,
             blob_digest = excluded.blob_digest,
             width = excluded.width,
             height = excluded.height,
             byte_len = excluded.byte_len,
             present = 1,
             last_seen_generation = excluded.last_seen_generation,
             file_seal = excluded.file_seal",
        params![
            collection_id.get(),
            encode_path(&candidate.path),
            asset_id.as_str(),
            candidate.identity.blob.as_str(),
            candidate.identity.width,
            candidate.identity.height,
            candidate.identity.byte_len,
            generation,
            candidate.seal.as_bytes(),
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
    occurrence_count: i64,
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
            occurrence_count: u32::try_from(self.occurrence_count)
                .map_err(|_| Fault::Corrupt("invalid occurrence count".to_owned()))?,
            favorite: self.favorite,
            duel_count: u32::try_from(self.duel_count)
                .map_err(|_| Fault::Corrupt("invalid duel count".to_owned()))?,
            preference_score: self.preference_score,
        })
    }
}
