use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, ensure};
use rusqlite::{Connection, Transaction, params};
use walkdir::WalkDir;

use super::{
    Discovery, DuelVictor, FileStamp, Harvest, Origin, Prepared, PromotionIntent,
    PromotionJudgment, RemoteItemId, SourceIx, StreamId,
};
use crate::configuration::{ImportPolicy, RemoteConfig, SourceConfig, SourceIdentity};

const SCHEMA: i64 = 4;
const PROMOTION_SCHEMA: &str = "CREATE TABLE pr_promotions(
         item_id TEXT PRIMARY KEY REFERENCES pr_items(item_id) ON DELETE CASCADE,
         collection_root BLOB NOT NULL,
         rotation_quarters INTEGER NOT NULL CHECK(rotation_quarters BETWEEN 0 AND 3),
         judgment_kind TEXT NOT NULL CHECK(judgment_kind IN ('duel', 'favorite')),
         session_id TEXT NOT NULL,
         anchor_asset_id TEXT,
         anchor_occurrence_id INTEGER,
         anchor_render_digest TEXT,
         anchor_rotation_quarters INTEGER CHECK(anchor_rotation_quarters BETWEEN 0 AND 3),
         victor TEXT CHECK(victor IN ('anchor', 'challenger')),
         command_id TEXT NOT NULL,
         response_ms INTEGER CHECK(response_ms BETWEEN 0 AND 4294967295),
         committed_at_ns INTEGER NOT NULL,
         CHECK(
             (judgment_kind = 'duel' AND anchor_asset_id IS NOT NULL
                 AND anchor_occurrence_id IS NOT NULL AND anchor_render_digest IS NOT NULL
                 AND anchor_rotation_quarters IS NOT NULL
                 AND victor IS NOT NULL AND response_ms IS NOT NULL)
             OR
             (judgment_kind = 'favorite' AND anchor_asset_id IS NULL
                 AND anchor_occurrence_id IS NULL AND anchor_render_digest IS NULL
                 AND anchor_rotation_quarters IS NULL
                 AND victor IS NULL AND response_ms IS NULL)
         )
     ) STRICT, WITHOUT ROWID;";

pub struct Restoration {
    pub harvests: Vec<Harvest>,
    pub prepared: Vec<Prepared>,
    pub promotions: Vec<PromotionIntent>,
}

pub struct RemoteStore {
    connection: Connection,
    cache_root: PathBuf,
}

impl RemoteStore {
    pub fn open(database: &Path, cache_root: PathBuf) -> Result<Self> {
        if let Some(parent) = database.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create remote data directory at {}", parent.display()))?;
        }
        fs::create_dir_all(&cache_root)
            .with_context(|| format!("create remote cache at {}", cache_root.display()))?;
        let mut connection = Connection::open(database)
            .with_context(|| format!("open remote state at {}", database.display()))?;
        connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;",
        )?;
        migrate(&mut connection)?;
        let store = Self {
            connection,
            cache_root,
        };
        store.recover_transients()?;
        Ok(store)
    }

    pub fn absorb(&mut self, harvest: Harvest) -> Result<Harvest> {
        let now = now_ns()?;
        let tx = self.connection.transaction()?;
        tx.execute(
            "UPDATE pr_items SET state = 'retired', updated_at_ns = ?2
             WHERE source_id = ?1 AND state = 'discovered'",
            params![harvest.source_identity.as_str(), now],
        )?;
        let mut admitted = Vec::new();
        for discovery in harvest.discoveries {
            ensure!(
                discovery.source_identity == harvest.source_identity,
                "harvest mixed remote source identities"
            );
            tx.execute(
                "INSERT INTO pr_streams(source_id, stream_id, title, vetoed, updated_at_ns)
                 VALUES (?1, ?2, ?3, 0, ?4)
                 ON CONFLICT(source_id, stream_id) DO UPDATE SET
                    title = excluded.title, updated_at_ns = excluded.updated_at_ns",
                params![
                    discovery.source_identity.as_str(),
                    discovery.stream_id.as_str(),
                    discovery.stream_title,
                    now,
                ],
            )?;
            let vetoed = tx.query_row(
                "SELECT vetoed FROM pr_streams WHERE source_id = ?1 AND stream_id = ?2",
                params![
                    discovery.source_identity.as_str(),
                    discovery.stream_id.as_str()
                ],
                |row| row.get::<_, bool>(0),
            )?;
            upsert_discovery(&tx, &discovery, now)?;
            let state = tx.query_row(
                "SELECT state FROM pr_items WHERE item_id = ?1",
                [discovery.item_id.as_str()],
                |row| row.get::<_, String>(0),
            )?;
            if !vetoed && matches!(state.as_str(), "discovered" | "fetching" | "prepared") {
                admitted.push(discovery);
            }
        }
        prune_retired(&tx)?;
        tx.commit()?;
        Ok(Harvest {
            source: harvest.source,
            source_identity: harvest.source_identity,
            discoveries: admitted,
        })
    }

    pub fn begin_fetch(&self, item: &RemoteItemId) -> Result<()> {
        let changed = self.connection.execute(
            "UPDATE pr_items SET state = 'fetching', updated_at_ns = ?2, error = NULL
             WHERE item_id = ?1 AND state = 'discovered'",
            params![item.as_str(), now_ns()?],
        )?;
        ensure!(changed == 1, "remote candidate {item} is not discoverable");
        Ok(())
    }

    pub fn fetch_failed(&self, item: &RemoteItemId, error: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE pr_items SET state = 'discovered', updated_at_ns = ?2, error = ?3
             WHERE item_id = ?1 AND state = 'fetching'",
            params![item.as_str(), now_ns()?, error],
        )?;
        Ok(())
    }

    pub fn discard_fetch(&self, candidate: &Prepared) -> Result<()> {
        self.connection.execute(
            "UPDATE pr_items SET state = 'discovered', cache_path = NULL, payload_digest = NULL,
                 updated_at_ns = ?2, error = 'obsolete fetch discarded'
             WHERE item_id = ?1 AND state = 'fetching'",
            params![candidate.discovery.item_id.as_str(), now_ns()?],
        )?;
        remove_cache(&candidate.cache_path)
    }

    pub fn prepared(&self, candidate: &Prepared) -> Result<()> {
        let changed = self.connection.execute(
            "UPDATE pr_items SET state = 'prepared', cache_path = ?2, payload_digest = ?3,
                 updated_at_ns = ?4, error = NULL
             WHERE item_id = ?1 AND state = 'fetching'",
            params![
                candidate.discovery.item_id.as_str(),
                path_bytes(&candidate.cache_path),
                candidate.payload_digest,
                now_ns()?,
            ],
        )?;
        ensure!(
            changed == 1,
            "remote candidate {} left fetching",
            candidate.discovery.item_id
        );
        Ok(())
    }

    pub fn restore(&self, config: &RemoteConfig) -> Result<Restoration> {
        let sources = source_index(config);
        let mut discovered = HashMap::<SourceIx, Vec<Discovery>>::new();
        let mut prepared = Vec::new();
        let mut promotions = Vec::new();
        let rows = {
            let mut statement = self.connection.prepare(
                "SELECT i.item_id, i.source_id, i.stream_id, i.title, s.title,
                        i.origin_kind, i.origin, i.origin_seal, i.expected_md5, i.extension,
                        i.width, i.height, i.byte_len, i.max_pixels,
                        i.state, i.cache_path, i.payload_digest,
                        p.collection_root, p.rotation_quarters, p.judgment_kind,
                        p.session_id, p.anchor_asset_id, p.anchor_occurrence_id,
                        p.anchor_render_digest, p.anchor_rotation_quarters,
                        p.victor, p.command_id, p.response_ms
                 FROM pr_items i
                 JOIN pr_streams s ON s.source_id = i.source_id AND s.stream_id = i.stream_id
                 LEFT JOIN pr_promotions p ON p.item_id = i.item_id
                 WHERE i.state IN ('discovered', 'prepared')
                   AND (s.vetoed = 0 OR p.item_id IS NOT NULL)
                 ORDER BY i.updated_at_ns DESC",
            )?;
            statement
                .query_map([], |row| {
                    Ok(PersistedItem {
                        item_id: row.get(0)?,
                        source_id: row.get(1)?,
                        stream_id: row.get(2)?,
                        title: row.get(3)?,
                        stream_title: row.get(4)?,
                        origin_kind: row.get(5)?,
                        origin: row.get(6)?,
                        origin_seal: row.get(7)?,
                        expected_md5: row.get(8)?,
                        extension: row.get(9)?,
                        width: row.get(10)?,
                        height: row.get(11)?,
                        byte_len: row.get(12)?,
                        max_pixels: row.get(13)?,
                        state: row.get(14)?,
                        cache_path: row.get(15)?,
                        payload_digest: row.get(16)?,
                        promotion_root: row.get(17)?,
                        promotion_rotation: row.get(18)?,
                        promotion_kind: row.get(19)?,
                        promotion_session: row.get(20)?,
                        promotion_anchor: row.get(21)?,
                        promotion_anchor_occurrence: row.get(22)?,
                        promotion_anchor_render: row.get(23)?,
                        promotion_anchor_rotation: row.get(24)?,
                        promotion_victor: row.get(25)?,
                        promotion_command: row.get(26)?,
                        promotion_response_ms: row.get(27)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for row in rows {
            ensure!(
                row.promotion_kind.is_none() || row.state == "prepared",
                "persisted promotion is not prepared"
            );
            let configured = sources.get(&row.source_id);
            let discovery = if let Some((source, source_config)) = configured {
                row.discovery(
                    *source,
                    source_config.identity(),
                    source_config.identity().to_string(),
                    source_config.import_policy,
                )?
            } else if row.promotion_kind.is_some() {
                row.discovery(
                    SourceIx::new(config.sources.len()),
                    SourceIdentity::new(row.source_id.clone()),
                    row.source_id.clone(),
                    ImportPolicy::NotX,
                )?
            } else {
                retire_absent_source(&self.connection, &row)?;
                continue;
            };
            if row.state == "prepared"
                && let (Some(path), Some(digest)) =
                    (row.cache_path.clone(), row.payload_digest.clone())
            {
                let path = path_from_bytes(path);
                if cache_is_sound(&path, &digest)? {
                    let candidate = Prepared {
                        discovery,
                        cache_path: path,
                        payload_digest: digest,
                    };
                    if row.promotion_kind.is_some() {
                        promotions.push(row.promotion(candidate)?);
                    } else {
                        prepared.push(candidate);
                    }
                    continue;
                }
                self.connection.execute(
                    "DELETE FROM pr_promotions WHERE item_id = ?1",
                    [&row.item_id],
                )?;
                self.connection.execute(
                    "UPDATE pr_items SET state = 'discovered', cache_path = NULL,
                         payload_digest = NULL, error = 'prepared cache was lost or corrupt'
                     WHERE item_id = ?1",
                    [&row.item_id],
                )?;
            }
            if let Some((source, _)) = configured {
                discovered.entry(*source).or_default().push(discovery);
            } else {
                retire_absent_source(&self.connection, &row)?;
            }
        }
        prune_retired(&self.connection)?;
        let harvests = discovered
            .into_iter()
            .map(|(source, discoveries)| Harvest {
                source,
                source_identity: config.sources[source.get()].identity(),
                discoveries,
            })
            .collect();
        Ok(Restoration {
            harvests,
            prepared,
            promotions,
        })
    }

    pub fn begin_promotion(&self, intent: &PromotionIntent) -> Result<()> {
        ensure!(
            intent.collection_root.is_absolute(),
            "promotion root is not absolute"
        );
        ensure!(intent.rotation_quarters < 4, "invalid promotion rotation");
        let (
            kind,
            session,
            anchor_asset,
            anchor_occurrence,
            anchor_render,
            anchor_rotation,
            victor,
            command,
            response_ms,
        ) = match &intent.judgment {
            PromotionJudgment::Duel {
                session_id,
                anchor,
                victor,
                command_id,
                response_ms,
            } => (
                "duel",
                session_id,
                Some(anchor.asset_id.as_str()),
                Some(anchor.occurrence_id.get()),
                Some(anchor.render.as_str()),
                Some(anchor.rotation_quarters),
                Some(match victor {
                    DuelVictor::Anchor => "anchor",
                    DuelVictor::Challenger => "challenger",
                }),
                command_id,
                Some(i64::from(*response_ms)),
            ),
            PromotionJudgment::Favorite {
                session_id,
                command_id,
            } => (
                "favorite", session_id, None, None, None, None, None, command_id, None,
            ),
        };
        let now = now_ns()?;
        let tx = self.connection.unchecked_transaction()?;
        let state = tx.query_row(
            "SELECT state FROM pr_items WHERE item_id = ?1",
            [intent.candidate.discovery.item_id.as_str()],
            |row| row.get::<_, String>(0),
        )?;
        ensure!(state == "prepared", "remote candidate is not prepared");
        tx.execute(
            "INSERT INTO pr_promotions(
                 item_id, collection_root, rotation_quarters, judgment_kind, session_id,
                 anchor_asset_id, anchor_occurrence_id, anchor_render_digest,
                 anchor_rotation_quarters, victor, command_id, response_ms, committed_at_ns
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                intent.candidate.discovery.item_id.as_str(),
                path_bytes(&intent.collection_root),
                intent.rotation_quarters,
                kind,
                session.as_str(),
                anchor_asset,
                anchor_occurrence,
                anchor_render,
                anchor_rotation,
                victor,
                command.as_str(),
                response_ms,
                now,
            ],
        )?;
        if let PromotionJudgment::Duel { victor, .. } = &intent.judgment {
            insert_event(
                &tx,
                &intent.candidate.discovery.item_id,
                if *victor == DuelVictor::Challenger {
                    "remote_won"
                } else {
                    "local_won"
                },
                now,
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn abort_promotion(&self, item: &RemoteItemId) -> Result<()> {
        let tx = self.connection.unchecked_transaction()?;
        let changed = tx.execute(
            "DELETE FROM pr_promotions WHERE item_id = ?1",
            [item.as_str()],
        )?;
        ensure!(changed == 1, "remote promotion {item} is not durable");
        tx.execute(
            "DELETE FROM pr_events
             WHERE item_id = ?1 AND kind IN ('remote_won', 'local_won')",
            [item.as_str()],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn promotion_failed(&self, item: &RemoteItemId, message: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE pr_items SET error = ?2, updated_at_ns = ?3 WHERE item_id = ?1",
            params![item.as_str(), message, now_ns()?],
        )?;
        Ok(())
    }

    pub fn release(&self, candidate: &Prepared) -> Result<()> {
        self.connection.execute(
            "UPDATE pr_items SET state = 'discovered', cache_path = NULL, payload_digest = NULL,
                 updated_at_ns = ?2 WHERE item_id = ?1 AND state = 'prepared'",
            params![candidate.discovery.item_id.as_str(), now_ns()?],
        )?;
        remove_cache(&candidate.cache_path)
    }

    pub fn reject(&self, candidate: &Prepared) -> Result<()> {
        self.finish(candidate, "rejected", "rejected")
    }

    pub fn veto_stream(&self, candidate: &Prepared) -> Result<()> {
        let now = now_ns()?;
        let tx = self.connection.unchecked_transaction()?;
        let mut cache_paths = Vec::new();
        {
            let mut statement = tx.prepare(
                "SELECT cache_path FROM pr_items
                 WHERE source_id = ?1 AND stream_id = ?2 AND cache_path IS NOT NULL
                   AND NOT EXISTS(
                       SELECT 1 FROM pr_promotions p WHERE p.item_id = pr_items.item_id
                   )",
            )?;
            let rows = statement.query_map(
                params![
                    candidate.discovery.source_identity.as_str(),
                    candidate.discovery.stream_id.as_str(),
                ],
                |row| row.get::<_, Vec<u8>>(0),
            )?;
            for path in rows {
                cache_paths.push(path_from_bytes(path?));
            }
        }
        tx.execute(
            "UPDATE pr_streams SET vetoed = 1, updated_at_ns = ?3
             WHERE source_id = ?1 AND stream_id = ?2",
            params![
                candidate.discovery.source_identity.as_str(),
                candidate.discovery.stream_id.as_str(),
                now,
            ],
        )?;
        tx.execute(
            "UPDATE pr_items SET state = 'rejected', cache_path = NULL, payload_digest = NULL,
                 updated_at_ns = ?3
             WHERE source_id = ?1 AND stream_id = ?2 AND state NOT IN ('promoted', 'rejected')
               AND NOT EXISTS(
                   SELECT 1 FROM pr_promotions p WHERE p.item_id = pr_items.item_id
               )",
            params![
                candidate.discovery.source_identity.as_str(),
                candidate.discovery.stream_id.as_str(),
                now,
            ],
        )?;
        insert_event(&tx, &candidate.discovery.item_id, "stream_vetoed", now)?;
        tx.commit()?;
        for path in cache_paths {
            remove_cache(&path)?;
        }
        Ok(())
    }

    pub fn promoted(&self, candidate: &Prepared) -> Result<()> {
        self.finish(candidate, "promoted", "promoted")
    }

    pub fn note_duel(&self, item: &RemoteItemId, remote_won: bool) -> Result<()> {
        let tx = self.connection.unchecked_transaction()?;
        insert_event(
            &tx,
            item,
            if remote_won {
                "remote_won"
            } else {
                "local_won"
            },
            now_ns()?,
        )?;
        tx.commit()?;
        Ok(())
    }

    fn finish(&self, candidate: &Prepared, state: &str, event: &str) -> Result<()> {
        let now = now_ns()?;
        let tx = self.connection.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM pr_promotions WHERE item_id = ?1",
            [candidate.discovery.item_id.as_str()],
        )?;
        tx.execute(
            "UPDATE pr_items SET state = ?2, cache_path = NULL, payload_digest = NULL,
                 updated_at_ns = ?3 WHERE item_id = ?1",
            params![candidate.discovery.item_id.as_str(), state, now],
        )?;
        insert_event(&tx, &candidate.discovery.item_id, event, now)?;
        tx.commit()?;
        if let Err(error) = remove_cache(&candidate.cache_path) {
            eprintln!(
                "Picmash left retired remote cache {} for startup cleanup: {error:#}",
                candidate.cache_path.display()
            );
        }
        Ok(())
    }

    fn recover_transients(&self) -> Result<()> {
        self.connection.execute(
            "UPDATE pr_items SET state = 'discovered', error = 'interrupted fetch recovered'
             WHERE state = 'fetching'",
            [],
        )?;
        let mut referenced = HashSet::new();
        let mut statement = self
            .connection
            .prepare("SELECT cache_path FROM pr_items WHERE state = 'prepared'")?;
        for path in statement.query_map([], |row| row.get::<_, Vec<u8>>(0))? {
            let _inserted = referenced.insert(path_from_bytes(path?));
        }
        for entry in WalkDir::new(&self.cache_root)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            if !referenced.contains(entry.path()) {
                remove_cache(entry.path())?;
            }
        }
        Ok(())
    }
}

fn source_index(config: &RemoteConfig) -> HashMap<String, (SourceIx, &SourceConfig)> {
    config
        .sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.enabled())
        .map(|(index, source)| {
            (
                source.identity().to_string(),
                (SourceIx::new(index), source),
            )
        })
        .collect()
}

fn retire_absent_source(connection: &Connection, row: &PersistedItem) -> Result<()> {
    if let Some(path) = row.cache_path.clone() {
        remove_cache(&path_from_bytes(path))?;
    }
    connection.execute(
        "UPDATE pr_items SET state = 'retired', cache_path = NULL,
             payload_digest = NULL, error = 'source is disabled or absent'
         WHERE item_id = ?1",
        [&row.item_id],
    )?;
    Ok(())
}

fn prune_retired(connection: &Connection) -> Result<()> {
    connection.execute(
        "DELETE FROM pr_items
         WHERE state = 'retired'
           AND NOT EXISTS(SELECT 1 FROM pr_events WHERE item_id = pr_items.item_id)",
        [],
    )?;
    connection.execute(
        "DELETE FROM pr_streams
         WHERE vetoed = 0
           AND NOT EXISTS(
               SELECT 1 FROM pr_items
               WHERE source_id = pr_streams.source_id AND stream_id = pr_streams.stream_id
           )",
        [],
    )?;
    Ok(())
}

fn upsert_discovery(tx: &Transaction<'_>, discovery: &Discovery, now: i64) -> Result<()> {
    let (origin_kind, origin, origin_seal, expected_md5) = match &discovery.origin {
        Origin::Network { url, expected_md5 } => (
            "network",
            url.as_bytes().to_vec(),
            None,
            expected_md5.as_deref(),
        ),
        Origin::Local { path, stamp } => ("local", path_bytes(path), Some(stamp.as_bytes()), None),
    };
    tx.execute(
        "INSERT INTO pr_items(
             item_id, source_id, stream_id, title, origin_kind, origin, origin_seal, expected_md5,
             extension, width, height, byte_len, max_pixels, state, updated_at_ns
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 'discovered', ?14)
         ON CONFLICT(item_id) DO UPDATE SET
             title = excluded.title, origin_kind = excluded.origin_kind, origin = excluded.origin,
             origin_seal = excluded.origin_seal, expected_md5 = excluded.expected_md5,
             extension = excluded.extension,
             width = excluded.width, height = excluded.height, byte_len = excluded.byte_len,
             max_pixels = excluded.max_pixels,
             state = CASE WHEN pr_items.state = 'retired' THEN 'discovered' ELSE pr_items.state END,
             updated_at_ns = excluded.updated_at_ns
         WHERE pr_items.state IN ('discovered', 'fetching', 'prepared', 'retired')",
        params![
            discovery.item_id.as_str(),
            discovery.source_identity.as_str(),
            discovery.stream_id.as_str(),
            discovery.title,
            origin_kind,
            origin,
            origin_seal,
            expected_md5,
            discovery.extension,
            discovery.width,
            discovery.height,
            discovery.byte_len,
            discovery.max_pixels,
            now,
        ],
    )?;
    Ok(())
}

fn insert_event(tx: &Transaction<'_>, item: &RemoteItemId, kind: &str, now: i64) -> Result<()> {
    tx.execute(
        "INSERT INTO pr_events(item_id, kind, recorded_at_ns)
         SELECT ?1, ?2, ?3
         WHERE NOT EXISTS(
             SELECT 1 FROM pr_events WHERE item_id = ?1 AND kind = ?2
         )",
        params![item.as_str(), kind, now],
    )?;
    Ok(())
}

struct PersistedItem {
    item_id: String,
    source_id: String,
    stream_id: String,
    title: String,
    stream_title: String,
    origin_kind: String,
    origin: Vec<u8>,
    origin_seal: Option<Vec<u8>>,
    expected_md5: Option<String>,
    extension: String,
    width: u32,
    height: u32,
    byte_len: u64,
    max_pixels: u64,
    state: String,
    cache_path: Option<Vec<u8>>,
    payload_digest: Option<String>,
    promotion_root: Option<Vec<u8>>,
    promotion_rotation: Option<u8>,
    promotion_kind: Option<String>,
    promotion_session: Option<String>,
    promotion_anchor: Option<String>,
    promotion_anchor_occurrence: Option<i64>,
    promotion_anchor_render: Option<String>,
    promotion_anchor_rotation: Option<u8>,
    promotion_victor: Option<String>,
    promotion_command: Option<String>,
    promotion_response_ms: Option<u32>,
}

impl PersistedItem {
    fn discovery(
        &self,
        source: SourceIx,
        source_identity: SourceIdentity,
        source_name: String,
        import_policy: ImportPolicy,
    ) -> Result<Discovery> {
        let origin = match self.origin_kind.as_str() {
            "network" => Origin::Network {
                url: String::from_utf8(self.origin.clone())
                    .context("decode persisted remote URL")?,
                expected_md5: self.expected_md5.clone(),
            },
            "local" => Origin::Local {
                path: path_from_bytes(self.origin.clone()),
                stamp: FileStamp::parse(
                    self.origin_seal
                        .clone()
                        .context("persisted local source lacks a file stamp")?,
                )?,
            },
            invalid => anyhow::bail!("invalid persisted remote origin `{invalid}`"),
        };
        Ok(Discovery {
            source,
            source_identity,
            source_name,
            import_policy,
            stream_id: StreamId::new(self.stream_id.clone()),
            stream_title: self.stream_title.clone(),
            item_id: RemoteItemId::new(self.item_id.clone()),
            title: self.title.clone(),
            origin,
            extension: self.extension.clone(),
            width: self.width,
            height: self.height,
            byte_len: self.byte_len,
            max_pixels: self.max_pixels,
        })
    }

    fn promotion(&self, candidate: Prepared) -> Result<PromotionIntent> {
        let kind = self
            .promotion_kind
            .as_deref()
            .context("persisted promotion lacks a judgment kind")?;
        let session = picmash_engine::SessionId::parse(
            self.promotion_session
                .clone()
                .context("persisted promotion lacks a session")?,
        )?;
        let command = picmash_engine::CommandId::parse(
            self.promotion_command
                .clone()
                .context("persisted promotion lacks a command")?,
        )?;
        let judgment = match kind {
            "duel" => PromotionJudgment::Duel {
                session_id: session,
                anchor: picmash_engine::PresentedAsset::from_persisted(
                    picmash_engine::AssetId::parse(
                        self.promotion_anchor
                            .clone()
                            .context("persisted duel promotion lacks an anchor")?,
                    )?,
                    self.promotion_anchor_occurrence
                        .context("persisted duel promotion lacks an anchor occurrence")?,
                    self.promotion_anchor_render
                        .clone()
                        .context("persisted duel promotion lacks an anchor render")?,
                    self.promotion_anchor_rotation
                        .context("persisted duel promotion lacks an anchor rotation")?,
                )?,
                victor: match self
                    .promotion_victor
                    .as_deref()
                    .context("persisted duel promotion lacks a victor")?
                {
                    "anchor" => DuelVictor::Anchor,
                    "challenger" => DuelVictor::Challenger,
                    invalid => anyhow::bail!("invalid persisted promotion victor `{invalid}`"),
                },
                command_id: command,
                response_ms: self
                    .promotion_response_ms
                    .context("persisted duel promotion lacks response time")?,
            },
            "favorite" => PromotionJudgment::Favorite {
                session_id: session,
                command_id: command,
            },
            invalid => anyhow::bail!("invalid persisted promotion kind `{invalid}`"),
        };
        Ok(PromotionIntent {
            candidate,
            collection_root: path_from_bytes(
                self.promotion_root
                    .clone()
                    .context("persisted promotion lacks a collection root")?,
            ),
            rotation_quarters: self
                .promotion_rotation
                .context("persisted promotion lacks a rotation")?,
            judgment,
        })
    }
}

fn cache_is_sound(path: &Path, expected: &str) -> Result<bool> {
    match fs::read(path) {
        Ok(bytes) => Ok(blake3::hash(&bytes).to_hex().as_str() == expected),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("read remote cache at {}", path.display()))
        }
    }
}

fn remove_cache(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("remove remote cache at {}", path.display()))
        }
    }
}

fn now_ns() -> Result<i64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock precedes Unix epoch")?;
    i64::try_from(elapsed.as_nanos()).context("system clock exceeds remote database range")
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt as _;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().into_owned().into_bytes()
}

#[cfg(unix)]
fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt as _};
    PathBuf::from(OsString::from_vec(bytes))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(&bytes).into_owned())
}

fn migrate(connection: &mut Connection) -> Result<()> {
    let tx = connection.transaction()?;
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS pr_schema(
             version INTEGER PRIMARY KEY,
             applied_at_ns INTEGER NOT NULL
         ) STRICT;",
    )?;
    let current = tx.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM pr_schema",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    ensure!(
        current <= SCHEMA,
        "remote database schema {current} is newer than supported {SCHEMA}"
    );
    if current == 0 {
        tx.execute_batch(
            "CREATE TABLE pr_streams(
                 source_id TEXT NOT NULL,
                 stream_id TEXT NOT NULL,
                 title TEXT NOT NULL,
                 vetoed INTEGER NOT NULL CHECK(vetoed IN (0, 1)),
                 updated_at_ns INTEGER NOT NULL,
                 PRIMARY KEY(source_id, stream_id)
             ) STRICT, WITHOUT ROWID;

             CREATE TABLE pr_items(
                 item_id TEXT PRIMARY KEY,
                 source_id TEXT NOT NULL,
                 stream_id TEXT NOT NULL,
                 title TEXT NOT NULL,
                 origin_kind TEXT NOT NULL CHECK(origin_kind IN ('network', 'local')),
                 origin BLOB NOT NULL,
                 origin_seal BLOB,
                 expected_md5 TEXT,
                 extension TEXT NOT NULL,
                 width INTEGER NOT NULL CHECK(width > 0),
                 height INTEGER NOT NULL CHECK(height > 0),
                 byte_len INTEGER NOT NULL CHECK(byte_len > 0),
                 max_pixels INTEGER NOT NULL CHECK(max_pixels BETWEEN 1000000 AND 64000000),
                 state TEXT NOT NULL CHECK(state IN (
                     'discovered', 'fetching', 'prepared', 'rejected', 'promoted', 'retired'
                 )),
                 cache_path BLOB,
                 payload_digest TEXT,
                 error TEXT,
                 updated_at_ns INTEGER NOT NULL,
                 FOREIGN KEY(source_id, stream_id) REFERENCES pr_streams(source_id, stream_id),
                 CHECK((origin_kind = 'local') = (origin_seal IS NOT NULL)),
                 CHECK((state = 'prepared') = (cache_path IS NOT NULL AND payload_digest IS NOT NULL))
             ) STRICT, WITHOUT ROWID;

             CREATE INDEX pr_items_frontier ON pr_items(source_id, state, updated_at_ns);

             CREATE TABLE pr_events(
                 id INTEGER PRIMARY KEY,
                 item_id TEXT NOT NULL REFERENCES pr_items(item_id),
                 kind TEXT NOT NULL CHECK(kind IN (
                     'rejected', 'stream_vetoed', 'promoted', 'remote_won', 'local_won'
                 )),
                 recorded_at_ns INTEGER NOT NULL
             ) STRICT;",
        )?;
        tx.execute_batch(PROMOTION_SCHEMA)?;
        tx.execute(
            "INSERT INTO pr_schema(version, applied_at_ns) VALUES (?1, ?2)",
            params![SCHEMA, now_ns()?],
        )?;
    }
    if current == 1 {
        tx.execute_batch(
            "ALTER TABLE pr_items
             ADD COLUMN max_pixels INTEGER NOT NULL DEFAULT 40000000
             CHECK(max_pixels BETWEEN 1000000 AND 64000000);",
        )?;
        tx.execute(
            "INSERT INTO pr_schema(version, applied_at_ns) VALUES (2, ?1)",
            [now_ns()?],
        )?;
    }
    if (1..=2).contains(&current) {
        tx.execute_batch(
            "ALTER TABLE pr_items ADD COLUMN origin_seal BLOB;
             UPDATE pr_items
             SET state = 'retired', cache_path = NULL, payload_digest = NULL,
                 error = 'legacy local snapshot retired during file-stamp migration'
             WHERE origin_kind = 'local' AND state IN ('discovered', 'fetching', 'prepared');",
        )?;
        tx.execute(
            "INSERT INTO pr_schema(version, applied_at_ns) VALUES (3, ?1)",
            [now_ns()?],
        )?;
    }
    if (1..=3).contains(&current) {
        tx.execute_batch(PROMOTION_SCHEMA)?;
        tx.execute(
            "INSERT INTO pr_schema(version, applied_at_ns) VALUES (4, ?1)",
            [now_ns()?],
        )?;
    }
    tx.commit()?;
    Ok(())
}
