use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OpenFlags, OptionalExtension, Transaction, params};

use crate::{
    Engine,
    catalog::encode_path,
    engine::now_ns,
    fault::{Fault, Result},
    ids::{AssetId, CollectionId},
    media::RenderDigest,
    model::LegacyImportReport,
};

#[derive(Debug)]
struct LegacyDump {
    corpora: Vec<LegacyCorpus>,
    assets: Vec<LegacyAsset>,
    occurrences: Vec<LegacyOccurrence>,
    sessions: Vec<LegacySession>,
    events: Vec<LegacyEvent>,
    external_assets: BTreeSet<String>,
}

#[derive(Debug)]
struct LegacyCorpus {
    id: i64,
    root: PathBuf,
    created_at: i64,
}

#[derive(Debug)]
struct LegacyAsset {
    id: String,
    created_at: i64,
    render_hash: Option<String>,
}

#[derive(Debug)]
struct LegacyOccurrence {
    corpus_id: i64,
    path: PathBuf,
    asset_id: String,
    blob_id: Option<String>,
    width: i64,
    height: i64,
    byte_len: i64,
    hidden: bool,
    rotation_quarters: i64,
}

#[derive(Debug)]
struct LegacySession {
    id: i64,
    corpus_id: i64,
    started_at: i64,
    ended_at: Option<i64>,
}

#[derive(Debug)]
struct LegacyEvent {
    created_at: i64,
    local_id: i64,
    payload: LegacyPayload,
}

#[derive(Debug)]
enum LegacyPayload {
    Duel {
        session_id: i64,
        left: String,
        right: String,
        winner: String,
    },
    Threshold {
        session_id: i64,
        asset: String,
        admit: bool,
    },
    Favorite {
        session_id: i64,
        asset: String,
        active: bool,
    },
    Similarity {
        corpus_id: i64,
        representation: String,
        assets: [String; 3],
        chosen_pair: String,
    },
}

impl LegacyPayload {
    const fn rank(&self) -> u8 {
        match self {
            Self::Duel { .. } => 0,
            Self::Threshold { .. } => 1,
            Self::Favorite { .. } => 2,
            Self::Similarity { .. } => 3,
        }
    }
}

impl Engine {
    /// Imports the final web-app schema through a read-only boundary.
    ///
    /// Learned scores, unversioned embeddings, external-source interactions,
    /// and face state are deliberately not promoted into engine authority.
    pub fn import_legacy(&self, source: impl AsRef<Path>) -> Result<LegacyImportReport> {
        let source = source
            .as_ref()
            .canonicalize()
            .map_err(|source_error| Fault::Filesystem {
                path: source.as_ref().to_path_buf(),
                source: source_error,
            })?;
        let legacy = Connection::open_with_flags(&source, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let dump = LegacyDump::load(&legacy)?;
        let fingerprint = format!(
            "legacy-v1:{}",
            blake3::hash(format!("{dump:?}").as_bytes()).to_hex()
        );

        let mut connection = self.connection.lock();
        if let Some(report) = prior_report(&connection, &fingerprint)? {
            return Ok(report);
        }
        let tx = connection.transaction()?;
        let now = now_ns()?;
        let collections = import_collections(&tx, &dump.corpora)?;
        let assets = import_assets(&tx, &dump.assets)?;
        import_occurrences(&tx, &dump.occurrences, &collections, &assets, now)?;
        import_sessions(&tx, &dump.sessions, &collections, &fingerprint)?;
        let (imported_observations, ambiguous_observations) =
            import_events(&tx, &dump, &collections, &assets, &fingerprint, now)?;
        let imported_assets = assets.values().collect::<BTreeSet<_>>().len();
        tx.execute(
            "INSERT INTO pm_legacy_imports(
                 source_fingerprint, source_path, imported_at_ns,
                 imported_assets, imported_observations, ambiguous_observations
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                fingerprint,
                encode_path(&source),
                now,
                imported_assets,
                imported_observations,
                ambiguous_observations,
            ],
        )?;
        tx.commit()?;
        Ok(LegacyImportReport {
            source_fingerprint: fingerprint,
            imported_assets,
            imported_observations,
            ambiguous_observations,
            already_imported: false,
        })
    }
}

impl LegacyDump {
    fn load(connection: &Connection) -> Result<Self> {
        for table in [
            "corpora",
            "assets",
            "corpus_assets",
            "sessions",
            "comparisons",
            "nudge_events",
            "heart_events",
            "similarity_triads",
        ] {
            if !table_exists(connection, table)? {
                return Err(Fault::UnsupportedLegacy(format!(
                    "required table {table} is absent"
                )));
            }
        }
        let corpora = load_rows(
            connection,
            "SELECT id, root_path, created_at FROM corpora ORDER BY id",
            |row| {
                Ok(LegacyCorpus {
                    id: row.get(0)?,
                    root: PathBuf::from(row.get::<_, String>(1)?),
                    created_at: row.get(2)?,
                })
            },
        )?;
        let assets = load_rows(
            connection,
            "SELECT id, created_at, render_hash FROM assets ORDER BY id",
            |row| {
                Ok(LegacyAsset {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    render_hash: row.get(2)?,
                })
            },
        )?;
        let occurrences = load_rows(
            connection,
            "SELECT ca.corpus_id, ca.path, ca.asset_id, ca.blob_id,
                    ca.blob_width, ca.blob_height, ca.blob_bytes, ca.hidden,
                    a.rotation_quarters
             FROM corpus_assets ca JOIN assets a ON a.id = ca.asset_id
             ORDER BY ca.corpus_id, ca.path",
            |row| {
                Ok(LegacyOccurrence {
                    corpus_id: row.get(0)?,
                    path: PathBuf::from(row.get::<_, String>(1)?),
                    asset_id: row.get(2)?,
                    blob_id: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    byte_len: row.get(6)?,
                    hidden: row.get(7)?,
                    rotation_quarters: row.get(8)?,
                })
            },
        )?;
        let sessions = load_rows(
            connection,
            "SELECT id, corpus_id, started_at, ended_at FROM sessions ORDER BY id",
            |row| {
                Ok(LegacySession {
                    id: row.get(0)?,
                    corpus_id: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                })
            },
        )?;
        let mut events = Vec::new();
        events.extend(load_rows(
            connection,
            "SELECT id, session_id, left_asset_id, right_asset_id,
                    winner_asset_id, created_at FROM comparisons",
            |row| {
                Ok(LegacyEvent {
                    local_id: row.get(0)?,
                    created_at: row.get(5)?,
                    payload: LegacyPayload::Duel {
                        session_id: row.get(1)?,
                        left: row.get(2)?,
                        right: row.get(3)?,
                        winner: row.get(4)?,
                    },
                })
            },
        )?);
        events.extend(load_rows(
            connection,
            "SELECT id, session_id, asset_id, direction, created_at FROM nudge_events",
            |row| {
                Ok(LegacyEvent {
                    local_id: row.get(0)?,
                    created_at: row.get(4)?,
                    payload: LegacyPayload::Threshold {
                        session_id: row.get(1)?,
                        asset: row.get(2)?,
                        admit: row.get::<_, f64>(3)? > 0.0,
                    },
                })
            },
        )?);
        events.extend(load_rows(
            connection,
            "SELECT id, session_id, asset_id, active, created_at FROM heart_events",
            |row| {
                Ok(LegacyEvent {
                    local_id: row.get(0)?,
                    created_at: row.get(4)?,
                    payload: LegacyPayload::Favorite {
                        session_id: row.get(1)?,
                        asset: row.get(2)?,
                        active: row.get(3)?,
                    },
                })
            },
        )?);
        events.extend(load_rows(
            connection,
            "SELECT id, corpus_id, model_name, asset_a_id, asset_b_id,
                    asset_c_id, chosen_pair, created_at FROM similarity_triads",
            |row| {
                Ok(LegacyEvent {
                    local_id: row.get(0)?,
                    created_at: row.get(7)?,
                    payload: LegacyPayload::Similarity {
                        corpus_id: row.get(1)?,
                        representation: row.get(2)?,
                        assets: [row.get(3)?, row.get(4)?, row.get(5)?],
                        chosen_pair: row.get(6)?,
                    },
                })
            },
        )?);
        events.sort_by_key(|event| (event.created_at, event.payload.rank(), event.local_id));
        let external_assets = if table_exists(connection, "asset_external_provenance")? {
            load_rows(
                connection,
                "SELECT DISTINCT asset_id FROM asset_external_provenance ORDER BY asset_id",
                |row| row.get(0),
            )?
            .into_iter()
            .collect()
        } else {
            BTreeSet::new()
        };
        Ok(Self {
            corpora,
            assets,
            occurrences,
            sessions,
            events,
            external_assets,
        })
    }
}

fn import_collections(
    tx: &Transaction<'_>,
    corpora: &[LegacyCorpus],
) -> Result<BTreeMap<i64, CollectionId>> {
    let mut mapping = BTreeMap::new();
    for corpus in corpora {
        let canonical_root = corpus
            .root
            .canonicalize()
            .unwrap_or_else(|_| corpus.root.clone());
        let root = encode_path(&canonical_root);
        tx.execute(
            "INSERT INTO pm_collections(root, scan_generation, catalog_revision, created_at_ns)
             VALUES (?1, 1, 1, ?2) ON CONFLICT(root) DO NOTHING",
            params![root, seconds_to_ns(corpus.created_at)],
        )?;
        let id = tx.query_row(
            "SELECT id FROM pm_collections WHERE root = ?1",
            [root],
            |row| row.get::<_, i64>(0),
        )?;
        mapping.insert(corpus.id, CollectionId::from_raw(id));
    }
    Ok(mapping)
}

fn import_assets(
    tx: &Transaction<'_>,
    assets: &[LegacyAsset],
) -> Result<BTreeMap<String, AssetId>> {
    let mut mapping = BTreeMap::new();
    for asset in assets {
        let (render, authority) = legacy_render(asset)?;
        let existing = tx
            .query_row(
                "SELECT id FROM pm_assets WHERE render_digest = ?1",
                [render.as_str()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let id = if let Some(id) = existing {
            AssetId::parse(id)?
        } else {
            let preferred = AssetId::parse(asset.id.clone())?;
            let occupied = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM pm_assets WHERE id = ?1)",
                [preferred.as_str()],
                |row| row.get::<_, bool>(0),
            )?;
            let id = if occupied {
                AssetId::fresh()
            } else {
                preferred
            };
            tx.execute(
                "INSERT INTO pm_assets(id, render_digest, identity_authority, created_at_ns)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    id.as_str(),
                    render.as_str(),
                    authority,
                    seconds_to_ns(asset.created_at)
                ],
            )?;
            id
        };
        mapping.insert(asset.id.clone(), id);
    }
    Ok(mapping)
}

fn import_occurrences(
    tx: &Transaction<'_>,
    occurrences: &[LegacyOccurrence],
    collections: &BTreeMap<i64, CollectionId>,
    assets: &BTreeMap<String, AssetId>,
    now: i64,
) -> Result<()> {
    for occurrence in occurrences {
        let Some(collection) = collections.get(&occurrence.corpus_id) else {
            continue;
        };
        let Some(asset) = assets.get(&occurrence.asset_id) else {
            continue;
        };
        let blob = occurrence.blob_id.as_deref().map_or_else(
            || {
                format!(
                    "legacy:missing-blob:{}.{}",
                    occurrence.corpus_id, occurrence.asset_id
                )
            },
            |blob| {
                if blob.len() == 64 && blob.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    format!("blake3:{blob}")
                } else {
                    format!("legacy:{blob}")
                }
            },
        );
        tx.execute(
            "INSERT INTO pm_occurrences(
                 collection_id, path, asset_id, blob_digest, width, height, byte_len,
                 rotation_quarters, present, last_seen_generation
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, 1)
             ON CONFLICT(collection_id, path) DO NOTHING",
            params![
                collection.get(),
                encode_path(&occurrence.path),
                asset.as_str(),
                blob,
                occurrence.width.max(0),
                occurrence.height.max(0),
                occurrence.byte_len.max(0),
                occurrence.rotation_quarters.rem_euclid(4),
            ],
        )?;
        tx.execute(
            "INSERT INTO pm_collection_assets(
                 collection_id, asset_id, hidden, favorite, updated_at_ns
             ) VALUES (?1, ?2, ?3, 0, ?4)
             ON CONFLICT(collection_id, asset_id) DO UPDATE SET
                 hidden = MAX(hidden, excluded.hidden)",
            params![collection.get(), asset.as_str(), occurrence.hidden, now],
        )?;
    }
    Ok(())
}

fn import_sessions(
    tx: &Transaction<'_>,
    sessions: &[LegacySession],
    collections: &BTreeMap<i64, CollectionId>,
    namespace: &str,
) -> Result<()> {
    for session in sessions {
        let Some(collection) = collections.get(&session.corpus_id) else {
            continue;
        };
        tx.execute(
            "INSERT INTO pm_sessions(
                 id, collection_id, context_revision, started_at_ns, ended_at_ns
             ) VALUES (?1, ?2, 'legacy-web-v0', ?3, ?4) ON CONFLICT(id) DO NOTHING",
            params![
                legacy_session(namespace, session.id),
                collection.get(),
                seconds_to_ns(session.started_at),
                session.ended_at.map(seconds_to_ns),
            ],
        )?;
    }
    Ok(())
}

fn import_events(
    tx: &Transaction<'_>,
    dump: &LegacyDump,
    collections: &BTreeMap<i64, CollectionId>,
    assets: &BTreeMap<String, AssetId>,
    namespace: &str,
    now: i64,
) -> Result<(usize, usize)> {
    let sessions = dump
        .sessions
        .iter()
        .map(|session| (session.id, session.corpus_id))
        .collect::<BTreeMap<_, _>>();
    let mut imported = 0;
    let mut ambiguous = 0;
    for event in &dump.events {
        let outcome = match &event.payload {
            LegacyPayload::Duel {
                session_id,
                left,
                right,
                winner,
            } => {
                if dump.external_assets.contains(left)
                    || dump.external_assets.contains(right)
                    || !sessions.contains_key(session_id)
                {
                    false
                } else {
                    import_duel(
                        tx,
                        event,
                        *session_id,
                        left,
                        right,
                        winner,
                        assets,
                        namespace,
                    )?
                }
            }
            LegacyPayload::Threshold {
                session_id,
                asset,
                admit,
            } => {
                if sessions.contains_key(session_id) {
                    import_threshold(tx, event, *session_id, asset, *admit, assets, namespace)?
                } else {
                    false
                }
            }
            LegacyPayload::Favorite {
                session_id,
                asset,
                active,
            } => import_favorite(
                tx,
                event,
                *session_id,
                asset,
                *active,
                &sessions,
                collections,
                assets,
                namespace,
                now,
            )?,
            LegacyPayload::Similarity {
                corpus_id,
                representation,
                assets: triad,
                chosen_pair,
            } => import_similarity(
                tx,
                event,
                *corpus_id,
                representation,
                triad,
                chosen_pair,
                collections,
                assets,
                namespace,
            )?,
        };
        if outcome {
            imported += 1;
        } else {
            ambiguous += 1;
        }
    }
    Ok((imported, ambiguous))
}

fn import_duel(
    tx: &Transaction<'_>,
    event: &LegacyEvent,
    session_id: i64,
    old_left: &str,
    old_right: &str,
    old_winner: &str,
    assets: &BTreeMap<String, AssetId>,
    namespace: &str,
) -> Result<bool> {
    let (Some(left), Some(right), Some(winner)) = (
        assets.get(old_left),
        assets.get(old_right),
        assets.get(old_winner),
    ) else {
        return Ok(false);
    };
    if left == right || (winner != left && winner != right) {
        return Ok(false);
    }
    let command = format!("{namespace}:comparison:{}", event.local_id);
    let Some(observation) = insert_legacy_envelope(
        tx,
        &command,
        &legacy_session(namespace, session_id),
        event.created_at,
        "legacy-asset-duel-v0",
        None,
        "asset_duel",
    )?
    else {
        return Ok(true);
    };
    let (left_occurrence, left_render, left_rotation) = presentation(tx, left)?;
    let (right_occurrence, right_render, right_rotation) = presentation(tx, right)?;
    tx.execute(
        "INSERT INTO pm_asset_duels(
             observation_id, left_asset_id, right_asset_id, winner_asset_id,
             left_occurrence_id, right_occurrence_id, left_render_digest, right_render_digest,
             left_rotation_quarters, right_rotation_quarters
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            observation,
            left.as_str(),
            right.as_str(),
            winner.as_str(),
            left_occurrence,
            right_occurrence,
            left_render.as_str(),
            right_render.as_str(),
            left_rotation,
            right_rotation,
        ],
    )?;
    Ok(true)
}

fn import_threshold(
    tx: &Transaction<'_>,
    event: &LegacyEvent,
    session_id: i64,
    old_asset: &str,
    admit: bool,
    assets: &BTreeMap<String, AssetId>,
    namespace: &str,
) -> Result<bool> {
    let Some(asset) = assets.get(old_asset) else {
        return Ok(false);
    };
    let command = format!("{namespace}:nudge:{}", event.local_id);
    let Some(observation) = insert_legacy_envelope(
        tx,
        &command,
        &legacy_session(namespace, session_id),
        event.created_at,
        "legacy-nudge-v0",
        None,
        "asset_threshold",
    )?
    else {
        return Ok(true);
    };
    let (occurrence, render, rotation) = presentation(tx, asset)?;
    tx.execute(
        "INSERT INTO pm_asset_thresholds(
             observation_id, asset_id, occurrence_id, render_digest,
             rotation_quarters, judgment
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            observation,
            asset.as_str(),
            occurrence,
            render.as_str(),
            rotation,
            if admit { "admit" } else { "reject" },
        ],
    )?;
    Ok(true)
}

fn import_favorite(
    tx: &Transaction<'_>,
    event: &LegacyEvent,
    session_id: i64,
    old_asset: &str,
    active: bool,
    sessions: &BTreeMap<i64, i64>,
    collections: &BTreeMap<i64, CollectionId>,
    assets: &BTreeMap<String, AssetId>,
    namespace: &str,
    now: i64,
) -> Result<bool> {
    let (Some(corpus), Some(asset)) = (sessions.get(&session_id), assets.get(old_asset)) else {
        return Ok(false);
    };
    let Some(collection) = collections.get(corpus) else {
        return Ok(false);
    };
    let exists = tx.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM pm_collection_assets WHERE collection_id = ?1 AND asset_id = ?2
         )",
        params![collection.get(), asset.as_str()],
        |row| row.get::<_, bool>(0),
    )?;
    if !exists {
        return Ok(false);
    }
    let command = format!("{namespace}:heart:{}", event.local_id);
    let Some(observation) = insert_legacy_envelope(
        tx,
        &command,
        &legacy_session(namespace, session_id),
        event.created_at,
        "legacy-heart-v0",
        None,
        "favorite_set",
    )?
    else {
        return Ok(true);
    };
    let changed = tx.execute(
        "UPDATE pm_collection_assets SET favorite = ?3, updated_at_ns = ?4
         WHERE collection_id = ?1 AND asset_id = ?2",
        params![collection.get(), asset.as_str(), active, now],
    )?;
    if changed == 0 {
        return Ok(false);
    }
    tx.execute(
        "INSERT INTO pm_favorite_events(observation_id, collection_id, asset_id, active)
         VALUES (?1, ?2, ?3, ?4)",
        params![observation, collection.get(), asset.as_str(), active],
    )?;
    Ok(true)
}

fn import_similarity(
    tx: &Transaction<'_>,
    event: &LegacyEvent,
    corpus_id: i64,
    representation: &str,
    old_assets: &[String; 3],
    chosen_pair: &str,
    collections: &BTreeMap<i64, CollectionId>,
    assets: &BTreeMap<String, AssetId>,
    namespace: &str,
) -> Result<bool> {
    let Some(collection) = collections.get(&corpus_id) else {
        return Ok(false);
    };
    let Some(mapped) = old_assets
        .iter()
        .map(|asset| assets.get(asset).cloned())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(false);
    };
    if mapped.iter().collect::<BTreeSet<_>>().len() != 3 {
        return Ok(false);
    }
    let nearest = match chosen_pair {
        "ab" => [&mapped[0], &mapped[1]],
        "ac" => [&mapped[0], &mapped[2]],
        "bc" => [&mapped[1], &mapped[2]],
        _ => return Ok(false),
    };
    let nearest = if nearest[0] < nearest[1] {
        nearest
    } else {
        [nearest[1], nearest[0]]
    };
    let session = format!("{namespace}:similarity-session:{corpus_id}");
    tx.execute(
        "INSERT INTO pm_sessions(id, collection_id, context_revision, started_at_ns)
         VALUES (?1, ?2, 'legacy-similarity-v0', ?3) ON CONFLICT(id) DO NOTHING",
        params![session, collection.get(), seconds_to_ns(event.created_at)],
    )?;
    let command = format!("{namespace}:similarity:{}", event.local_id);
    let Some(observation) = insert_legacy_envelope(
        tx,
        &command,
        &session,
        event.created_at,
        "legacy-similarity-triad-v0",
        Some(representation),
        "similarity_triad",
    )?
    else {
        return Ok(true);
    };
    let presentations = mapped
        .iter()
        .map(|asset| presentation(tx, asset))
        .collect::<Result<Vec<_>>>()?;
    tx.execute(
        "INSERT INTO pm_similarity_triads(
             observation_id, a_asset_id, b_asset_id, c_asset_id,
             nearest_low_asset_id, nearest_high_asset_id,
             a_occurrence_id, b_occurrence_id, c_occurrence_id,
             a_render_digest, b_render_digest, c_render_digest,
             a_rotation_quarters, b_rotation_quarters, c_rotation_quarters
         ) VALUES (
             ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15
         )",
        params![
            observation,
            mapped[0].as_str(),
            mapped[1].as_str(),
            mapped[2].as_str(),
            nearest[0].as_str(),
            nearest[1].as_str(),
            presentations[0].0,
            presentations[1].0,
            presentations[2].0,
            presentations[0].1.as_str(),
            presentations[1].1.as_str(),
            presentations[2].1.as_str(),
            presentations[0].2,
            presentations[1].2,
            presentations[2].2,
        ],
    )?;
    Ok(true)
}

fn insert_legacy_envelope(
    tx: &Transaction<'_>,
    command: &str,
    session: &str,
    created_at: i64,
    prompt_revision: &str,
    representation_revision: Option<&str>,
    kind: &str,
) -> Result<Option<i64>> {
    let inserted = tx.execute(
        "INSERT INTO pm_observations(
             command_id, payload_digest, session_id, recorded_at_ns, prompt_revision,
             representation_revision, ordering_authority, legacy_source, kind
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'legacy_inferred', ?1, ?7)
         ON CONFLICT(command_id) DO NOTHING",
        params![
            command,
            format!(
                "legacy-command-v1:{}",
                blake3::hash(command.as_bytes()).to_hex()
            ),
            session,
            seconds_to_ns(created_at),
            prompt_revision,
            representation_revision,
            kind,
        ],
    )?;
    Ok((inserted == 1).then(|| tx.last_insert_rowid()))
}

fn presentation(tx: &Transaction<'_>, asset: &AssetId) -> Result<(Option<i64>, RenderDigest, i64)> {
    let render = tx.query_row(
        "SELECT render_digest FROM pm_assets WHERE id = ?1",
        [asset.as_str()],
        |row| row.get::<_, String>(0),
    )?;
    let occurrence = tx
        .query_row(
            "SELECT id, rotation_quarters FROM pm_occurrences WHERE asset_id = ?1
             ORDER BY width * height DESC, byte_len DESC, path ASC LIMIT 1",
            [asset.as_str()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?;
    let (occurrence, rotation) =
        occurrence.map_or((None, 0), |(id, rotation)| (Some(id), rotation));
    Ok((occurrence, RenderDigest::parse(render)?, rotation))
}

fn legacy_render(asset: &LegacyAsset) -> Result<(RenderDigest, &'static str)> {
    let exact = asset
        .render_hash
        .as_deref()
        .and_then(|hash| hash.strip_prefix("render:v1:"))
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if let Some(digest) = exact {
        Ok((RenderDigest::parse(format!("rgba-v1:{digest}"))?, "exact"))
    } else {
        Ok((RenderDigest::legacy(&asset.id), "legacy"))
    }
}

fn legacy_session(namespace: &str, id: i64) -> String {
    format!("{namespace}:session:{id}")
}

fn seconds_to_ns(seconds: i64) -> i64 {
    seconds.saturating_mul(1_000_000_000)
}

fn table_exists(connection: &Connection, table: &str) -> Result<bool> {
    Ok(connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [table],
        |row| row.get(0),
    )?)
}

fn load_rows<T>(
    connection: &Connection,
    sql: &str,
    map: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
) -> Result<Vec<T>> {
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map([], map)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Fault::from)
}

fn prior_report(connection: &Connection, fingerprint: &str) -> Result<Option<LegacyImportReport>> {
    connection
        .query_row(
            "SELECT imported_assets, imported_observations, ambiguous_observations
             FROM pm_legacy_imports WHERE source_fingerprint = ?1",
            [fingerprint],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()?
        .map(|(assets, observations, ambiguous)| {
            Ok(LegacyImportReport {
                source_fingerprint: fingerprint.to_owned(),
                imported_assets: usize::try_from(assets)
                    .map_err(|_| Fault::Corrupt("invalid legacy asset count".to_owned()))?,
                imported_observations: usize::try_from(observations)
                    .map_err(|_| Fault::Corrupt("invalid legacy observation count".to_owned()))?,
                ambiguous_observations: usize::try_from(ambiguous)
                    .map_err(|_| Fault::Corrupt("invalid ambiguous count".to_owned()))?,
                already_imported: true,
            })
        })
        .transpose()
}
