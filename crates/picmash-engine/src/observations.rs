use std::collections::HashSet;

use rusqlite::{OptionalExtension, Transaction, params};

use crate::{
    Engine,
    engine::now_ns,
    fault::{Fault, Result},
    ids::{AssetId, CollectionId, CommandId, ObservationId, PromptId, SessionId, SnapshotId},
    media::RenderDigest,
    model::{ComparisonPrompt, JudgmentSession, PresentedAsset, ThresholdJudgment},
};

const PAIR_POLICY: &str = "coverage-anchor-matched-balanced-v2";
const REMOTE_ADMISSION_POLICY: &str = "remote-admission-v1";

impl Engine {
    pub fn start_session(
        &self,
        collection_id: CollectionId,
        context_revision: impl Into<String>,
    ) -> Result<JudgmentSession> {
        let context_revision = nonempty(context_revision.into(), "context revision")?;
        let now = now_ns()?;
        let id = SessionId::fresh();
        let connection = self.connection.lock();
        connection.execute(
            "INSERT INTO pm_sessions(id, collection_id, context_revision, started_at_ns)
             VALUES (?1, ?2, ?3, ?4)",
            params![id.as_str(), collection_id.get(), context_revision, now],
        )?;
        Ok(JudgmentSession {
            id,
            collection_id,
            context_revision,
            started_at_ns: now,
            ended_at_ns: None,
        })
    }

    pub fn end_session(&self, session_id: &SessionId) -> Result<()> {
        let connection = self.connection.lock();
        let changed = connection.execute(
            "UPDATE pm_sessions SET ended_at_ns = ?2 WHERE id = ?1 AND ended_at_ns IS NULL",
            params![session_id.as_str(), now_ns()?],
        )?;
        if changed == 0 {
            let exists = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM pm_sessions WHERE id = ?1)",
                [session_id.as_str()],
                |row| row.get::<_, bool>(0),
            )?;
            if !exists {
                return Err(Fault::InvalidInput(format!(
                    "unknown judgment session {session_id}"
                )));
            }
        }
        Ok(())
    }

    pub fn propose_comparison(&self, session_id: &SessionId) -> Result<ComparisonPrompt> {
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let collection_id = active_session_collection(&tx, session_id)?;
        let pair = tx
            .query_row(
                r"
                WITH representatives AS (
                    SELECT o.asset_id, o.id AS occurrence_id, a.render_digest,
                           o.rotation_quarters,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.asset_id
                               ORDER BY o.width * o.height DESC, o.byte_len DESC, o.path ASC
                           ) AS rank
                    FROM pm_occurrences o
                    JOIN pm_assets a ON a.id = o.asset_id
                    JOIN pm_collection_assets ca
                      ON ca.collection_id = o.collection_id AND ca.asset_id = o.asset_id
                    WHERE o.collection_id = ?1 AND o.present = 1 AND ca.hidden = 0
                ), latest_snapshot AS (
                    SELECT id FROM pm_preference_snapshots
                    WHERE collection_id = ?1 ORDER BY id DESC LIMIT 1
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
                ), visible AS (
                    SELECT r.asset_id, r.occurrence_id, r.render_digest, r.rotation_quarters,
                           COALESCE(ps.score, 0.0) AS score,
                           COALESCE(dc.duel_count, 0) AS duel_count
                    FROM representatives r
                    LEFT JOIN latest_snapshot ls
                    LEFT JOIN pm_preference_scores ps
                      ON ps.snapshot_id = ls.id AND ps.asset_id = r.asset_id
                    LEFT JOIN duel_counts dc ON dc.asset_id = r.asset_id
                    WHERE r.rank = 1
                ), anchor AS (
                    SELECT * FROM visible
                    ORDER BY duel_count ASC, asset_id ASC
                    LIMIT 1
                ), pair_counts AS (
                    SELECT
                        CASE WHEN d.left_asset_id < d.right_asset_id
                            THEN d.left_asset_id ELSE d.right_asset_id END AS low_id,
                        CASE WHEN d.left_asset_id < d.right_asset_id
                            THEN d.right_asset_id ELSE d.left_asset_id END AS high_id,
                        COUNT(*) AS duel_count
                    FROM pm_asset_duels d
                    JOIN pm_observations o ON o.id = d.observation_id
                    JOIN pm_sessions s ON s.id = o.session_id
                    WHERE s.collection_id = ?1
                    GROUP BY low_id, high_id
                )
                SELECT a.asset_id, a.occurrence_id, a.render_digest, a.rotation_quarters,
                       opponent.asset_id, opponent.occurrence_id,
                       opponent.render_digest, opponent.rotation_quarters,
                       (SELECT id FROM latest_snapshot)
                FROM anchor a
                JOIN visible opponent ON opponent.asset_id != a.asset_id
                LEFT JOIN pair_counts pc ON
                    pc.low_id = MIN(a.asset_id, opponent.asset_id)
                    AND pc.high_id = MAX(a.asset_id, opponent.asset_id)
                ORDER BY COALESCE(pc.duel_count, 0) ASC,
                         opponent.duel_count ASC,
                         ABS(a.score - opponent.score) ASC,
                         opponent.asset_id ASC
                LIMIT 1
                ",
                [collection_id.get()],
                |row| {
                    Ok(PairRow {
                        left_id: row.get(0)?,
                        left_occurrence: row.get(1)?,
                        left_render: row.get(2)?,
                        left_rotation: row.get(3)?,
                        right_id: row.get(4)?,
                        right_occurrence: row.get(5)?,
                        right_render: row.get(6)?,
                        right_rotation: row.get(7)?,
                        snapshot_id: row.get(8)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| {
                Fault::InvalidInput("a comparison requires at least two visible assets".to_owned())
            })?;
        let issued = tx.query_row(
            "SELECT COUNT(*) FROM pm_prompts p
             JOIN pm_sessions s ON s.id = p.session_id
             WHERE s.collection_id = ?1",
            [collection_id.get()],
            |row| row.get::<_, i64>(0),
        )?;
        let prompt = pair.finish(session_id.clone(), now, issued % 2 != 0)?;
        insert_prompt(&tx, &prompt)?;
        tx.commit()?;
        Ok(prompt)
    }

    pub fn record_comparison(
        &self,
        prompt_id: &PromptId,
        winner: &AssetId,
        command_id: &CommandId,
        response_ms: Option<u32>,
    ) -> Result<ObservationId> {
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let digest = payload_digest(&[
            "asset-duel-v1",
            prompt_id.as_str(),
            winner.as_str(),
            &response_ms.map_or_else(String::new, |value| value.to_string()),
        ]);
        if let Some(existing) = existing_command(&tx, command_id, &digest)? {
            return Ok(existing);
        }
        let prompt = load_prompt(&tx, prompt_id)?;
        if prompt.answered || (winner != &prompt.left.asset_id && winner != &prompt.right.asset_id)
        {
            return Err(Fault::StalePrompt(prompt_id.clone()));
        }
        let observation_id = insert_observation(
            &tx,
            command_id,
            &prompt.session_id,
            now,
            "asset-duel-v1",
            Some(&prompt.policy_revision),
            None,
            response_ms,
            "asset_duel",
            &digest,
        )?;
        insert_duel(&tx, observation_id, &prompt, winner)?;
        let answered = tx.execute(
            "UPDATE pm_prompts SET answered_observation_id = ?2
             WHERE id = ?1 AND answered_observation_id IS NULL",
            params![prompt_id.as_str(), observation_id.get()],
        )?;
        if answered != 1 {
            return Err(Fault::StalePrompt(prompt_id.clone()));
        }
        tx.commit()?;
        Ok(observation_id)
    }

    /// Atomically seals the comparison whose challenger became local only after
    /// the user judged it. No prompt can survive without its matching duel.
    pub fn record_promoted_comparison(
        &self,
        session_id: &SessionId,
        left: &AssetId,
        right: &AssetId,
        winner: &AssetId,
        command_id: &CommandId,
        response_ms: Option<u32>,
    ) -> Result<ObservationId> {
        if left == right || (winner != left && winner != right) {
            return Err(Fault::InvalidInput(
                "a promoted comparison requires two distinct assets and one winner".to_owned(),
            ));
        }
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let response = response_ms.map_or_else(String::new, |value| value.to_string());
        let digest = payload_digest(&[
            "remote-admission-duel-v1",
            session_id.as_str(),
            left.as_str(),
            right.as_str(),
            winner.as_str(),
            &response,
        ]);
        if let Some(existing) = existing_command(&tx, command_id, &digest)? {
            return Ok(existing);
        }
        let collection = active_session_collection(&tx, session_id)?;
        let snapshot_id = tx
            .query_row(
                "SELECT id FROM pm_preference_snapshots
                 WHERE collection_id = ?1 ORDER BY id DESC LIMIT 1",
                [collection.get()],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .map(SnapshotId::from_raw);
        let prompt = ComparisonPrompt {
            id: PromptId::fresh(),
            session_id: session_id.clone(),
            left: representative(&tx, collection, left)?,
            right: representative(&tx, collection, right)?,
            policy_revision: REMOTE_ADMISSION_POLICY.to_owned(),
            snapshot_id,
            issued_at_ns: now,
        };
        insert_prompt(&tx, &prompt)?;
        let observation_id = insert_observation(
            &tx,
            command_id,
            session_id,
            now,
            "asset-duel-v1",
            Some(REMOTE_ADMISSION_POLICY),
            None,
            response_ms,
            "asset_duel",
            &digest,
        )?;
        insert_duel(&tx, observation_id, &LoadedPrompt::from(&prompt), winner)?;
        tx.execute(
            "UPDATE pm_prompts SET answered_observation_id = ?2 WHERE id = ?1",
            params![prompt.id.as_str(), observation_id.get()],
        )?;
        tx.commit()?;
        Ok(observation_id)
    }

    pub fn set_favorite(
        &self,
        session_id: &SessionId,
        asset_id: &AssetId,
        active: bool,
        command_id: &CommandId,
    ) -> Result<ObservationId> {
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let digest = payload_digest(&[
            "favorite-v1",
            session_id.as_str(),
            asset_id.as_str(),
            if active { "true" } else { "false" },
        ]);
        if let Some(existing) = existing_command(&tx, command_id, &digest)? {
            return Ok(existing);
        }
        let collection_id = active_session_collection(&tx, session_id)?;
        let observation_id = insert_observation(
            &tx,
            command_id,
            session_id,
            now,
            "favorite-v1",
            None,
            None,
            None,
            "favorite_set",
            &digest,
        )?;
        let changed = tx.execute(
            "UPDATE pm_collection_assets SET favorite = ?3, updated_at_ns = ?4
             WHERE collection_id = ?1 AND asset_id = ?2",
            params![collection_id.get(), asset_id.as_str(), active, now],
        )?;
        if changed != 1 {
            return Err(Fault::InvalidInput(format!(
                "asset {asset_id} does not belong to session collection {collection_id}"
            )));
        }
        tx.execute(
            "INSERT INTO pm_favorite_events(observation_id, collection_id, asset_id, active)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                observation_id.get(),
                collection_id.get(),
                asset_id.as_str(),
                active
            ],
        )?;
        tx.commit()?;
        Ok(observation_id)
    }

    pub fn record_threshold(
        &self,
        session_id: &SessionId,
        asset: &PresentedAsset,
        judgment: ThresholdJudgment,
        command_id: &CommandId,
        response_ms: Option<u32>,
    ) -> Result<ObservationId> {
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let occurrence = asset.occurrence_id.to_string();
        let rotation = asset.rotation_quarters.to_string();
        let response = response_ms.map_or_else(String::new, |value| value.to_string());
        let digest = payload_digest(&[
            "asset-threshold-v1",
            session_id.as_str(),
            asset.asset_id.as_str(),
            &occurrence,
            asset.render.as_str(),
            &rotation,
            judgment.as_str(),
            &response,
        ]);
        if let Some(existing) = existing_command(&tx, command_id, &digest)? {
            return Ok(existing);
        }
        validate_presentation(&tx, session_id, asset)?;
        let id = insert_observation(
            &tx,
            command_id,
            session_id,
            now,
            "asset-threshold-v1",
            None,
            None,
            response_ms,
            "asset_threshold",
            &digest,
        )?;
        tx.execute(
            "INSERT INTO pm_asset_thresholds(
                 observation_id, asset_id, occurrence_id, render_digest,
                 rotation_quarters, judgment
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id.get(),
                asset.asset_id.as_str(),
                asset.occurrence_id.get(),
                asset.render.as_str(),
                asset.rotation_quarters,
                judgment.as_str()
            ],
        )?;
        tx.commit()?;
        Ok(id)
    }

    pub fn record_similarity(
        &self,
        session_id: &SessionId,
        assets: &[PresentedAsset; 3],
        nearest: [&AssetId; 2],
        representation_revision: impl Into<String>,
        command_id: &CommandId,
        response_ms: Option<u32>,
    ) -> Result<ObservationId> {
        let representation_revision =
            nonempty(representation_revision.into(), "representation revision")?;
        let mut ids = HashSet::new();
        if !assets
            .iter()
            .all(|asset| ids.insert(asset.asset_id.as_str()))
            || nearest[0] == nearest[1]
            || !nearest
                .iter()
                .all(|id| assets.iter().any(|asset| &asset.asset_id == *id))
        {
            return Err(Fault::InvalidInput(
                "similarity judgment requires three distinct assets and two distinct members"
                    .to_owned(),
            ));
        }
        let nearest = if nearest[0] < nearest[1] {
            nearest
        } else {
            [nearest[1], nearest[0]]
        };
        let now = now_ns()?;
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let parts = similarity_digest_parts(
            session_id,
            assets,
            nearest,
            &representation_revision,
            response_ms,
        );
        let references = parts.iter().map(String::as_str).collect::<Vec<_>>();
        let digest = payload_digest(&references);
        if let Some(existing) = existing_command(&tx, command_id, &digest)? {
            return Ok(existing);
        }
        for asset in assets {
            validate_presentation(&tx, session_id, asset)?;
        }
        let id = insert_observation(
            &tx,
            command_id,
            session_id,
            now,
            "similarity-triad-v1",
            None,
            Some(&representation_revision),
            response_ms,
            "similarity_triad",
            &digest,
        )?;
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
                id.get(),
                assets[0].asset_id.as_str(),
                assets[1].asset_id.as_str(),
                assets[2].asset_id.as_str(),
                nearest[0].as_str(),
                nearest[1].as_str(),
                assets[0].occurrence_id.get(),
                assets[1].occurrence_id.get(),
                assets[2].occurrence_id.get(),
                assets[0].render.as_str(),
                assets[1].render.as_str(),
                assets[2].render.as_str(),
                assets[0].rotation_quarters,
                assets[1].rotation_quarters,
                assets[2].rotation_quarters,
            ],
        )?;
        tx.commit()?;
        Ok(id)
    }
}

fn insert_observation(
    tx: &Transaction<'_>,
    command_id: &CommandId,
    session_id: &SessionId,
    now: i64,
    prompt_revision: &str,
    policy_revision: Option<&str>,
    representation_revision: Option<&str>,
    response_ms: Option<u32>,
    kind: &str,
    payload_digest: &str,
) -> Result<ObservationId> {
    tx.execute(
        "INSERT INTO pm_observations(
             command_id, payload_digest, session_id, recorded_at_ns, prompt_revision, policy_revision,
             representation_revision, response_ms, ordering_authority, kind
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'recorded', ?9)",
        params![
            command_id.as_str(),
            payload_digest,
            session_id.as_str(),
            now,
            prompt_revision,
            policy_revision,
            representation_revision,
            response_ms,
            kind,
        ],
    )?;
    Ok(ObservationId::from_raw(tx.last_insert_rowid()))
}

fn active_session_collection(tx: &Transaction<'_>, id: &SessionId) -> Result<CollectionId> {
    let (collection_id, ended) = tx
        .query_row(
            "SELECT collection_id, ended_at_ns FROM pm_sessions WHERE id = ?1",
            [id.as_str()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?)),
        )
        .optional()?
        .ok_or_else(|| Fault::InvalidInput(format!("unknown judgment session {id}")))?;
    if ended.is_some() {
        return Err(Fault::InvalidInput(format!(
            "judgment session {id} has ended"
        )));
    }
    Ok(CollectionId::from_raw(collection_id))
}

fn validate_presentation(
    tx: &Transaction<'_>,
    session_id: &SessionId,
    presented: &PresentedAsset,
) -> Result<()> {
    let collection_id = active_session_collection(tx, session_id)?;
    let valid = tx.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM pm_occurrences o JOIN pm_assets a ON a.id = o.asset_id
             WHERE o.id = ?1 AND o.collection_id = ?2 AND o.asset_id = ?3
               AND a.render_digest = ?4 AND o.rotation_quarters = ?5
         )",
        params![
            presented.occurrence_id.get(),
            collection_id.get(),
            presented.asset_id.as_str(),
            presented.render.as_str(),
            presented.rotation_quarters,
        ],
        |row| row.get::<_, bool>(0),
    )?;
    if valid {
        Ok(())
    } else {
        Err(Fault::InvalidInput(
            "presented asset does not match its session, occurrence, and render".to_owned(),
        ))
    }
}

fn representative(
    tx: &Transaction<'_>,
    collection: CollectionId,
    asset: &AssetId,
) -> Result<PresentedAsset> {
    tx.query_row(
        "SELECT o.id, a.render_digest, o.rotation_quarters
         FROM pm_occurrences o
         JOIN pm_assets a ON a.id = o.asset_id
         JOIN pm_collection_assets ca
           ON ca.collection_id = o.collection_id AND ca.asset_id = o.asset_id
         WHERE o.collection_id = ?1 AND o.asset_id = ?2 AND o.present = 1 AND ca.hidden = 0
         ORDER BY o.width * o.height DESC, o.byte_len DESC, o.path ASC
         LIMIT 1",
        params![collection.get(), asset.as_str()],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        },
    )
    .optional()?
    .ok_or_else(|| {
        Fault::InvalidInput(format!(
            "asset {asset} is not visible in collection {collection}"
        ))
    })
    .and_then(|(occurrence, render, rotation)| {
        Ok(PresentedAsset {
            asset_id: asset.clone(),
            occurrence_id: crate::OccurrenceId::from_raw(occurrence),
            render: RenderDigest::parse(render)?,
            rotation_quarters: u8::try_from(rotation)
                .map_err(|_| Fault::Corrupt("invalid occurrence rotation".to_owned()))?,
        })
    })
}

fn insert_prompt(tx: &Transaction<'_>, prompt: &ComparisonPrompt) -> Result<()> {
    tx.execute(
        "INSERT INTO pm_prompts(
             id, session_id, left_asset_id, right_asset_id,
             left_occurrence_id, right_occurrence_id,
             left_render_digest, right_render_digest,
             left_rotation_quarters, right_rotation_quarters,
             policy_revision, snapshot_id, issued_at_ns
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            prompt.id.as_str(),
            prompt.session_id.as_str(),
            prompt.left.asset_id.as_str(),
            prompt.right.asset_id.as_str(),
            prompt.left.occurrence_id.get(),
            prompt.right.occurrence_id.get(),
            prompt.left.render.as_str(),
            prompt.right.render.as_str(),
            prompt.left.rotation_quarters,
            prompt.right.rotation_quarters,
            prompt.policy_revision,
            prompt.snapshot_id.map(SnapshotId::get),
            prompt.issued_at_ns,
        ],
    )?;
    Ok(())
}

fn insert_duel(
    tx: &Transaction<'_>,
    observation: ObservationId,
    prompt: &LoadedPrompt,
    winner: &AssetId,
) -> Result<()> {
    tx.execute(
        "INSERT INTO pm_asset_duels(
             observation_id, prompt_id, left_asset_id, right_asset_id, winner_asset_id,
             left_occurrence_id, right_occurrence_id, left_render_digest, right_render_digest,
             left_rotation_quarters, right_rotation_quarters
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            observation.get(),
            prompt.id.as_str(),
            prompt.left.asset_id.as_str(),
            prompt.right.asset_id.as_str(),
            winner.as_str(),
            prompt.left.occurrence_id.get(),
            prompt.right.occurrence_id.get(),
            prompt.left.render.as_str(),
            prompt.right.render.as_str(),
            prompt.left.rotation_quarters,
            prompt.right.rotation_quarters,
        ],
    )?;
    Ok(())
}

struct LoadedPrompt {
    id: PromptId,
    session_id: SessionId,
    left: PresentedAsset,
    right: PresentedAsset,
    policy_revision: String,
    answered: bool,
}

impl From<&ComparisonPrompt> for LoadedPrompt {
    fn from(prompt: &ComparisonPrompt) -> Self {
        Self {
            id: prompt.id.clone(),
            session_id: prompt.session_id.clone(),
            left: prompt.left.clone(),
            right: prompt.right.clone(),
            policy_revision: prompt.policy_revision.clone(),
            answered: false,
        }
    }
}

fn load_prompt(tx: &Transaction<'_>, id: &PromptId) -> Result<LoadedPrompt> {
    tx.query_row(
        "SELECT session_id, left_asset_id, right_asset_id,
                left_occurrence_id, right_occurrence_id,
                left_render_digest, right_render_digest,
                left_rotation_quarters, right_rotation_quarters,
                policy_revision, answered_observation_id IS NOT NULL
         FROM pm_prompts WHERE id = ?1",
        [id.as_str()],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, i64>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, bool>(10)?,
            ))
        },
    )
    .optional()?
    .ok_or_else(|| Fault::StalePrompt(id.clone()))
    .and_then(|row| {
        Ok(LoadedPrompt {
            id: id.clone(),
            session_id: SessionId::parse(row.0)?,
            left: PresentedAsset {
                asset_id: AssetId::parse(row.1)?,
                occurrence_id: crate::OccurrenceId::from_raw(row.3),
                render: RenderDigest::parse(row.5)?,
                rotation_quarters: u8::try_from(row.7)
                    .map_err(|_| Fault::Corrupt("invalid prompt rotation".to_owned()))?,
            },
            right: PresentedAsset {
                asset_id: AssetId::parse(row.2)?,
                occurrence_id: crate::OccurrenceId::from_raw(row.4),
                render: RenderDigest::parse(row.6)?,
                rotation_quarters: u8::try_from(row.8)
                    .map_err(|_| Fault::Corrupt("invalid prompt rotation".to_owned()))?,
            },
            policy_revision: row.9,
            answered: row.10,
        })
    })
}

fn existing_command(
    tx: &Transaction<'_>,
    command: &CommandId,
    expected_digest: &str,
) -> Result<Option<ObservationId>> {
    let existing = tx
        .query_row(
            "SELECT id, payload_digest FROM pm_observations WHERE command_id = ?1",
            [command.as_str()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    match existing {
        Some((id, digest)) if digest == expected_digest => Ok(Some(ObservationId::from_raw(id))),
        Some(_) => Err(Fault::CommandCollision),
        None => Ok(None),
    }
}

fn payload_digest(parts: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    format!("command-v1:{}", hasher.finalize().to_hex())
}

fn similarity_digest_parts(
    session_id: &SessionId,
    assets: &[PresentedAsset; 3],
    nearest: [&AssetId; 2],
    representation_revision: &str,
    response_ms: Option<u32>,
) -> Vec<String> {
    let mut parts = vec![
        "similarity-triad-v1".to_owned(),
        session_id.to_string(),
        representation_revision.to_owned(),
        nearest[0].to_string(),
        nearest[1].to_string(),
        response_ms.map_or_else(String::new, |value| value.to_string()),
    ];
    for asset in assets {
        parts.extend([
            asset.asset_id.to_string(),
            asset.occurrence_id.to_string(),
            asset.render.as_str().to_owned(),
            asset.rotation_quarters.to_string(),
        ]);
    }
    parts
}

struct PairRow {
    left_id: String,
    left_occurrence: i64,
    left_render: String,
    left_rotation: i64,
    right_id: String,
    right_occurrence: i64,
    right_render: String,
    right_rotation: i64,
    snapshot_id: Option<i64>,
}

impl PairRow {
    fn finish(
        self,
        session_id: SessionId,
        issued_at_ns: i64,
        reverse_sides: bool,
    ) -> Result<ComparisonPrompt> {
        let left = PresentedAsset {
            asset_id: AssetId::parse(self.left_id)?,
            occurrence_id: crate::OccurrenceId::from_raw(self.left_occurrence),
            render: RenderDigest::parse(self.left_render)?,
            rotation_quarters: u8::try_from(self.left_rotation)
                .map_err(|_| Fault::Corrupt("invalid occurrence rotation".to_owned()))?,
        };
        let right = PresentedAsset {
            asset_id: AssetId::parse(self.right_id)?,
            occurrence_id: crate::OccurrenceId::from_raw(self.right_occurrence),
            render: RenderDigest::parse(self.right_render)?,
            rotation_quarters: u8::try_from(self.right_rotation)
                .map_err(|_| Fault::Corrupt("invalid occurrence rotation".to_owned()))?,
        };
        let (left, right) = if reverse_sides {
            (right, left)
        } else {
            (left, right)
        };
        Ok(ComparisonPrompt {
            id: PromptId::fresh(),
            session_id,
            left,
            right,
            policy_revision: PAIR_POLICY.to_owned(),
            snapshot_id: self.snapshot_id.map(SnapshotId::from_raw),
            issued_at_ns,
        })
    }
}

fn nonempty(value: String, field: &str) -> Result<String> {
    if value.trim().is_empty() {
        Err(Fault::InvalidInput(format!("{field} cannot be empty")))
    } else {
        Ok(value)
    }
}
