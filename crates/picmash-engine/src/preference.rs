use std::collections::BTreeMap;

use rusqlite::{OptionalExtension, params};

use crate::{
    Engine,
    engine::now_ns,
    fault::{Fault, Result},
    ids::{AssetId, CollectionId, ObservationId, SnapshotId},
    model::{PreferenceEvaluation, PreferenceScore, PreferenceSnapshot},
};

const MODEL_REVISION: &str = "bradley-terry-l2-v1";
const REGULARIZATION: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
struct Duel {
    winner: usize,
    loser: usize,
}

impl Engine {
    pub fn rebuild_preferences(&self, collection_id: CollectionId) -> Result<PreferenceSnapshot> {
        let material = self.preference_material(collection_id)?;
        if let Some(snapshot) =
            self.exact_snapshot(collection_id, material.frontier, material.catalog_revision)?
        {
            return Ok(snapshot);
        }
        let evaluation = evaluate(&material.duels, material.assets.len());
        let fitted = fit(material.assets.len(), &material.duels);
        let counts = duel_counts(material.assets.len(), &material.duels);

        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let current_catalog = tx.query_row(
            "SELECT catalog_revision FROM pm_collections WHERE id = ?1",
            [collection_id.get()],
            |row| row.get::<_, i64>(0),
        )?;
        let current_frontier = collection_frontier(&tx, collection_id)?;
        if current_catalog != material.catalog_revision || current_frontier != material.frontier {
            return Err(Fault::StaleSnapshot);
        }
        if let Some(id) = exact_snapshot_id(
            &tx,
            collection_id,
            material.frontier,
            material.catalog_revision,
        )? {
            tx.commit()?;
            drop(connection);
            return self.snapshot(SnapshotId::from_raw(id));
        }
        let now = now_ns()?;
        let evaluation_values = evaluation.map(|value| {
            (
                i64::try_from(value.training_duels).unwrap_or(i64::MAX),
                i64::try_from(value.held_out_duels).unwrap_or(i64::MAX),
                value.log_loss,
                value.accuracy,
            )
        });
        tx.execute(
            "INSERT INTO pm_preference_snapshots(
                 collection_id, model_revision, observation_frontier, catalog_revision,
                 created_at_ns, training_duels, held_out_duels,
                 held_out_log_loss, held_out_accuracy
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                collection_id.get(),
                MODEL_REVISION,
                material.frontier,
                material.catalog_revision,
                now,
                evaluation_values.map(|value| value.0),
                evaluation_values.map(|value| value.1),
                evaluation_values.map(|value| value.2),
                evaluation_values.map(|value| value.3),
            ],
        )?;
        let snapshot_id = SnapshotId::from_raw(tx.last_insert_rowid());
        {
            let mut insert = tx.prepare(
                "INSERT INTO pm_preference_scores(snapshot_id, asset_id, score, duel_count)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (index, asset) in material.assets.iter().enumerate() {
                insert.execute(params![
                    snapshot_id.get(),
                    asset.as_str(),
                    fitted[index],
                    counts[index]
                ])?;
            }
        }
        tx.commit()?;
        drop(connection);
        self.snapshot(snapshot_id)
    }

    pub fn latest_preferences(
        &self,
        collection_id: CollectionId,
    ) -> Result<Option<PreferenceSnapshot>> {
        let connection = self.connection.lock();
        let id = connection
            .query_row(
                "SELECT id FROM pm_preference_snapshots
                 WHERE collection_id = ?1 AND model_revision = ?2
                 ORDER BY id DESC LIMIT 1",
                params![collection_id.get(), MODEL_REVISION],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        drop(connection);
        id.map(|id| self.snapshot(SnapshotId::from_raw(id)))
            .transpose()
    }

    pub fn snapshot(&self, id: SnapshotId) -> Result<PreferenceSnapshot> {
        let connection = self.connection.lock();
        let header = connection
            .query_row(
                "SELECT collection_id, model_revision, observation_frontier, catalog_revision,
                        training_duels, held_out_duels, held_out_log_loss, held_out_accuracy
                 FROM pm_preference_snapshots WHERE id = ?1",
                [id.get()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<i64>>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                        row.get::<_, Option<f64>>(6)?,
                        row.get::<_, Option<f64>>(7)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Fault::InvalidInput(format!("unknown preference snapshot {id}")))?;
        let evaluation = match (header.4, header.5, header.6, header.7) {
            (Some(training), Some(held_out), Some(log_loss), Some(accuracy)) => {
                Some(PreferenceEvaluation {
                    training_duels: usize::try_from(training)
                        .map_err(|_| Fault::Corrupt("invalid training duel count".to_owned()))?,
                    held_out_duels: usize::try_from(held_out)
                        .map_err(|_| Fault::Corrupt("invalid held-out duel count".to_owned()))?,
                    log_loss,
                    accuracy,
                })
            }
            (None, None, None, None) => None,
            _ => {
                return Err(Fault::Corrupt(
                    "partial preference evaluation in database".to_owned(),
                ));
            }
        };
        let mut statement = connection.prepare(
            "SELECT asset_id, score, duel_count FROM pm_preference_scores
             WHERE snapshot_id = ?1 ORDER BY score DESC, asset_id ASC",
        )?;
        let rows = statement.query_map([id.get()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        let scores = rows
            .map(|row| {
                let (asset, score, count) = row?;
                Ok(PreferenceScore {
                    asset_id: AssetId::parse(asset)?,
                    score,
                    duel_count: u32::try_from(count)
                        .map_err(|_| Fault::Corrupt("invalid duel count".to_owned()))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PreferenceSnapshot {
            id,
            collection_id: CollectionId::from_raw(header.0),
            model_revision: header.1,
            observation_frontier: ObservationId::from_raw(header.2),
            catalog_revision: u64::try_from(header.3)
                .map_err(|_| Fault::Corrupt("invalid catalog revision".to_owned()))?,
            evaluation,
            scores,
        })
    }

    fn preference_material(&self, collection_id: CollectionId) -> Result<PreferenceMaterial> {
        let connection = self.connection.lock();
        let catalog_revision = connection.query_row(
            "SELECT catalog_revision FROM pm_collections WHERE id = ?1",
            [collection_id.get()],
            |row| row.get::<_, i64>(0),
        )?;
        let frontier = collection_frontier(&connection, collection_id)?;
        let mut statement = connection.prepare(
            "SELECT ca.asset_id FROM pm_collection_assets ca
             WHERE ca.collection_id = ?1 AND ca.hidden = 0
               AND EXISTS (
                   SELECT 1 FROM pm_occurrences o
                   WHERE o.collection_id = ca.collection_id AND o.asset_id = ca.asset_id
                     AND o.present = 1
               )
             ORDER BY ca.asset_id",
        )?;
        let assets = statement
            .query_map([collection_id.get()], |row| row.get::<_, String>(0))?
            .map(|row| AssetId::parse(row?).map_err(Fault::from))
            .collect::<Result<Vec<_>>>()?;
        let index = assets
            .iter()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let mut statement = connection.prepare(
            "SELECT d.left_asset_id, d.right_asset_id, d.winner_asset_id
             FROM pm_asset_duels d
             JOIN pm_observations o ON o.id = d.observation_id
             JOIN pm_sessions s ON s.id = o.session_id
             WHERE s.collection_id = ?1
             ORDER BY o.id",
        )?;
        let rows = statement.query_map([collection_id.get()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut duels = Vec::new();
        for row in rows {
            let (left, right, winner) = row?;
            let (Some(&left), Some(&right)) = (index.get(left.as_str()), index.get(right.as_str()))
            else {
                continue;
            };
            duels.push(if winner == assets[left].as_str() {
                Duel {
                    winner: left,
                    loser: right,
                }
            } else if winner == assets[right].as_str() {
                Duel {
                    winner: right,
                    loser: left,
                }
            } else {
                return Err(Fault::Corrupt(
                    "duel winner is absent from its pair".to_owned(),
                ));
            });
        }
        Ok(PreferenceMaterial {
            assets,
            duels,
            frontier,
            catalog_revision,
        })
    }

    fn exact_snapshot(
        &self,
        collection_id: CollectionId,
        frontier: i64,
        catalog_revision: i64,
    ) -> Result<Option<PreferenceSnapshot>> {
        let connection = self.connection.lock();
        let id = exact_snapshot_id(&connection, collection_id, frontier, catalog_revision)?;
        drop(connection);
        id.map(|id| self.snapshot(SnapshotId::from_raw(id)))
            .transpose()
    }
}

struct PreferenceMaterial {
    assets: Vec<AssetId>,
    duels: Vec<Duel>,
    frontier: i64,
    catalog_revision: i64,
}

fn collection_frontier(
    connection: &rusqlite::Connection,
    collection_id: CollectionId,
) -> Result<i64> {
    Ok(connection.query_row(
        "SELECT COALESCE(MAX(o.id), 0)
         FROM pm_observations o
         JOIN pm_asset_duels d ON d.observation_id = o.id
         JOIN pm_sessions s ON s.id = o.session_id
         WHERE s.collection_id = ?1",
        [collection_id.get()],
        |row| row.get(0),
    )?)
}

fn exact_snapshot_id(
    connection: &rusqlite::Connection,
    collection_id: CollectionId,
    frontier: i64,
    catalog_revision: i64,
) -> Result<Option<i64>> {
    Ok(connection
        .query_row(
            "SELECT id FROM pm_preference_snapshots
             WHERE collection_id = ?1 AND model_revision = ?2
               AND observation_frontier = ?3 AND catalog_revision = ?4",
            params![
                collection_id.get(),
                MODEL_REVISION,
                frontier,
                catalog_revision
            ],
            |row| row.get(0),
        )
        .optional()?)
}

fn fit(asset_count: usize, duels: &[Duel]) -> Vec<f64> {
    let mut scores = vec![0.0; asset_count];
    for _ in 0..200 {
        let mut gradient: Vec<f64> = scores.iter().map(|score| -REGULARIZATION * score).collect();
        let mut curvature = vec![REGULARIZATION; asset_count];
        for duel in duels {
            let probability = sigmoid(scores[duel.winner] - scores[duel.loser]);
            let residual = 1.0 - probability;
            let weight = probability * (1.0 - probability);
            gradient[duel.winner] += residual;
            gradient[duel.loser] -= residual;
            curvature[duel.winner] += weight;
            curvature[duel.loser] += weight;
        }
        let mut max_step: f64 = 0.0;
        for index in 0..asset_count {
            let step = 0.5 * gradient[index] / curvature[index];
            scores[index] += step;
            max_step = max_step.max(step.abs());
        }
        if !scores.is_empty() {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            for score in &mut scores {
                *score -= mean;
            }
        }
        if max_step < 1e-10 {
            break;
        }
    }
    scores
}

fn evaluate(duels: &[Duel], asset_count: usize) -> Option<PreferenceEvaluation> {
    if duels.len() < 20 {
        return None;
    }
    let held_out = (duels.len() / 5).max(1);
    let split = duels.len() - held_out;
    let scores = fit(asset_count, &duels[..split]);
    let mut log_loss = 0.0;
    let mut correct = 0;
    for duel in &duels[split..] {
        let probability = sigmoid(scores[duel.winner] - scores[duel.loser]);
        log_loss -= probability.max(f64::MIN_POSITIVE).ln();
        correct += usize::from(probability > 0.5);
    }
    Some(PreferenceEvaluation {
        training_duels: split,
        held_out_duels: held_out,
        log_loss: log_loss / held_out as f64,
        accuracy: correct as f64 / held_out as f64,
    })
}

fn duel_counts(asset_count: usize, duels: &[Duel]) -> Vec<u32> {
    let mut counts = vec![0_u32; asset_count];
    for duel in duels {
        counts[duel.winner] = counts[duel.winner].saturating_add(1);
        counts[duel.loser] = counts[duel.loser].saturating_add(1);
    }
    counts
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::{Duel, fit};

    #[test]
    fn repeated_wins_order_the_pair_without_moving_its_center() {
        let duels = vec![
            Duel {
                winner: 0,
                loser: 1,
            };
            12
        ];
        let scores = fit(2, &duels);
        assert!(scores[0] > scores[1]);
        assert!((scores[0] + scores[1]).abs() < 1e-10);
    }
}
