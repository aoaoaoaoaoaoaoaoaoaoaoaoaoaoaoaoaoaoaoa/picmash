use super::*;
use crate::maintenance::{
    ClaimedMaintenanceJob, MaintenanceJobKind, MaintenanceJobSpec, MaintenancePriority,
};

impl Store {
    pub fn enqueue_maintenance_job(&self, job: &MaintenanceJobSpec) -> anyhow::Result<()> {
        self.conn.execute(
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
                job.kind.as_str(),
                job.key,
                job.priority.as_i64(),
                job.not_before_ts,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn claim_next_maintenance_job(&mut self) -> anyhow::Result<Option<ClaimedMaintenanceJob>> {
        let tx = self
            .conn
            .transaction()
            .context("opening maintenance claim transaction")?;
        let now = now_ts();
        let claimed = tx
            .query_row(
                r"
                SELECT kind, job_key, priority, generation
                FROM maintenance_jobs
                WHERE running_generation IS NULL
                  AND next_run_at <= ?1
                ORDER BY priority ASC, next_run_at ASC, updated_at ASC
                LIMIT 1
                ",
                params![now],
                |row| {
                    Ok(ClaimedMaintenanceJob {
                        kind: row
                            .get::<_, String>(0)?
                            .parse()
                            .map_err(maintenance_payload_into_rusqlite)?,
                        key: row.get(1)?,
                        priority: MaintenancePriority::from_i64(row.get::<_, i64>(2)?)
                            .map_err(maintenance_payload_into_rusqlite)?,
                        generation: row.get(3)?,
                    })
                },
            )
            .optional()?;
        let Some(job) = claimed else {
            tx.commit()
                .context("committing empty maintenance claim transaction")?;
            return Ok(None);
        };
        let claimed_rows = tx.execute(
            r"
            UPDATE maintenance_jobs
            SET running_generation = ?3,
                attempts = attempts + 1,
                last_error = NULL,
                updated_at = ?4
            WHERE kind = ?1
              AND job_key = ?2
              AND generation = ?3
              AND running_generation IS NULL
            ",
            params![job.kind.as_str(), job.key, job.generation, now],
        )?;
        tx.commit()
            .context("committing maintenance claim transaction")?;
        Ok((claimed_rows > 0).then_some(job))
    }

    pub fn complete_maintenance_job(&self, job: &ClaimedMaintenanceJob) -> anyhow::Result<()> {
        let deleted = self.conn.execute(
            r"
            DELETE FROM maintenance_jobs
            WHERE kind = ?1
              AND job_key = ?2
              AND generation = ?3
              AND running_generation = ?3
            ",
            params![job.kind.as_str(), job.key, job.generation],
        )?;
        if deleted == 0 {
            self.conn.execute(
                r"
                UPDATE maintenance_jobs
                SET running_generation = NULL,
                    last_error = NULL,
                    next_run_at = MIN(next_run_at, ?3),
                    updated_at = ?3
                WHERE kind = ?1
                  AND job_key = ?2
                  AND running_generation = ?4
                ",
                params![job.kind.as_str(), job.key, now_ts(), job.generation],
            )?;
        }
        Ok(())
    }

    pub fn fail_maintenance_job(
        &self,
        job: &ClaimedMaintenanceJob,
        error: &str,
        retry_at: i64,
    ) -> anyhow::Result<()> {
        let updated = self.conn.execute(
            r"
            UPDATE maintenance_jobs
            SET running_generation = NULL,
                next_run_at = ?4,
                last_error = ?5,
                updated_at = ?6
            WHERE kind = ?1
              AND job_key = ?2
              AND generation = ?3
              AND running_generation = ?3
            ",
            params![
                job.kind.as_str(),
                job.key,
                job.generation,
                retry_at,
                error,
                now_ts(),
            ],
        )?;
        if updated == 0 {
            let now = now_ts();
            self.conn.execute(
                r"
                UPDATE maintenance_jobs
                SET running_generation = NULL,
                    next_run_at = MIN(next_run_at, ?3),
                    last_error = ?4,
                    updated_at = ?3
                WHERE kind = ?1
                  AND job_key = ?2
                  AND running_generation = ?5
                ",
                params![job.kind.as_str(), job.key, now, error, job.generation],
            )?;
        }
        Ok(())
    }

    pub fn has_pending_maintenance_job(
        &self,
        kind: MaintenanceJobKind,
        key: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM maintenance_jobs
                WHERE kind = ?1
                  AND job_key = ?2
                LIMIT 1
                ",
                params![kind.as_str(), key],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }
}

fn maintenance_payload_into_rusqlite(error: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(error.to_string())),
    )
}
