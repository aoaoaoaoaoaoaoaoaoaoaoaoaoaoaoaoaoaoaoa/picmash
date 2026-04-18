use super::*;
use std::time::Instant;
use time::OffsetDateTime;
use tracing::{field, info_span, warn};

const MAINTENANCE_IDLE_POLL_SECONDS: u64 = 2;
const IDENTITY_REVIEW_REFRESH_DEBOUNCE_SECONDS: i64 = 8;
const BOOTSTRAP_MAINTENANCE_BATCH: usize = 32;
const CORPUS_FACE_SCAN_BATCH: usize = 8;
const CORPUS_FACE_RECOGNITION_BATCH: usize = 32;
const CORPUS_QUALITY_FEATURE_BATCH: usize = 32;
const SLOW_MAINTENANCE_JOB_MS: u128 = 150;

impl AppState {
    pub fn maintenance_idle_poll(&self) -> Duration {
        Duration::seconds(i64::try_from(MAINTENANCE_IDLE_POLL_SECONDS).unwrap_or(2))
    }

    pub fn schedule_bootstrap_maintenance(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::keyed(
            MaintenanceJobKind::BootstrapMaintenance,
            crate::store::BOOTSTRAP_PHASE_INITIAL,
            MaintenancePriority::Cold,
            maintenance_now_ts(),
        ));
    }

    fn schedule_bootstrap_phase(&self, phase: &str) {
        self.schedule_maintenance_job(MaintenanceJobSpec::keyed(
            MaintenanceJobKind::BootstrapMaintenance,
            phase,
            MaintenancePriority::Cold,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_corpus_ingest(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::CorpusIngest,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_local_directory_refresh(&self, source_key: &str) {
        if self
            .read_store()
            .ok()
            .and_then(|store| {
                store
                    .has_pending_maintenance_job(
                        MaintenanceJobKind::LocalDirectoryRefresh,
                        source_key,
                    )
                    .ok()
            })
            .unwrap_or(false)
        {
            return;
        }
        self.schedule_maintenance_job(MaintenanceJobSpec::keyed(
            MaintenanceJobKind::LocalDirectoryRefresh,
            source_key,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_corpus_face_scan_backfill(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::CorpusFaceScanBackfill,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_corpus_face_recognition_backfill(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::CorpusFaceRecognitionBackfill,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_corpus_quality_feature_backfill(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::CorpusQualityFeatureBackfill,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_external_face_embedding_backfill(&self, source_key: &str) {
        self.schedule_maintenance_job(MaintenanceJobSpec::keyed(
            MaintenanceJobKind::ExternalFaceEmbeddingBackfill,
            source_key,
            MaintenancePriority::Warm,
            maintenance_now_ts(),
        ));
    }

    pub fn schedule_quality_model_refresh(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::QualityModelRefresh,
            MaintenancePriority::Hot,
            maintenance_now_ts()
                + i64::try_from(QUALITY_MODEL_REFRESH_DEBOUNCE_SECONDS).unwrap_or(4),
        ));
    }

    pub fn schedule_identity_review_refresh(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::IdentityReviewRefresh,
            MaintenancePriority::Hot,
            maintenance_now_ts() + IDENTITY_REVIEW_REFRESH_DEBOUNCE_SECONDS,
        ));
    }

    fn schedule_maintenance_job(&self, job: MaintenanceJobSpec) {
        let job_for_write = job.clone();
        if let Err(error) = self.with_write_store("enqueue_maintenance_job", move |store| {
            store.enqueue_maintenance_job(&job_for_write)
        }) {
            warn!(
                error = %format!("{error:#}"),
                kind = job.kind.as_str(),
                key = job.key,
                "failed to enqueue maintenance job"
            );
            return;
        }
        self.maintenance_notify.notify_one();
    }

    pub async fn wait_for_maintenance_signal(&self) {
        self.maintenance_notify.notified().await;
    }

    pub fn devour_one_maintenance_job(&self) -> anyhow::Result<bool> {
        let claimed = self.with_write_store("claim_maintenance_job", |store| {
            store.claim_next_maintenance_job()
        })?;
        let Some(job) = claimed else {
            return Ok(false);
        };
        let started = Instant::now();
        let span = info_span!(
            "maintenance.job",
            kind = job.kind.as_str(),
            key = %job.key,
            priority = job.priority.as_i64(),
            generation = job.generation,
            elapsed_ms = field::Empty,
            retry_at = field::Empty,
        );
        let _entered = span.enter();
        let result = self.execute_maintenance_job(&job);
        let elapsed_ms = started.elapsed().as_millis();
        span.record("elapsed_ms", field::display(elapsed_ms));
        if elapsed_ms > SLOW_MAINTENANCE_JOB_MS {
            warn!(elapsed_ms, "slow maintenance job");
        }
        match result {
            Ok(()) => {
                self.with_write_store("complete_maintenance_job", move |store| {
                    store.complete_maintenance_job(&job)
                })?;
            }
            Err(error) => {
                let retry_at = maintenance_now_ts() + maintenance_retry_seconds(job.kind);
                let job_for_write = job.clone();
                let error_message = format!("{error:#}");
                span.record("retry_at", field::display(retry_at));
                self.with_write_store("fail_maintenance_job", move |store| {
                    store.fail_maintenance_job(&job_for_write, &error_message, retry_at)
                })?;
                warn!(
                    error = %format!("{error:#}"),
                    kind = job.kind.as_str(),
                    key = job.key,
                    retry_at,
                    "maintenance job failed"
                );
            }
        }
        Ok(true)
    }

    fn execute_maintenance_job(&self, job: &ClaimedMaintenanceJob) -> anyhow::Result<()> {
        match job.kind {
            MaintenanceJobKind::BootstrapMaintenance => {
                let progress = self.devour_bootstrap_maintenance_batch(
                    if job.key.is_empty() {
                        crate::store::BOOTSTRAP_PHASE_INITIAL
                    } else {
                        &job.key
                    },
                    BOOTSTRAP_MAINTENANCE_BATCH,
                )?;
                if let Some(phase) = progress.requeue_phase {
                    self.schedule_bootstrap_phase(phase);
                }
                Ok(())
            }
            MaintenanceJobKind::CorpusIngest => self.ingest_corpus(),
            MaintenanceJobKind::LocalDirectoryRefresh => {
                self.devour_local_directory_refresh(&job.key)
            }
            MaintenanceJobKind::ExternalOutcomeSeal => {
                self.devour_pending_external_import_outcome(RemoteItemId(job.key.parse()?))
            }
            MaintenanceJobKind::CorpusFaceScanBackfill => {
                let scanned = self.devour_corpus_face_scan_batch(CORPUS_FACE_SCAN_BATCH)?;
                if scanned == CORPUS_FACE_SCAN_BATCH {
                    self.schedule_corpus_face_scan_backfill();
                }
                Ok(())
            }
            MaintenanceJobKind::CorpusFaceRecognitionBackfill => {
                let backfilled = self
                    .devour_corpus_face_recognition_backfill_batch(CORPUS_FACE_RECOGNITION_BATCH)?;
                if backfilled == CORPUS_FACE_RECOGNITION_BATCH {
                    self.schedule_corpus_face_recognition_backfill();
                }
                Ok(())
            }
            MaintenanceJobKind::CorpusQualityFeatureBackfill => {
                let warmed =
                    self.devour_corpus_quality_feature_batch(CORPUS_QUALITY_FEATURE_BATCH)?;
                if warmed == CORPUS_QUALITY_FEATURE_BATCH {
                    self.schedule_corpus_quality_feature_backfill();
                }
                if warmed > 0 {
                    self.schedule_quality_model_refresh();
                }
                Ok(())
            }
            MaintenanceJobKind::ExternalFaceEmbeddingBackfill => {
                let embedded = self.backfill_external_face_embeddings_for_source(
                    &job.key,
                    EXTERNAL_FACE_BACKFILL_BATCH,
                )?;
                if embedded == EXTERNAL_FACE_BACKFILL_BATCH {
                    self.schedule_external_face_embedding_backfill(&job.key);
                }
                Ok(())
            }
            MaintenanceJobKind::QualityModelRefresh => self.devour_quality_model_refresh(),
            MaintenanceJobKind::IdentityReviewRefresh => {
                self.devour_identity_review_refresh()?;
                self.schedule_quality_model_refresh();
                Ok(())
            }
        }
    }
}

fn maintenance_now_ts() -> i64 {
    OffsetDateTime::now_utc().unix_timestamp()
}

fn maintenance_retry_seconds(kind: MaintenanceJobKind) -> i64 {
    match kind {
        MaintenanceJobKind::BootstrapMaintenance => 60,
        MaintenanceJobKind::CorpusIngest => 30,
        MaintenanceJobKind::LocalDirectoryRefresh => 20,
        MaintenanceJobKind::ExternalOutcomeSeal => 5,
        MaintenanceJobKind::CorpusFaceScanBackfill => 20,
        MaintenanceJobKind::CorpusFaceRecognitionBackfill => 20,
        MaintenanceJobKind::CorpusQualityFeatureBackfill => 20,
        MaintenanceJobKind::ExternalFaceEmbeddingBackfill => 15,
        MaintenanceJobKind::QualityModelRefresh => 10,
        MaintenanceJobKind::IdentityReviewRefresh => 10,
    }
}
