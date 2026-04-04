use super::*;
use time::OffsetDateTime;

const MAINTENANCE_IDLE_POLL_SECONDS: u64 = 2;
const IDENTITY_REVIEW_REFRESH_DEBOUNCE_SECONDS: i64 = 8;
const CORPUS_FACE_SCAN_BATCH: usize = 8;
const CORPUS_FACE_RECOGNITION_BATCH: usize = 32;
const CORPUS_QUALITY_FEATURE_BATCH: usize = 32;

impl AppState {
    pub fn maintenance_idle_poll(&self) -> Duration {
        Duration::seconds(i64::try_from(MAINTENANCE_IDLE_POLL_SECONDS).unwrap_or(2))
    }

    pub fn schedule_bootstrap_maintenance(&self) {
        self.schedule_maintenance_job(MaintenanceJobSpec::singleton(
            MaintenanceJobKind::BootstrapMaintenance,
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
            .store
            .lock()
            .has_pending_maintenance_job(MaintenanceJobKind::LocalDirectoryRefresh, source_key)
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
        if let Err(error) = self.with_fresh_store_write(|store| store.enqueue_maintenance_job(&job))
        {
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
        let claimed = self.with_fresh_store_write(|store| store.claim_next_maintenance_job())?;
        let Some(job) = claimed else {
            return Ok(false);
        };
        let result = self.execute_maintenance_job(&job);
        match result {
            Ok(()) => {
                self.with_fresh_store_write(|store| store.complete_maintenance_job(&job))?;
            }
            Err(error) => {
                let retry_at = maintenance_now_ts() + maintenance_retry_seconds(job.kind);
                self.with_fresh_store_write(|store| {
                    store.fail_maintenance_job(&job, &format!("{error:#}"), retry_at)
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
            MaintenanceJobKind::BootstrapMaintenance => self.devour_bootstrap_maintenance(),
            MaintenanceJobKind::CorpusIngest => self.ingest_corpus(),
            MaintenanceJobKind::LocalDirectoryRefresh => {
                self.devour_local_directory_refresh(&job.key)
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
        MaintenanceJobKind::CorpusFaceScanBackfill => 20,
        MaintenanceJobKind::CorpusFaceRecognitionBackfill => 20,
        MaintenanceJobKind::CorpusQualityFeatureBackfill => 20,
        MaintenanceJobKind::ExternalFaceEmbeddingBackfill => 15,
        MaintenanceJobKind::QualityModelRefresh => 10,
        MaintenanceJobKind::IdentityReviewRefresh => 10,
    }
}
