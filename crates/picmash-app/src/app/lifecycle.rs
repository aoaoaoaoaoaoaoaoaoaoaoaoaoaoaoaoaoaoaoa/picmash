use super::*;
use crate::quality_features::{
    QUALITY_FEATURE_REVISION, StoredAssetQualityFeatures, extract_asset_quality_features,
};

struct PreparedDetectedFace {
    face: DetectedFace,
    embedding: Option<crate::model::EmbeddingRecord>,
    recognition: Option<crate::model::EmbeddingRecord>,
    display_crop_png: Option<Vec<u8>>,
}

struct PreparedCorpusFacePass {
    asset_id: AssetId,
    detector_model: String,
    faces: Vec<PreparedDetectedFace>,
}

struct PersistedDetectedFaceCrop {
    face_id: FaceId,
    display_crop_png: Vec<u8>,
}

struct PersistedCorpusFacePass {
    detected_count: usize,
    embedded_count: usize,
    crops: Vec<PersistedDetectedFaceCrop>,
}

#[derive(Debug, Clone)]
struct AppBootPaths {
    db_path: PathBuf,
    model_cache_root: PathBuf,
    cache_root: PathBuf,
    source_cache_root: PathBuf,
}

impl AppBootPaths {
    fn resolve() -> anyhow::Result<Self> {
        let dirs = ProjectDirs::from("moe", "swarm", "picmash")
            .context("resolving XDG directories for picmash")?;
        Ok(Self {
            db_path: dirs.data_local_dir().join("picmash.sqlite3"),
            model_cache_root: dirs.cache_dir().to_path_buf(),
            cache_root: dirs.cache_dir().join("renditions"),
            source_cache_root: dirs.cache_dir().join("sources"),
        })
    }
}

impl AppState {
    pub fn boot(
        root_path: &Path,
        config: AppConfig,
        config_path: PathBuf,
        config_digest: String,
    ) -> anyhow::Result<Self> {
        Self::boot_with_paths(
            root_path,
            config,
            config_path,
            config_digest,
            AppBootPaths::resolve()?,
        )
    }

    fn boot_with_paths(
        root_path: &Path,
        config: AppConfig,
        config_path: PathBuf,
        config_digest: String,
        paths: AppBootPaths,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(&paths.model_cache_root).with_context(|| {
            format!(
                "creating model cache root {}",
                paths.model_cache_root.display()
            )
        })?;
        fs::create_dir_all(&paths.cache_root).with_context(|| {
            format!(
                "creating rendition cache root {}",
                paths.cache_root.display()
            )
        })?;
        fs::create_dir_all(&paths.source_cache_root).with_context(|| {
            format!(
                "creating source cache root {}",
                paths.source_cache_root.display()
            )
        })?;
        let root_path = root_path
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root_path.display()))?;

        info!(db = %paths.db_path.display(), "opening store");
        let mut store = Store::open(&paths.db_path)?;
        let mut quality_model = store.active_quality_model()?;
        if quality_model.formal_version
            != crate::quality::CURRENT_RUNTIME_QUALITY_MODEL.as_version()
        {
            quality_model = crate::quality::QualityModelRecord::runtime_default(
                time::OffsetDateTime::now_utc(),
            );
            store.set_active_quality_model(&quality_model)?;
        }

        info!("initializing ONNX inference engine");
        let embedder = OnnxEngine::from_env(&paths.model_cache_root);
        if embedder.enabled() {
            info!(model = %embedder.model_name(), "ONNX DINO inference ready");
        } else {
            info!(note = ?embedder.status_note(), "ONNX DINO inference disabled");
        }
        if embedder.face_detection_enabled() {
            info!("ONNX SCRFD face detection ready");
        } else if let Some(note) = embedder.scrfd_note() {
            info!(note, "ONNX SCRFD face detection disabled");
        }
        if embedder.recognition_enabled() {
            info!(
                model = %embedder.recognition_model_name(),
                "ONNX ArcFace recognition ready"
            );
        } else if let Some(note) = embedder.arcface_note() {
            info!(note, "ONNX ArcFace recognition disabled");
        }
        let quality_replay = store.rebuild_active_quality_state(embedder.model_name())?;
        let purged_quality_rows = store.purge_quality_cache_except(quality_model.formal_version)?;
        info!(
            formal_version = %quality_model.formal_version,
            prior_family = %quality_model.prior_family,
            prior_revision = %quality_model.prior_revision,
            purged_stale_quality_rows = purged_quality_rows,
            "quality model cache regime ready"
        );
        info!(
            formal_version = %quality_replay.formal_version,
            assets = quality_replay.asset_count,
            sessions = quality_replay.session_count,
            subjects = quality_replay.subject_count,
            comparisons = quality_replay.comparison_events,
            nudges = quality_replay.nudge_events,
            hearts = quality_replay.heart_events,
            "quality state rebuilt from replay truth"
        );

        let source_scanner = SourceScanner::new(&paths.source_cache_root)?;

        let corpus_id = store.ensure_corpus_id(&root_path)?;

        info!("resuming session");
        let session = store.resume_or_create_session(corpus_id, SESSION_SNAP_WINDOW)?;

        Ok(Self {
            writer: writer::DbWriter::spawn(&paths.db_path)?,
            db_path: paths.db_path,
            config: RwLock::new(config),
            config_path,
            config_reload: Mutex::new(ConfigReloadState::forge(config_digest)),
            embedder,
            source_scanner,
            active: ActiveArena {
                corpus_id,
                session_id: session.id,
            },
            root_path,
            cache_root: paths.cache_root,
            source_cache_root: paths.source_cache_root,
            session_field_cache: RwLock::new(None),
            explore_layouts: RwLock::new(HashMap::new()),
            explore_vectors: RwLock::new(None),
            asset_domain_oracle: RwLock::new(None),
            duplicate_frontier: RwLock::new(None),
            face_oracle: RwLock::new(None),
            maintenance_notify: Notify::new(),
            arena_session: Mutex::new(ArenaSessionRuntime::default()),
            recent_facemash_pairs: Mutex::new(VecDeque::with_capacity(
                FACEMASH_RECENT_PAIR_EXCLUDE,
            )),
            recent_facemash_identities: Mutex::new(VecDeque::with_capacity(
                FACEMASH_RECENT_FACE_EXCLUDE,
            )),
            recent_facemash_faces: Mutex::new(VecDeque::with_capacity(
                FACEMASH_RECENT_FACE_EXCLUDE,
            )),
            identity_handle_key: rng().random(),
        })
    }

    /// Background corpus ingest. Safe to call after the app is already serving.
    pub fn ingest_corpus(&self) -> anyhow::Result<()> {
        info!(root = %self.root_path.display(), "ingesting corpus");
        let mut store = Store::open_hot(&self.db_path)?;
        store.ingest_corpus(&self.root_path, self.active.corpus_id, &self.embedder)?;
        self.invalidate_session_field_cache();
        self.purge_explore_vectors();
        self.purge_all_explore_layouts();
        self.schedule_corpus_face_scan_backfill();
        self.schedule_corpus_face_recognition_backfill();
        self.schedule_corpus_quality_feature_backfill();
        self.schedule_quality_model_refresh();
        Ok(())
    }

    pub fn devour_bootstrap_maintenance_batch(
        &self,
        phase: &str,
        limit: usize,
    ) -> anyhow::Result<crate::store::BootstrapMaintenanceProgress> {
        let progress = self.with_write_store("bootstrap_maintenance_batch", {
            let phase = phase.to_owned();
            move |store| store.devour_bootstrap_maintenance_batch(&phase, limit)
        })?;
        if progress.processed > 0 {
            self.invalidate_session_field_cache();
            if phase == crate::store::BOOTSTRAP_PHASE_FACE_IDENTITY_BINDINGS
                && progress.requeue_phase != Some(phase)
            {
                let store = self.read_store()?;
                self.retrain_face_oracle(&store)?;
            }
            info!(
                phase,
                processed = progress.processed,
                next_phase = progress.requeue_phase,
                "deferred bootstrap maintenance batch complete"
            );
        }
        Ok(progress)
    }

    pub(super) fn devour_corpus_face_scan_batch(&self, limit: usize) -> anyhow::Result<usize> {
        if !self.embedder.face_detection_enabled() {
            return Ok(0);
        }

        let store = Store::open_hot(&self.db_path)?;
        let assets = store.corpus_assets(self.active.corpus_id)?;
        let detector_model = self.embedder.face_detection_model_name().to_owned();
        let mut scanned_assets = 0usize;
        let mut detected_total = 0usize;
        let mut embedded_total = 0usize;

        for asset in assets {
            if scanned_assets >= limit {
                break;
            }
            if store.face_scan_exists_for_asset(&asset.id, &detector_model)? {
                continue;
            }
            scanned_assets += 1;
            let Some(prepared) = self.prepare_corpus_face_pass(&asset, &detector_model)? else {
                continue;
            };
            let persisted = self.persist_corpus_face_pass(prepared)?;
            detected_total += persisted.detected_count;
            embedded_total += persisted.embedded_count;
            self.write_persisted_face_crops(persisted.crops)?;
        }

        if scanned_assets > 0 {
            info!(
                assets = scanned_assets,
                faces = detected_total,
                embedded = embedded_total,
                "corpus face detection batch complete"
            );
        }
        Ok(scanned_assets)
    }

    fn prepare_corpus_face_pass(
        &self,
        asset: &AssetRecord,
        detector_model: &str,
    ) -> anyhow::Result<Option<PreparedCorpusFacePass>> {
        let Ok(bytes) = fs::read(&asset.path) else {
            return Ok(None);
        };
        let Ok(image) = crate::identity::canonical_embedding_image(&bytes) else {
            return Ok(None);
        };
        let faces = self.embedder.detect_faces(&image)?;
        let prepared_faces = faces
            .into_iter()
            .filter(|face| {
                face.bbox.w.min(face.bbox.h) >= crate::store::FACE_DETECTION_MIN_FACE_SIDE
            })
            .map(|face| {
                let embedding_crop = align_face_for_embedding(&image, &face);
                let embedding = self.embedder.embed_image(&embedding_crop.crop)?;
                let recognition = self.embedder.recognize_face_image(&embedding_crop.crop)?;
                let display_crop_png = encode_display_crop_png(&image, &face);
                Ok(PreparedDetectedFace {
                    face,
                    embedding,
                    recognition,
                    display_crop_png,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Some(PreparedCorpusFacePass {
            asset_id: asset.id.clone(),
            detector_model: detector_model.to_owned(),
            faces: prepared_faces,
        }))
    }

    fn persist_corpus_face_pass(
        &self,
        prepared: PreparedCorpusFacePass,
    ) -> anyhow::Result<PersistedCorpusFacePass> {
        self.with_write_store("persist_corpus_face_pass", move |store| {
            if store.face_scan_exists_for_asset(&prepared.asset_id, &prepared.detector_model)? {
                return Ok(PersistedCorpusFacePass {
                    detected_count: 0,
                    embedded_count: 0,
                    crops: Vec::new(),
                });
            }

            let mut detected_count = 0usize;
            let mut embedded_count = 0usize;
            let mut crops = Vec::new();

            for prepared_face in prepared.faces {
                if store.face_is_tombstoned_for_detection(
                    Some(&prepared.asset_id),
                    None,
                    &prepared_face.face,
                )? {
                    continue;
                }
                let face_id = store.insert_face(
                    Some(&prepared.asset_id),
                    None,
                    &prepared.detector_model,
                    &prepared_face.face,
                    None,
                    prepared_face.embedding.as_ref().map(|embedding| {
                        (embedding.model_name.as_str(), embedding.vector.as_slice())
                    }),
                    prepared_face.recognition.as_ref().map(|recognition| {
                        (
                            recognition.model_name.as_str(),
                            recognition.vector.as_slice(),
                        )
                    }),
                )?;
                if prepared_face.embedding.is_some() {
                    embedded_count += 1;
                }
                if let Some(display_crop_png) = prepared_face.display_crop_png {
                    crops.push(PersistedDetectedFaceCrop {
                        face_id,
                        display_crop_png,
                    });
                }
                detected_count += 1;
            }
            store.note_face_scan_for_asset(&prepared.asset_id, &prepared.detector_model)?;

            Ok(PersistedCorpusFacePass {
                detected_count,
                embedded_count,
                crops,
            })
        })
    }

    fn write_persisted_face_crops(
        &self,
        crops: Vec<PersistedDetectedFaceCrop>,
    ) -> anyhow::Result<()> {
        let mut persisted_paths = Vec::new();
        for crop in crops {
            let crop_path = self.face_crop_path(crop.face_id);
            if let Some(parent) = crop_path.parent() {
                fs::create_dir_all(parent).ok();
            }
            if fs::write(&crop_path, &crop.display_crop_png).is_ok() {
                persisted_paths.push((crop.face_id, crop_path.to_string_lossy().into_owned()));
            }
        }
        if persisted_paths.is_empty() {
            return Ok(());
        }
        self.with_write_store("set_face_aligned_paths", move |store| {
            store.set_face_aligned_paths(&persisted_paths)
        })
    }

    pub(super) fn devour_corpus_face_recognition_backfill_batch(
        &self,
        limit: usize,
    ) -> anyhow::Result<usize> {
        if !self.embedder.recognition_enabled() {
            return Ok(0);
        }

        let detector_model = self.embedder.face_detection_model_name().to_owned();
        let recognition_model = self.embedder.recognition_model_name().to_owned();
        let missing = self.read_store()?.faces_missing_recognition(
            self.active.corpus_id,
            &detector_model,
            &recognition_model,
        )?;
        let mut backfilled = 0usize;

        for face in missing.into_iter().take(limit) {
            let Some(asset_id) = face.asset_id.as_ref() else {
                continue;
            };
            let Some(asset) = self.maybe_image_asset(asset_id)? else {
                continue;
            };
            let Ok(bytes) = fs::read(&asset.path) else {
                continue;
            };
            let Ok(image) = crate::identity::canonical_embedding_image(&bytes) else {
                continue;
            };
            let aligned = align_face_for_embedding(
                &image,
                &DetectedFace {
                    bbox: crate::face::FaceBbox {
                        x: face.bbox_x,
                        y: face.bbox_y,
                        w: face.bbox_w,
                        h: face.bbox_h,
                    },
                    landmarks: face.landmarks.clone(),
                    confidence: face.confidence,
                },
            );
            let Some(recognition) = self.embedder.recognize_face_image(&aligned.crop)? else {
                continue;
            };
            if self.with_write_store("save_face_recognition", move |store| {
                store.save_face_recognition(face.id, &recognition.model_name, &recognition.vector)
            })? {
                backfilled += 1;
            }
        }

        if backfilled > 0 {
            info!(
                faces = backfilled,
                "face recognition backfill batch complete"
            );
        }
        Ok(backfilled)
    }

    pub(super) fn devour_corpus_quality_feature_batch(
        &self,
        limit: usize,
    ) -> anyhow::Result<usize> {
        let store = Store::open_hot(&self.db_path)?;
        let missing = store
            .assets_missing_quality_features(self.active.corpus_id, QUALITY_FEATURE_REVISION)?;
        let mut warmed = Vec::new();
        for asset in missing.into_iter().take(limit) {
            let Ok(bytes) = fs::read(&asset.path) else {
                continue;
            };
            let Ok(features) = extract_asset_quality_features(&bytes) else {
                continue;
            };
            warmed.push(StoredAssetQualityFeatures {
                asset_id: asset.id.clone(),
                extractor_revision: QUALITY_FEATURE_REVISION.to_owned(),
                features,
                updated_at: time::OffsetDateTime::now_utc(),
            });
        }
        if !warmed.is_empty() {
            let warmed_count = warmed.len();
            self.with_write_store("save_asset_quality_features_batch", move |store| {
                store.save_asset_quality_features_batch(&warmed)
            })?;
            info!(
                assets = warmed_count,
                "asset quality feature batch complete"
            );
            return Ok(warmed_count);
        }
        Ok(0)
    }

    pub fn devour_quality_model_refresh(&self) -> anyhow::Result<()> {
        let model_name = self.embedder.model_name().to_owned();
        let active_model = {
            let store = self.read_store()?;
            store.active_quality_model()?
        };
        let log_stats = |stats: &crate::quality::QualityReplayStats| {
            info!(
                formal_version = stats.formal_version.as_str(),
                assets = stats.asset_count,
                sessions = stats.session_count,
                subjects = stats.subject_count,
                comparisons = stats.comparison_events,
                nudges = stats.nudge_events,
                hearts = stats.heart_events,
                "quality model refresh complete"
            );
        };

        if active_model.formal_version
            != crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3
        {
            let result = self.with_write_store("quality_refresh_rebuild", move |store| {
                let stats = store.rebuild_active_quality_state(&model_name)?;
                log_stats(&stats);
                Ok(())
            });
            if result.is_ok() {
                self.invalidate_session_field_cache();
            }
            return result;
        }

        for attempt in 1..=4 {
            let prepared = {
                let store = Store::open_hot(&self.db_path)?;
                store.prepare_perturbative_replay_v3(&model_name)?
            };
            let maybe_stats = self.with_write_store("quality_refresh_commit", move |store| {
                store.commit_prepared_perturbative_replay(prepared)
            })?;
            if let Some(stats) = maybe_stats {
                log_stats(&stats);
                self.invalidate_session_field_cache();
                return Ok(());
            }
            warn!(
                attempt,
                "quality refresh snapshot went stale before commit; retrying"
            );
        }

        warn!("quality refresh kept racing newer events; requeueing instead of in-gate rebuild");
        self.schedule_quality_model_refresh();
        Ok(())
    }
}

fn encode_display_crop_png(image: &image::DynamicImage, face: &DetectedFace) -> Option<Vec<u8>> {
    let display_crop = align_face_for_display(image, face);
    let mut png = Vec::new();
    display_crop
        .crop
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .ok()?;
    Some(png)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ArenaConfig, ImportPolicy, LocalDirectorySource, RemoteImageFilterConfig, SourceConfig,
        UpstreamSource,
    };
    use crate::model::ExternalEventKind;
    use std::sync::{Mutex as StdMutex, OnceLock};

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<StdMutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .expect("lock lifecycle test guard")
    }

    fn test_root(name: &str) -> PathBuf {
        let salt = time::OffsetDateTime::now_utc().unix_timestamp_nanos();
        let base = std::env::var_os("CARGO_TARGET_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/test-tmp")
                    .canonicalize()
                    .unwrap_or_else(|_| {
                        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp")
                    })
            });
        let root = base.join(format!("picmash-lifecycle-{name}-{salt}"));
        std::fs::create_dir_all(&base).expect("create lifecycle test base root");
        if root.exists() {
            std::fs::remove_dir_all(&root).expect("clear old lifecycle test root");
        }
        std::fs::create_dir_all(&root).expect("create lifecycle test root");
        root
    }

    fn solid_png(path: &Path, rgb: [u8; 3]) {
        let image = image::RgbImage::from_fn(96, 96, |_x, _y| image::Rgb(rgb));
        image.save(path).expect("write test png");
    }

    fn app_config_with_source_mix(external_probability: f32) -> AppConfig {
        AppConfig {
            arena: ArenaConfig {
                external_probability,
                explore: 0.0,
                dedup_radius: 0.0,
            },
            ..AppConfig::default()
        }
    }

    fn drain_maintenance(state: &AppState) {
        while state
            .devour_one_maintenance_job()
            .expect("maintenance job should run")
        {}
    }

    fn source_remote_items(state: &AppState, source_key: &str) -> Vec<RemoteItemId> {
        let store = state
            .read_store()
            .expect("open read store for lifecycle test");
        (1..=64)
            .map(RemoteItemId)
            .filter(|item_id| {
                store
                    .remote_item(*item_id)
                    .expect("load remote item during lifecycle test")
                    .is_some_and(|item| item.source_key == source_key)
            })
            .collect()
    }

    #[test]
    fn write_gate_writes_do_not_wait_on_shared_store_read_lock() {
        let _guard = test_guard();
        let root = test_root("write-gate-fresh-write");
        let corpus_root = root.join("corpus");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed.png"), [32, 48, 64]);

        let config = AppConfig::default();
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state = std::sync::Arc::new(
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state"),
        );
        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        let read_store = state
            .read_store()
            .expect("open read store for write-path lifecycle test");
        let worker = std::sync::Arc::clone(&state);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let session_id = worker.active.session_id;
            let result = worker.with_write_store("touch_session", move |store| {
                store.touch_session(session_id)
            });
            tx.send(result.map(|_| ())).expect("send write result");
        });

        let result = rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("fresh write should not wait on shared store read lock");
        drop(read_store);
        result.expect("touch session through fresh write store");
    }

    #[test]
    fn cold_start_local_directory_accepts_images_one_by_one() {
        let _guard = test_guard();
        let root = test_root("cold-start-local-directory");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed.png"), [32, 48, 64]);

        let mut config = app_config_with_source_mix(1.0);
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root.clone(),
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");

        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        assert_eq!(
            state
                .read_store()
                .expect("open read store for seed corpus count")
                .corpus_assets(state.active.corpus_id)
                .expect("seed corpus assets")
                .len(),
            1,
            "seed corpus should ingest on cold start"
        );

        solid_png(&source_root.join("remote-a.png"), [180, 40, 60]);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest first local-directory image");
        let first_remote = source_remote_items(&state, &source_key)
            .into_iter()
            .next()
            .expect("first harvested remote item");
        state
            .seal_imported_external_outcome(first_remote, ExternalEventKind::RemoteWin)
            .expect("accept first remote");
        state.schedule_quality_model_refresh();
        drain_maintenance(&state);
        assert_eq!(
            state
                .read_store()
                .expect("open read store after first accept")
                .corpus_assets(state.active.corpus_id)
                .expect("assets after first accept")
                .len(),
            2,
            "first remote should import into the corpus"
        );

        solid_png(&source_root.join("remote-b.png"), [40, 160, 90]);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest second local-directory image");
        let second_remote = source_remote_items(&state, &source_key)
            .into_iter()
            .find(|item_id| *item_id != first_remote)
            .expect("second harvested remote item");
        assert_ne!(
            first_remote, second_remote,
            "second harvested remote should be the newly added source image"
        );
        state
            .seal_imported_external_outcome(second_remote, ExternalEventKind::RemoteWin)
            .expect("accept second remote");
        state.schedule_quality_model_refresh();
        drain_maintenance(&state);
        assert_eq!(
            state
                .read_store()
                .expect("open read store after second accept")
                .corpus_assets(state.active.corpus_id)
                .expect("assets after second accept")
                .len(),
            3,
            "second remote should also import into the corpus"
        );
    }

    #[test]
    fn nonforced_local_directory_refresh_is_deferred_to_maintenance() {
        let _guard = test_guard();
        let root = test_root("deferred-local-directory-refresh");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed.png"), [32, 48, 64]);
        solid_png(&source_root.join("remote-a.png"), [180, 40, 60]);

        let mut config = AppConfig::default();
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        state
            .refresh_external_sources_if_due(false)
            .expect("queue local-directory refresh");

        assert!(
            state
                .read_store()
                .expect("open read store for local refresh job check")
                .has_pending_maintenance_job(
                    crate::maintenance::MaintenanceJobKind::LocalDirectoryRefresh,
                    &source_key,
                )
                .expect("check local refresh job"),
            "local-directory refresh should queue a maintenance job"
        );
        assert!(
            source_remote_items(&state, &source_key).is_empty(),
            "local-directory refresh should not harvest inline on the caller path"
        );

        drain_maintenance(&state);

        assert_eq!(
            source_remote_items(&state, &source_key).len(),
            1,
            "queued maintenance should eventually harvest the local-directory item"
        );
    }

    #[test]
    fn imported_remote_handles_are_not_live_in_arena() {
        let _guard = test_guard();
        let root = test_root("imported-remote-stale-handle");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed.png"), [32, 48, 64]);
        solid_png(&source_root.join("remote-a.png"), [180, 40, 60]);

        let mut config = AppConfig::default();
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let local_asset = state
            .read_store()
            .expect("open read store for local asset")
            .corpus_assets(state.active.corpus_id)
            .expect("load corpus assets")
            .into_iter()
            .next()
            .expect("seed asset present");
        let remote_item = source_remote_items(&state, &source_key)
            .into_iter()
            .next()
            .expect("remote item present");
        let remote_handle = ArenaHandle::Remote(remote_item);
        assert!(
            state
                .arena_handle_is_live(&remote_handle)
                .expect("remote handle should start live"),
            "harvested remote should be arena-live before import"
        );

        state
            .seal_imported_external_outcome(remote_item, ExternalEventKind::RemoteWin)
            .expect("accept remote");

        assert!(
            !state
                .arena_handle_is_live(&remote_handle)
                .expect("reload imported remote liveness"),
            "imported remote handle should become stale immediately"
        );
        assert!(
            state
                .arena_pair(&ArenaHandle::Local(local_asset.id), &remote_handle)
                .expect("load arena pair with stale imported remote")
                .is_none(),
            "arena should refuse to render imported remote handles"
        );
    }

    #[test]
    fn remote_win_retires_handle_before_deferred_import_finalizes() {
        let _guard = test_guard();
        let root = test_root("remote-win-pending-import");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed.png"), [32, 48, 64]);
        solid_png(&source_root.join("remote-a.png"), [180, 40, 60]);

        let mut config = AppConfig::default();
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let local_asset = state
            .read_store()
            .expect("open read store for local asset")
            .corpus_assets(state.active.corpus_id)
            .expect("load corpus assets")
            .into_iter()
            .next()
            .expect("seed asset present");
        let remote_item = source_remote_items(&state, &source_key)
            .into_iter()
            .next()
            .expect("remote item present");
        let local_handle = ArenaHandle::Local(local_asset.id);
        let remote_handle = ArenaHandle::Remote(remote_item);

        state
            .vote(&local_handle, &remote_handle, &remote_handle)
            .expect("queue remote import");

        let store = state
            .read_store()
            .expect("open read store after queuing remote import");
        assert!(
            store
                .external_item_import_pending(remote_item)
                .expect("load pending import bit"),
            "remote win should retire the remote immediately under import_pending"
        );
        assert!(
            store
                .pending_external_import_outcome(remote_item)
                .expect("load pending external import row")
                .is_some(),
            "remote win should enqueue a deferred import outcome"
        );
        assert_eq!(
            store
                .external_item_resolved_asset_id(remote_item)
                .expect("load resolved asset before finalize"),
            None,
            "deferred import should not fabricate the imported asset before maintenance runs"
        );
        assert!(
            !state
                .arena_handle_is_live(&remote_handle)
                .expect("reload pending remote liveness"),
            "pending import should already make the remote arena-stale"
        );

        for _ in 0..8 {
            if !state
                .read_store()
                .expect("open read store in maintenance loop")
                .external_item_import_pending(remote_item)
                .expect("reload pending state in maintenance loop")
            {
                break;
            }
            assert!(
                state
                    .devour_one_maintenance_job()
                    .expect("run one maintenance job"),
                "pending import should leave a maintenance job to devour"
            );
        }

        let store = state
            .read_store()
            .expect("open read store after deferred finalize");
        assert!(
            !store
                .external_item_import_pending(remote_item)
                .expect("load cleared pending import bit"),
            "deferred finalize should clear import_pending"
        );
        assert!(
            store
                .external_item_resolved_asset_id(remote_item)
                .expect("load resolved asset after finalize")
                .is_some(),
            "deferred finalize should eventually resolve the imported asset"
        );
    }

    #[test]
    fn subsource_lock_overrides_source_mix_until_stream_exhaustion() {
        let _guard = test_guard();
        let root = test_root("subsource-lock");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
        std::fs::create_dir_all(source_root.join("b")).expect("create source stream b");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&source_root.join("a").join("remote-a.png"), [180, 40, 60]);
        solid_png(&source_root.join("b").join("remote-b.png"), [40, 160, 90]);

        let mut config = app_config_with_source_mix(0.0);
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let mut remote_items = source_remote_items(&state, &source_key)
            .into_iter()
            .map(|item_id| {
                let item = state
                    .read_store()
                    .expect("open read store for locked remote item")
                    .remote_item(item_id)
                    .expect("load remote item")
                    .expect("remote item present");
                (item_id, item.stream_id)
            })
            .collect::<Vec<_>>();
        remote_items.sort_by_key(|(_, stream_id)| *stream_id);
        let (locked_item_id, locked_stream_id) = remote_items[0];

        state
            .set_external_subsource_lock(locked_item_id, true)
            .expect("lock subsource");

        let locked_target = state.arena_target().expect("arena target under lock");
        let RedirectTarget::ArenaPair { left, right } = locked_target else {
            panic!("expected arena pair under lock");
        };
        let locked_remote_id = match (&left, &right) {
            (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
            | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => *item_id,
            _ => panic!("expected local-vs-remote pair under lock"),
        };
        let locked_remote = state
            .read_store()
            .expect("open read store for locked remote reload")
            .remote_item(locked_remote_id)
            .expect("reload locked remote")
            .expect("locked remote present");
        assert_eq!(
            locked_remote.stream_id, locked_stream_id,
            "locked stream should override external/local mix even at zero external probability"
        );

        state
            .with_write_store("block_external_stream", move |store| {
                store.block_external_stream(
                    state.active.session_id,
                    state.active.corpus_id,
                    locked_item_id,
                    None,
                )
            })
            .expect("block locked stream");

        let unlocked_target = state
            .arena_target()
            .expect("arena target after exhausting locked stream");
        let RedirectTarget::ArenaPair { left, right } = unlocked_target else {
            panic!("expected arena pair after exhausting lock");
        };
        assert!(
            matches!(
                (&left, &right),
                (ArenaHandle::Local(_), ArenaHandle::Local(_))
            ),
            "after lock exhaustion the chooser should fall back to the configured local-only mix"
        );
        assert_eq!(
            state
                .read_store()
                .expect("open read store for cleared lock reload")
                .session_subsource_lock(state.active.session_id)
                .expect("reload cleared lock"),
            None,
            "exhausted subsource lock should clear itself"
        );
    }

    #[test]
    fn locking_subsource_command_flushes_pipeline_without_rerolling_current_pair() {
        let _guard = test_guard();
        let root = test_root("subsource-lock-current-pair");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&source_root.join("a").join("remote-a.png"), [180, 40, 60]);

        let mut config = app_config_with_source_mix(1.0);
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let page = state.arena_page_state(None).expect("arena page state");
        let current = page.current.expect("current turn before lock");
        let old_epoch = current.sampler_epoch();
        let remote_handle = match (&current.pair().left, &current.pair().right) {
            (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
            | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => {
                ArenaHandle::Remote(*item_id)
            }
            _ => panic!("expected local-vs-remote pair before lock"),
        };

        let outcome = state
            .apply_arena_command(ArenaCommand::LockThread {
                command_id: ArenaCommandId::forge(),
                expected_revision: current.revision(),
                expected_sampler_epoch: current.sampler_epoch(),
                turn_id: current.id().clone(),
                action_token: current.action_token().clone(),
                handle: remote_handle,
                active: true,
            })
            .expect("lock current remote subsource through command");
        assert_eq!(outcome.status, ArenaCommandStatus::Applied);
        let locked = outcome.current.expect("current turn after lock");
        assert_eq!(
            locked.id(),
            current.id(),
            "locking should preserve the active turn id"
        );
        assert_eq!(
            &locked.pair().left,
            &current.pair().left,
            "locking should not advance the left handle"
        );
        assert_eq!(
            &locked.pair().right,
            &current.pair().right,
            "locking should not advance the right handle"
        );
        assert_eq!(
            locked.sampler_epoch().0,
            old_epoch.0 + 1,
            "locking is an immediate sampler transition"
        );
        assert!(
            state
                .read_store()
                .expect("open read store for subsource lock reload")
                .session_subsource_lock(state.active.session_id)
                .expect("reload subsource lock")
                .is_some(),
            "locking should persist the session subsource lock"
        );

        let post_lock_vote = state
            .apply_arena_command(ArenaCommand::Vote {
                command_id: ArenaCommandId::forge(),
                expected_revision: locked.revision(),
                expected_sampler_epoch: locked.sampler_epoch(),
                turn_id: locked.id().clone(),
                action_token: locked.action_token().clone(),
                winner: locked.pair().left.clone(),
            })
            .expect("vote preserved current after lock");
        assert_eq!(
            post_lock_vote.status,
            ArenaCommandStatus::Applied,
            "the preserved turn should remain commandable in its new sampler epoch"
        );
    }

    #[test]
    fn arena_prefetch_excludes_visual_keys_without_replacement() {
        let _guard = test_guard();
        let root = test_root("arena-prefetch-excluded-visual-keys");
        let corpus_root = root.join("corpus");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&corpus_root.join("seed-c.png"), [180, 40, 60]);

        let config = app_config_with_source_mix(0.0);
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        let assets = state
            .read_store()
            .expect("open read store for local assets")
            .corpus_assets(state.active.corpus_id)
            .expect("load corpus assets");
        let excluded_visual_key = assets[0]
            .visual_key
            .clone()
            .expect("ingested local asset should have visual key");
        let excluded = std::collections::HashSet::from([excluded_visual_key.clone()]);

        let target = state
            .arena_prefetch_target_excluding(&excluded)
            .expect("prefetch target excluding visual key");
        let RedirectTarget::ArenaPair { left, right } = target else {
            panic!("expected local arena pair after excluding one visual identity");
        };
        let Some(view) = state
            .arena_pair(&left, &right)
            .expect("load excluded prefetch pair")
        else {
            panic!("expected rendered arena pair");
        };
        let pair = view.pair.expect("prefetch pair present");
        assert!(
            !pair.visual_keys().contains(&excluded_visual_key),
            "prefetch without replacement should exclude already-buffered visual identities"
        );
    }

    #[test]
    fn arena_recent_visual_buffer_excludes_last_seen_local_pair() {
        let _guard = test_guard();
        let root = test_root("arena-recent-visual-buffer");
        let corpus_root = root.join("corpus");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        for (name, rgb) in [
            ("seed-a.png", [32, 48, 64]),
            ("seed-b.png", [64, 48, 32]),
            ("seed-c.png", [180, 40, 60]),
            ("seed-d.png", [20, 160, 120]),
        ] {
            solid_png(&corpus_root.join(name), rgb);
        }

        let config = app_config_with_source_mix(0.0);
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        let initial_target = state.arena_target().expect("initial arena target");
        let RedirectTarget::ArenaPair { left, right } = initial_target else {
            panic!("expected initial local arena pair");
        };
        assert!(
            matches!(
                (&left, &right),
                (ArenaHandle::Local(_), ArenaHandle::Local(_))
            ),
            "local-only arena should serve a local pair"
        );
        let seen_pair = state
            .arena_pair(&left, &right)
            .expect("load initial arena pair")
            .expect("rendered initial arena pair")
            .pair
            .expect("initial arena pair payload");
        let seen_visual_keys = seen_pair.visual_keys();
        assert_eq!(
            seen_visual_keys.len(),
            2,
            "a local duel should expose two distinct visual identities"
        );

        let next_target = state.vote(&left, &right, &left).expect("vote local pair");
        let RedirectTarget::ArenaPair {
            left: next_left,
            right: next_right,
        } = next_target
        else {
            panic!("expected next local arena pair after voting");
        };
        let next_pair = state
            .arena_pair(&next_left, &next_right)
            .expect("load next arena pair")
            .expect("rendered next arena pair")
            .pair
            .expect("next arena pair payload");
        assert!(
            next_pair.visual_keys().is_disjoint(&seen_visual_keys),
            "the rolling recent-visual buffer should exclude the just-seen pair"
        );
    }

    #[test]
    fn arena_command_promotes_issued_lookahead_once() {
        let _guard = test_guard();
        let root = test_root("arena-command-promotes-lookahead");
        let corpus_root = root.join("corpus");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        for (name, rgb) in [
            ("seed-a.png", [32, 48, 64]),
            ("seed-b.png", [64, 48, 32]),
            ("seed-c.png", [180, 40, 60]),
            ("seed-d.png", [20, 160, 120]),
        ] {
            solid_png(&corpus_root.join(name), rgb);
        }

        let config = app_config_with_source_mix(0.0);
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };
        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);

        let page = state.arena_page_state(None).expect("arena page state");
        let current = page.current.expect("current turn");
        let lookahead = page.lookahead.expect("lookahead turn");
        let winner = current.pair().left.clone();
        let command = ArenaCommand::Vote {
            command_id: ArenaCommandId::forge(),
            expected_revision: current.revision(),
            expected_sampler_epoch: current.sampler_epoch(),
            turn_id: current.id().clone(),
            action_token: current.action_token().clone(),
            winner,
        };

        let applied = state
            .apply_arena_command(command.clone())
            .expect("apply arena command");
        assert_eq!(applied.status, ArenaCommandStatus::Applied);
        let promoted = applied.current.expect("promoted current turn");
        assert_eq!(
            promoted.id(),
            lookahead.id(),
            "a hot vote should promote the server-issued lookahead"
        );
        assert_eq!(
            promoted.revision().0,
            current.revision().0 + 1,
            "the promoted turn should be activated at the next revision"
        );

        let comparisons_after_apply = state
            .read_store()
            .expect("open store after apply")
            .session(state.active.session_id)
            .expect("load session after apply")
            .comparisons;
        assert_eq!(comparisons_after_apply, 1);

        let replayed = state
            .apply_arena_command(command)
            .expect("replay arena command");
        assert_eq!(replayed.status, ArenaCommandStatus::Replayed);
        let comparisons_after_replay = state
            .read_store()
            .expect("open store after replay")
            .session(state.active.session_id)
            .expect("load session after replay")
            .comparisons;
        assert_eq!(
            comparisons_after_replay, comparisons_after_apply,
            "command id replay must not duplicate the vote"
        );

        let stale_outcome = state
            .apply_arena_command(ArenaCommand::Vote {
                command_id: ArenaCommandId::forge(),
                expected_revision: current.revision(),
                expected_sampler_epoch: current.sampler_epoch(),
                turn_id: current.id().clone(),
                action_token: current.action_token().clone(),
                winner: current.pair().left.clone(),
            })
            .expect("reject stale arena command");
        assert_eq!(stale_outcome.status, ArenaCommandStatus::Stale);

        let promoted_epoch = promoted.sampler_epoch();
        state.apply_arena_sampler_invalidation(SamplerInvalidation::Immediate);
        let old_epoch_outcome = state
            .apply_arena_command(ArenaCommand::Vote {
                command_id: ArenaCommandId::forge(),
                expected_revision: promoted.revision(),
                expected_sampler_epoch: promoted_epoch,
                turn_id: promoted.id().clone(),
                action_token: promoted.action_token().clone(),
                winner: promoted.pair().left.clone(),
            })
            .expect("reject command from invalidated sampler epoch");
        assert_eq!(old_epoch_outcome.status, ArenaCommandStatus::Stale);

        let refreshed = state
            .arena_page_state(None)
            .expect("refresh arena after sampler invalidation")
            .current
            .expect("current turn after sampler invalidation");
        assert_eq!(
            refreshed.sampler_epoch().0,
            promoted_epoch.0 + 1,
            "immediate sampler invalidation should hard-fence old issued turns"
        );

        state.apply_arena_sampler_invalidation(SamplerInvalidation::Eventual);
        let preserved = state
            .arena_page_state(None)
            .expect("refresh arena after eventual sampler change")
            .current
            .expect("current turn after eventual sampler change");
        assert_eq!(
            preserved.id(),
            refreshed.id(),
            "eventual sampler changes should not discard issued turns"
        );
        assert_eq!(preserved.sampler_epoch(), refreshed.sampler_epoch());
    }

    #[test]
    fn arena_hide_command_promotes_issued_lookahead_after_retiring_clicked_remote() {
        let _guard = test_guard();
        let root = test_root("arena-command-remote-hide-retires");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&source_root.join("remote-a.png"), [180, 40, 60]);
        solid_png(&source_root.join("remote-b.png"), [40, 180, 60]);

        let mut config = app_config_with_source_mix(1.0);
        config.sources = vec![SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        }];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let page = state.arena_page_state(None).expect("arena page state");
        let current = page.current.expect("current turn");
        let lookahead = page.lookahead.expect("issued lookahead turn");
        let remote_handle = match (&current.pair().left, &current.pair().right) {
            (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
            | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => {
                ArenaHandle::Remote(*item_id)
            }
            _ => panic!("expected a remote arena pair"),
        };
        let ArenaHandle::Remote(remote_item_id) = remote_handle else {
            panic!("expected remote handle")
        };

        let outcome = state
            .apply_arena_command(ArenaCommand::Hide {
                command_id: ArenaCommandId::forge(),
                expected_revision: current.revision(),
                expected_sampler_epoch: current.sampler_epoch(),
                turn_id: current.id().clone(),
                action_token: current.action_token().clone(),
                handle: remote_handle.clone(),
                hidden: true,
                cluster_ids: Vec::new(),
            })
            .expect("hide remote through arena command");
        assert_eq!(outcome.status, ArenaCommandStatus::Applied);
        assert!(
            state
                .read_store()
                .expect("open store after remote hide")
                .external_item_hidden(remote_item_id)
                .expect("reload hidden remote"),
            "remote hide must retire the clicked item before selecting the next turn"
        );
        assert!(
            outcome
                .current
                .as_ref()
                .is_none_or(|turn| !turn.pair().contains(&remote_handle)),
            "hide must not promote a turn containing the just-rejected remote"
        );
        assert_eq!(
            outcome.current.as_ref().map(|turn| turn.id()),
            Some(lookahead.id()),
            "hide should consume the same issued pipeline as vote when the pipeline survives the hide effect"
        );
    }

    #[test]
    fn vetoing_a_locked_thread_lifts_the_lock_before_rerolling() {
        let _guard = test_guard();
        let root = test_root("subsource-lock-veto-clears");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
        std::fs::create_dir_all(source_root.join("b")).expect("create source stream b");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&source_root.join("a").join("remote-a.png"), [180, 40, 60]);
        solid_png(&source_root.join("b").join("remote-b.png"), [40, 160, 90]);

        let mut config = app_config_with_source_mix(0.0);
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let mut remote_items = source_remote_items(&state, &source_key)
            .into_iter()
            .map(|item_id| {
                let item = state
                    .read_store()
                    .expect("open read store for locked remote item")
                    .remote_item(item_id)
                    .expect("load remote item")
                    .expect("remote item present");
                (item_id, item.stream_id)
            })
            .collect::<Vec<_>>();
        remote_items.sort_by_key(|(_, stream_id)| *stream_id);
        let (locked_item_id, locked_stream_id) = remote_items[0];

        state
            .set_external_subsource_lock(locked_item_id, true)
            .expect("lock subsource");

        let locked_target = state.arena_target().expect("arena target under lock");
        let RedirectTarget::ArenaPair { left, right } = locked_target else {
            panic!("expected arena pair under lock");
        };
        let remote_handle = match (&left, &right) {
            (ArenaHandle::Local(_), ArenaHandle::Remote(item_id))
            | (ArenaHandle::Remote(item_id), ArenaHandle::Local(_)) => {
                ArenaHandle::Remote(*item_id)
            }
            _ => panic!("expected local-vs-remote pair under lock"),
        };
        let locked_remote = state
            .read_store()
            .expect("open read store for locked remote reload")
            .remote_item(locked_item_id)
            .expect("reload locked remote")
            .expect("locked remote present");
        assert_eq!(
            locked_remote.stream_id, locked_stream_id,
            "expected the locked stream to drive the current remote candidate"
        );

        let unlocked_target = state
            .veto_external_thread_for_handle(&remote_handle, &left, &right)
            .expect("veto locked thread");
        let RedirectTarget::ArenaPair { left, right } = unlocked_target else {
            panic!("expected arena pair after vetoing locked thread");
        };
        assert!(
            matches!(
                (&left, &right),
                (ArenaHandle::Local(_), ArenaHandle::Local(_))
            ),
            "vetoing the locked thread should behave like unlocking before rerolling"
        );
        assert_eq!(
            state
                .read_store()
                .expect("open read store for cleared lock reload")
                .session_subsource_lock(state.active.session_id)
                .expect("reload cleared lock"),
            None,
            "vetoing the locked thread should clear the active subsource lock"
        );
    }

    #[test]
    fn subsource_lock_prefetch_does_not_clear_on_speculative_exhaustion() {
        let _guard = test_guard();
        let root = test_root("subsource-lock-prefetch");
        let corpus_root = root.join("corpus");
        let source_root = root.join("source");
        let config_root = root.join("config");
        let app_data_root = root.join("xdg-data");
        let app_cache_root = root.join("xdg-cache");
        std::fs::create_dir_all(&corpus_root).expect("create corpus root");
        std::fs::create_dir_all(source_root.join("a")).expect("create source stream a");
        std::fs::create_dir_all(&config_root).expect("create config root");
        std::fs::create_dir_all(&app_data_root).expect("create data root");
        std::fs::create_dir_all(&app_cache_root).expect("create cache root");

        solid_png(&corpus_root.join("seed-a.png"), [32, 48, 64]);
        solid_png(&corpus_root.join("seed-b.png"), [64, 48, 32]);
        solid_png(&source_root.join("a").join("remote-a.png"), [180, 40, 60]);

        let mut config = app_config_with_source_mix(0.0);
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 0,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root: source_root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 0,
                    ..RemoteImageFilterConfig::default()
                },
            }),
        };
        let source_key = source.source_key();
        config.sources = vec![source];
        let config_path = config_root.join("config.toml");
        let config_digest = config.write(&config_path).expect("write config");
        let app_paths = AppBootPaths {
            db_path: app_data_root.join("picmash.sqlite3"),
            model_cache_root: app_cache_root.clone(),
            cache_root: app_cache_root.join("renditions"),
            source_cache_root: app_cache_root.join("sources"),
        };

        let state =
            AppState::boot_with_paths(&corpus_root, config, config_path, config_digest, app_paths)
                .expect("boot app state");
        state.schedule_corpus_ingest();
        drain_maintenance(&state);
        state
            .refresh_external_sources_if_due(true)
            .expect("harvest local-directory source");

        let locked_item_id = source_remote_items(&state, &source_key)[0];
        state
            .set_external_subsource_lock(locked_item_id, true)
            .expect("lock subsource");

        let locked_target = state.arena_target().expect("arena target under lock");
        let RedirectTarget::ArenaPair { left, right } = locked_target else {
            panic!("expected locked arena pair");
        };
        let local_anchor = match (&left, &right) {
            (ArenaHandle::Local(asset_id), ArenaHandle::Remote(_))
            | (ArenaHandle::Remote(_), ArenaHandle::Local(asset_id)) => asset_id.clone(),
            _ => panic!("expected local-vs-remote pair under lock"),
        };

        let prefetch = state
            .arena_prefetch_target_preserving_local_anchor(Some(&local_anchor))
            .expect("prefetch target under exhausted lock");
        assert!(
            matches!(prefetch, RedirectTarget::ArenaRoot),
            "speculative exhausted prefetch should decline to invent an unlocked pair"
        );
        assert!(
            state
                .read_store()
                .expect("open read store for preserved lock reload")
                .session_subsource_lock(state.active.session_id)
                .expect("reload preserved lock")
                .is_some(),
            "speculative prefetch must not clear an exhausted subsource lock"
        );
    }
}
