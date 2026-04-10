use super::*;

impl AppState {
    pub fn home_target(&self) -> anyhow::Result<RedirectTarget> {
        self.arena_target()
    }

    pub fn config_reload_pulse(&self) -> Duration {
        CONFIG_RELOAD_PULSE
    }

    pub fn external_scan_pulse(&self) -> Duration {
        EXTERNAL_SCAN_PULSE
    }

    pub fn quality_refresh_is_inline(&self) -> anyhow::Result<bool> {
        let store = self.read_store()?;
        Ok(matches!(
            store.active_quality_model()?.formal_version,
            QualityFormalVersion::LegacyIndependentV1
                | QualityFormalVersion::HierarchicalGaussianV1
        ))
    }

    pub fn reload_config_if_changed(&self) -> anyhow::Result<()> {
        let raw = match fs::read_to_string(&self.config_path) {
            Ok(raw) => raw,
            Err(error) => {
                warn!(
                    path = %self.config_path.display(),
                    error = %format!("{error:#}"),
                    "live config reload skipped"
                );
                return Ok(());
            }
        };
        let digest = blake3::hash(raw.as_bytes()).to_hex().to_string();
        {
            let reload = self.config_reload.lock();
            if reload.is_live_digest(&digest) {
                return Ok(());
            }
        }
        let (mut candidate, _) = match AppConfig::parse(&raw, &self.config_path) {
            Ok(parsed) => parsed,
            Err(error) => {
                let mut reload = self.config_reload.lock();
                if reload.should_warn_rejected(&digest) {
                    warn!(
                        path = %self.config_path.display(),
                        error = %format!("{error:#}"),
                        "ignoring invalid live config update"
                    );
                    reload.note_rejected(digest);
                }
                return Ok(());
            }
        };

        let current_runtime = self.config.read().runtime.clone();
        if candidate.runtime.bind_addr != current_runtime.bind_addr
            || candidate.runtime.corpus_root != current_runtime.corpus_root
        {
            warn!(
                path = %self.config_path.display(),
                "ignoring live changes to runtime config; restart required for bind_addr/corpus_root"
            );
            candidate.runtime = current_runtime;
        }

        *self.config.write() = candidate;
        self.config_reload.lock().note_applied(digest);
        info!(path = %self.config_path.display(), "reloaded live config");
        Ok(())
    }

    pub(super) fn persist_live_config(&self, config: &AppConfig) -> anyhow::Result<()> {
        let digest = config.write(&self.config_path)?;
        self.config_reload.lock().note_applied(digest);
        Ok(())
    }

    pub fn external_status(&self) -> anyhow::Result<ExternalArenaStatus> {
        let config = self.config.read().clone();
        let sources = config
            .sources
            .iter()
            .map(|source| crate::model::ExternalSourceOption {
                source_key: source.source_key(),
                label: source.display_name(),
                weight: source.weight,
            })
            .collect::<Vec<_>>();
        let store = self.read_store()?;
        let (active_streams, blocked_streams, cached_items) =
            config
                .sources
                .iter()
                .try_fold((0usize, 0usize, 0usize), |counts, source| {
                    let (active, blocked, cached) =
                        store.external_source_counts(&source.source_key())?;
                    Ok::<_, anyhow::Error>((
                        counts.0 + active,
                        counts.1 + blocked,
                        counts.2 + cached,
                    ))
                })?;
        Ok(ExternalArenaStatus {
            sources,
            external_probability: (config.external_probability() * 100.0).round() as u8,
            arena_explore_percent: (config.arena_explore() * 100.0).round() as u8,
            active_streams,
            blocked_streams,
            cached_items,
            dedup_radius_percent: (config.dedup_radius() * 100.0).round() as u8,
        })
    }

    pub fn set_external_probability_percent(&self, percent: u8) -> anyhow::Result<()> {
        let probability = f32::from(percent) / 100.0;
        let snapshot = {
            let mut config = self.config.write();
            config.shove_external_probability(probability);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(
            external_probability = probability,
            "updated external sampling probability"
        );
        Ok(())
    }

    pub fn arena_explore(&self) -> f32 {
        self.config.read().arena_explore()
    }

    pub fn set_arena_explore_percent(&self, percent: u8) -> anyhow::Result<()> {
        let explore = f32::from(percent) / 100.0;
        let snapshot = {
            let mut config = self.config.write();
            config.shove_arena_explore(explore);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(
            arena_explore = explore,
            "updated arena exploration pressure"
        );
        Ok(())
    }

    pub fn refresh_external_sources_if_due(&self, force: bool) -> anyhow::Result<()> {
        let mut due_sources = Vec::new();
        let mut due_local_sources = Vec::new();
        let store = self.read_store()?;
        for source in self.configured_sources() {
            let source_key = source.source_key();
            let due = force
                || store.external_scan_due(
                    &source_key,
                    Duration::seconds(source.scan_interval_seconds as i64),
                )?;
            if !due {
                continue;
            }
            if !force && source.local_directory().is_some() {
                due_local_sources.push(source_key);
                continue;
            }
            if !force && self.remote_source_scan_can_rest(&store, &source)? {
                info!(source = %source_key, "remote source warm buffer saturated; skipping refresh");
                continue;
            }
            due_sources.push(source);
        }
        drop(store);
        for source_key in due_local_sources {
            self.schedule_local_directory_refresh(&source_key);
        }
        if due_sources.is_empty() {
            return Ok(());
        }

        let scanner = self.source_scanner.clone();
        let harvests = thread::scope(|scope| {
            #[allow(clippy::needless_collect)]
            let handles = due_sources
                .into_iter()
                .map(|source| {
                    let scanner = scanner.clone();
                    let source_key = source.source_key();
                    info!(source = %source_key, force, "refreshing external source");
                    scope.spawn(move || {
                        let result = scanner.harvest(&source);
                        (source, result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("external source worker panicked"))
                })
                .map(|handle| handle.map_err(anyhow::Error::from))
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let mut harvests = harvests;
        harvests.sort_by(|(lhs_source, lhs_harvest), (rhs_source, rhs_harvest)| {
            let lhs_local = lhs_source.local_directory().is_some();
            let rhs_local = rhs_source.local_directory().is_some();
            rhs_local
                .cmp(&lhs_local)
                .then_with(|| {
                    let lhs_streams = lhs_harvest
                        .as_ref()
                        .ok()
                        .map(|harvest| harvest.streams.len())
                        .unwrap_or(usize::MAX);
                    let rhs_streams = rhs_harvest
                        .as_ref()
                        .ok()
                        .map(|harvest| harvest.streams.len())
                        .unwrap_or(usize::MAX);
                    lhs_streams.cmp(&rhs_streams)
                })
                .then_with(|| lhs_source.source_key().cmp(&rhs_source.source_key()))
        });

        for (source, harvest) in harvests {
            let source_key = source.source_key();
            if let Err(error) =
                harvest.and_then(|harvest| self.devour_external_harvest(&source, &harvest))
            {
                let message = format!("{error:#}");
                warn!(source = %source_key, error = %message, "external source scan failed");
                let source_key = source_key.clone();
                self.with_write_store("external_scan_fault", move |store| {
                    store.external_scan_fault(&source_key, &message)
                })?;
            }
        }
        Ok(())
    }

    pub fn close(&self) -> anyhow::Result<()> {
        let session_id = self.active.session_id;
        self.with_write_store("close_session", move |store| {
            store.close_session(session_id)
        })
    }

    pub(super) fn source_config_for_key(&self, source_key: &str) -> Option<SourceConfig> {
        self.config
            .read()
            .sources
            .iter()
            .find(|source| source.source_key() == source_key)
            .cloned()
    }

    pub(super) fn configured_sources(&self) -> Vec<SourceConfig> {
        self.config.read().sources.clone()
    }

    pub(super) fn asset_domain_oracle(&self, store: &Store) -> anyhow::Result<AssetDomainOracle> {
        if let Some(oracle) = self.asset_domain_oracle.read().as_ref().cloned() {
            return Ok(oracle);
        }
        let oracle = AssetDomainOracle::train(
            &store.asset_domain_training_rows(self.active.corpus_id, self.embedder.model_name())?,
        );
        *self.asset_domain_oracle.write() = Some(oracle.clone());
        Ok(oracle)
    }

    pub(super) fn retrain_asset_domain_oracle(
        &self,
        store: &Store,
    ) -> anyhow::Result<AssetDomainOracle> {
        let oracle = AssetDomainOracle::train(
            &store.asset_domain_training_rows(self.active.corpus_id, self.embedder.model_name())?,
        );
        *self.asset_domain_oracle.write() = Some(oracle.clone());
        Ok(oracle)
    }

    pub(super) fn asset_domain_overlay(
        &self,
        store: &Store,
        asset_ids: &[AssetId],
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> anyhow::Result<(AssetDomainStatus, HashMap<AssetId, AssetDomainView>)> {
        let manual = store.asset_domain_labels(asset_ids)?;
        let oracle = self.asset_domain_oracle(store)?;
        let status = oracle.status();
        let views = asset_ids
            .iter()
            .map(|asset_id| {
                let predicted = embeddings
                    .get(asset_id)
                    .and_then(|embedding| oracle.predict(embedding));
                (
                    asset_id.clone(),
                    AssetDomainView {
                        manual: manual.get(asset_id).copied(),
                        predicted,
                    },
                )
            })
            .collect();
        Ok((status, views))
    }

    pub(super) fn asset_domain_view(
        &self,
        store: &Store,
        asset_id: &AssetId,
        embedding: Option<&[f32]>,
    ) -> anyhow::Result<AssetDomainView> {
        let manual = store
            .asset_domain_labels(std::slice::from_ref(asset_id))?
            .get(asset_id)
            .copied();
        let predicted = embedding.and_then(|embedding| {
            self.asset_domain_oracle(store)
                .ok()
                .and_then(|oracle| oracle.predict(embedding))
        });
        Ok(AssetDomainView { manual, predicted })
    }

    pub(super) fn pairing_asset_domain_labels(
        &self,
        store: &Store,
        asset_ids: &[AssetId],
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> anyhow::Result<Option<HashMap<AssetId, AssetDomainLabel>>> {
        let (status, views) = self.asset_domain_overlay(store, asset_ids, embeddings)?;
        if !status.ready() {
            return Ok(None);
        }
        Ok(Some(
            asset_ids
                .iter()
                .filter_map(|asset_id| {
                    views.get(asset_id).copied().and_then(|view| {
                        view.manual
                            .or_else(|| view.predicted.map(AssetDomainPrediction::label))
                            .map(|label| (asset_id.clone(), label))
                    })
                })
                .collect(),
        ))
    }

    pub fn rescan(&self) -> anyhow::Result<StartupSummary> {
        let mut store = Store::open_hot(&self.db_path)?;
        store.ingest_corpus(&self.root_path, self.active.corpus_id, &self.embedder)?;
        let session_id = self.active.session_id;
        self.with_write_store("touch_session", move |store| {
            store.touch_session(session_id)
        })?;
        self.invalidate_session_field_cache();
        self.purge_explore_vectors();
        self.purge_all_explore_layouts();
        self.refresh_external_sources_if_due(true)?;
        self.startup_summary()
    }

    pub(super) fn purge_duplicate_frontier(&self) {
        self.duplicate_frontier.write().take();
    }

    pub(super) fn ensure_duplicate_frontier(&self, store: &Store) -> anyhow::Result<()> {
        if self.duplicate_frontier.read().is_some() {
            return Ok(());
        }
        let points = store.all_frontier_embeddings(self.embedder.model_name())?;
        let frontier = DuplicateFrontier {
            tree: VpTree::forge(points),
        };
        *self.duplicate_frontier.write() = Some(frontier);
        Ok(())
    }

    pub(super) fn dedup_radius(&self) -> f32 {
        self.config.read().dedup_radius()
    }

    pub fn set_dedup_radius(&self, radius: f32) -> anyhow::Result<()> {
        let snapshot = {
            let mut config = self.config.write();
            config.shove_dedup_radius(radius);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)
    }

    pub fn startup_summary(&self) -> anyhow::Result<StartupSummary> {
        let store = self.read_store()?;
        let visible_assets = visible_assets(&store, self.active.corpus_id)?.len();
        let embedded_assets = store
            .corpus_embeddings(self.active.corpus_id, self.embedder.model_name())?
            .len();
        Ok(StartupSummary {
            corpus_id: self.active.corpus_id,
            session_id: self.active.session_id,
            visible_assets,
            embedded_assets,
        })
    }
}
