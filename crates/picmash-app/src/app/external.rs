use super::support::asset_visual_key_excluded;
use super::*;
use std::collections::{HashMap, HashSet};

use crate::{
    crush::crush_import_image,
    face::align_face_for_embedding,
    identity::{ImageIdentity, VisualKey, canonical_embedding_image, inspect_image_bytes},
    quality_features::{
        LinearTechnicalPriorHead, QUALITY_FEATURE_REVISION, technical_prior_mean_with_head,
        technical_prior_variance_with_head,
    },
};
use time::OffsetDateTime;

const EXTERNAL_STREAM_WRITE_STREAM_CAP: usize = 8;
const EXTERNAL_STREAM_WRITE_ITEM_BUDGET: usize = 512;

struct PreparedRemoteImport {
    import_path: PathBuf,
    identity: ImageIdentity,
    rotation_quarters: i32,
    embedding: Option<crate::model::EmbeddingRecord>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ExternalWarmOutcome {
    became_ready: bool,
    needs_face_backfill: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RemoteSourceArenaBonuses {
    pub(super) stream_size: f32,
    pub(super) freshness: f32,
}

impl AppState {
    fn arena_recent_visual_exclusions(
        &self,
        store: &Store,
        exempt_visual_key: Option<&VisualKey>,
    ) -> anyhow::Result<HashSet<VisualKey>> {
        let mut excluded = store
            .recent_arena_visual_keys(self.active.session_id, ARENA_RECENT_VISUAL_EXCLUDE)?
            .into_iter()
            .collect::<HashSet<_>>();
        if let Some(exempt_visual_key) = exempt_visual_key {
            excluded.remove(exempt_visual_key);
        }
        Ok(excluded)
    }

    fn arena_effective_visual_exclusions(
        &self,
        store: &Store,
        excluded_visual_keys: &HashSet<VisualKey>,
        exempt_visual_key: Option<&VisualKey>,
    ) -> anyhow::Result<HashSet<VisualKey>> {
        let mut effective = self.arena_recent_visual_exclusions(store, exempt_visual_key)?;
        effective.extend(excluded_visual_keys.iter().cloned());
        if let Some(exempt_visual_key) = exempt_visual_key {
            effective.remove(exempt_visual_key);
        }
        Ok(effective)
    }

    pub(super) fn set_external_subsource_lock(
        &self,
        item_id: RemoteItemId,
        active: bool,
    ) -> anyhow::Result<PipelineDisposition> {
        let session_id = self.active.session_id;
        let transition = self.with_write_store("set_external_subsource_lock", move |store| {
            let scope = ArenaScope::from(store.session_subsource_lock(session_id)?);
            let transition = if active {
                let item = store
                    .remote_item(item_id)?
                    .with_context(|| format!("missing remote item {}", item_id.0))?;
                let transition = scope.reduce(ArenaScopeAction::Lock(ThreadKey::from(&item)));
                transition.apply(store, session_id)?;
                info!(
                    source = %item.source_key,
                    stream_id = item.stream_id,
                    thread_no = item.thread_no,
                    title = %item.stream_title,
                    "locked arena to external subsource"
                );
                transition
            } else {
                let transition = scope.reduce(ArenaScopeAction::Unlock);
                transition.apply(store, session_id)?;
                if transition.clears_lock() {
                    info!("cleared arena external subsource lock");
                }
                transition
            };
            store.touch_session(session_id)?;
            Ok(transition)
        })?;
        if let Some(field) = self.session_field_cache.write().as_mut() {
            field.subsource_lock = transition.next_lock();
        }
        if active && let Some(lock) = transition.next_lock() {
            self.request_locked_stream_refresh(lock);
        }
        Ok(transition.pipeline())
    }

    pub(super) fn clear_external_subsource_lock(&self) -> anyhow::Result<()> {
        let session_id = self.active.session_id;
        self.with_write_store("clear_external_subsource_lock", move |store| {
            store.clear_session_subsource_lock(session_id)?;
            store.touch_session(session_id)
        })?;
        if let Some(field) = self.session_field_cache.write().as_mut() {
            field.subsource_lock = None;
        }
        Ok(())
    }

    pub(super) fn locked_stream_ready_target(live_items: usize) -> usize {
        live_items.min(EXTERNAL_LOCKED_STREAM_READY_TARGET)
    }

    fn locked_stream_frontier_counts(
        &self,
        store: &Store,
        lock: &crate::model::SessionSubsourceLock,
    ) -> anyhow::Result<Option<crate::store::ExternalStreamFrontierCounts>> {
        store.external_stream_frontier_counts(
            &lock.source_key,
            lock.stream_id,
            self.embedder.model_name(),
        )
    }

    pub(super) fn locked_source_needs_refresh(
        &self,
        store: &Store,
        source: &SourceConfig,
    ) -> anyhow::Result<bool> {
        let Some(lock) = store.session_subsource_lock(self.active.session_id)? else {
            return Ok(false);
        };
        let source_key = source.source_key();
        if lock.source_key != source_key {
            return Ok(false);
        }
        let Some(counts) = self.locked_stream_frontier_counts(store, &lock)? else {
            return Ok(true);
        };
        Ok(counts.ready_items < Self::locked_stream_ready_target(counts.live_items))
    }

    pub(super) fn locked_stream_should_hold_lock(
        &self,
        store: &Store,
        lock: &crate::model::SessionSubsourceLock,
    ) -> anyhow::Result<bool> {
        let Some(source) = self.source_config_for_key(&lock.source_key) else {
            return Ok(false);
        };
        let Some(counts) = self.locked_stream_frontier_counts(store, lock)? else {
            return Ok(source.scan_interval_seconds > 0);
        };
        if counts.ready_items < counts.live_items {
            return Ok(true);
        }
        if source.scan_interval_seconds == 0 {
            return Ok(false);
        }
        store.external_scan_due(
            &lock.source_key,
            Duration::seconds(source.scan_interval_seconds as i64),
        )
    }

    pub(super) fn refill_locked_source_now(
        &self,
        lock: &crate::model::SessionSubsourceLock,
    ) -> anyhow::Result<()> {
        let Some(source) = self.source_config_for_key(&lock.source_key) else {
            return Ok(());
        };
        self.warm_locked_stream_from_store(&source, lock)?;
        let store = self.read_store()?;
        if !self.locked_source_needs_refresh(&store, &source)? {
            return Ok(());
        }
        drop(store);
        if source.local_directory().is_some() {
            self.devour_local_directory_refresh(&lock.source_key)?;
            self.warm_locked_stream_from_store(&source, lock)?;
        }
        Ok(())
    }

    pub fn refresh_locked_stream_now(
        &self,
        lock: &crate::model::SessionSubsourceLock,
    ) -> anyhow::Result<()> {
        let Some(active_lock) = self
            .read_store()?
            .session_subsource_lock(self.active.session_id)?
        else {
            return Ok(());
        };
        if active_lock != *lock {
            return Ok(());
        }
        self.refill_locked_source_now(lock)
    }

    fn warm_locked_stream_from_store(
        &self,
        source: &SourceConfig,
        lock: &crate::model::SessionSubsourceLock,
    ) -> anyhow::Result<()> {
        let store = Store::open_hot(&self.db_path)?;
        let Some(counts) = self.locked_stream_frontier_counts(&store, lock)? else {
            return Ok(());
        };
        let target = Self::locked_stream_ready_target(counts.live_items);
        if counts.ready_items >= target {
            return Ok(());
        }
        let candidates = store.external_stream_warm_candidates(&lock.source_key, lock.stream_id)?;
        let mut ready_items = counts.ready_items;
        let mut needs_face_backfill = false;
        for candidate in candidates {
            if ready_items >= target {
                break;
            }
            let outcome = self.devour_external_warm_candidate(
                &store,
                source,
                &lock.source_key,
                lock.stream_id,
                candidate.item_id,
                &candidate.snapshot,
            )?;
            needs_face_backfill |= outcome.needs_face_backfill;
            ready_items += usize::from(outcome.became_ready);
        }
        if needs_face_backfill {
            self.schedule_external_face_embedding_backfill(&lock.source_key);
        }
        if ready_items > counts.ready_items {
            self.purge_duplicate_frontier();
        }
        Ok(())
    }

    pub(super) fn devour_local_directory_refresh(&self, source_key: &str) -> anyhow::Result<()> {
        let Some(source) = self.source_config_for_key(source_key) else {
            return Ok(());
        };
        if source.local_directory().is_none() {
            return Ok(());
        }
        info!(source = %source_key, "refreshing local directory source in maintenance");
        let harvest = self.source_scanner.harvest(&source)?;
        self.devour_external_harvest(&source, &harvest)
    }

    pub(super) fn remote_source_scan_can_rest(
        &self,
        store: &Store,
        source: &SourceConfig,
    ) -> anyhow::Result<bool> {
        if source.local_directory().is_some() {
            return Ok(false);
        }
        if self.locked_source_needs_refresh(store, source)? {
            return Ok(false);
        }
        let source_key = source.source_key();
        let (active_streams, _, _) = store.external_source_counts(&source_key)?;
        if active_streams == 0 {
            return Ok(false);
        }
        let recently_selected = store.external_source_recently_selected(
            self.active.session_id,
            &source_key,
            OffsetDateTime::now_utc().unix_timestamp()
                - REMOTE_SOURCE_IDLE_SCAN_GRACE.whole_seconds(),
        )?;
        let (total_ready, _) =
            store.external_source_ready_profile(&source_key, self.embedder.model_name())?;
        if total_ready == 0
            && !recently_selected
            && !store.external_scan_due(&source_key, REMOTE_SOURCE_EMPTY_SCAN_BACKOFF)?
        {
            return Ok(true);
        }
        let frontier = SourceReadyFrontier::new(
            total_ready,
            HashMap::new(),
            active_streams,
            ReadyTargetProfile::for_source(source),
        );
        if !frontier.source_idle_warm() {
            return Ok(false);
        }
        Ok(!recently_selected)
    }

    fn ensure_remote_face_embedding(
        &self,
        store: &Store,
        item_id: RemoteItemId,
        source_key: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        if let Some(embedding) =
            store.external_item_face_embedding(item_id, self.embedder.recognition_model_name())?
        {
            return Ok(Some(embedding));
        }
        self.schedule_external_face_embedding_backfill(source_key);
        Ok(None)
    }

    pub(super) fn backfill_external_face_embeddings_for_source(
        &self,
        source_key: &str,
        limit: usize,
    ) -> anyhow::Result<usize> {
        if !self.embedder.face_detection_enabled() || limit == 0 {
            return Ok(0);
        }
        let store = Store::open_hot(&self.db_path)?;
        let missing = store.external_items_missing_face_embedding_batch(
            source_key,
            self.embedder.model_name(),
            self.embedder.recognition_model_name(),
            limit,
        )?;
        let mut attempted = 0usize;
        let mut embedded = 0usize;
        for (item_id, cached_path) in missing {
            if !cached_path.exists() {
                continue;
            }
            let Ok(bytes) = fs::read(&cached_path) else {
                continue;
            };
            let face_embedding =
                self.devour_remote_face_embedding(&store, item_id, &cached_path, &bytes)?;
            let embedded_now = face_embedding.is_some();
            let recognition_model = self.embedder.recognition_model_name().to_owned();
            self.with_write_store("save_external_face_embedding", move |store| {
                store.save_external_face_embedding(
                    item_id,
                    &recognition_model,
                    face_embedding.as_ref(),
                )
            })?;
            attempted += 1;
            if embedded_now {
                embedded += 1;
            }
        }
        if attempted > 0 {
            info!(
                source = source_key,
                attempted, embedded, "external frontier face backfill pass complete"
            );
        }
        Ok(embedded)
    }

    pub(crate) fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    pub(crate) fn dino_status_note(&self) -> Option<&str> {
        self.embedder.status_note()
    }

    pub(super) fn devour_external_harvest(
        &self,
        source: &SourceConfig,
        harvest: &SourceHarvest,
    ) -> anyhow::Result<()> {
        let source_key = source.source_key();
        self.with_write_store("upsert_external_source", {
            let source_key = source_key.clone();
            let display_name = harvest.display_name.clone();
            let source_type = source.source_type_name().to_owned();
            let source_locator = source.source_locator();
            move |store| {
                store.upsert_external_source(
                    &source_key,
                    &display_name,
                    &source_type,
                    &source_locator,
                    None,
                )
            }
        })?;

        let live_thread_nos = harvest
            .streams
            .iter()
            .map(|stream| stream.thread_no)
            .collect::<Vec<_>>();
        self.with_write_store("retire_missing_external_streams", {
            let source_key = source_key.clone();
            move |store| store.retire_missing_external_streams(&source_key, &live_thread_nos)
        })?;

        if source.local_directory().is_some() {
            let live_post_nos = harvest
                .streams
                .iter()
                .flat_map(|stream| stream.items.iter().map(|item| item.post_no))
                .collect::<Vec<_>>();
            self.with_write_store("withdraw_missing_external_items", {
                let source_key = source_key.clone();
                move |store| store.withdraw_missing_external_items(&source_key, &live_post_nos)
            })?;
        }

        let store = Store::open_hot(&self.db_path)?;
        let ready_target_profile = self.remote_source_materialization_profile(&store, source)?;
        let (total_ready, ready_by_stream) =
            store.external_source_ready_profile(&source_key, self.embedder.model_name())?;
        let mut ready_frontier = SourceReadyFrontier::new(
            total_ready,
            ready_by_stream,
            harvest.streams.len(),
            ready_target_profile,
        );
        let active_lock = store.session_subsource_lock(self.active.session_id)?;
        let mut needs_face_backfill = false;

        let mut chunk_start = 0usize;
        while chunk_start < harvest.streams.len() {
            let mut chunk_end = chunk_start;
            let mut chunk_items = 0usize;
            while chunk_end < harvest.streams.len()
                && chunk_end - chunk_start < EXTERNAL_STREAM_WRITE_STREAM_CAP
            {
                let next_items = harvest.streams[chunk_end].items.len();
                if chunk_end > chunk_start
                    && chunk_items + next_items > EXTERNAL_STREAM_WRITE_ITEM_BUDGET
                {
                    break;
                }
                chunk_items += next_items;
                chunk_end += 1;
            }
            let stream_chunk = &harvest.streams[chunk_start..chunk_end];
            let batch_streams = stream_chunk.to_vec();
            let outcomes = self.with_write_store("upsert_external_stream_batch", {
                let source_key = source_key.clone();
                move |store| store.upsert_external_streams_batch(&source_key, &batch_streams)
            })?;

            for (stream, outcome) in stream_chunk.iter().zip(outcomes) {
                if outcome.blocked {
                    continue;
                }
                let stream_id = outcome.stream_id;
                let item_ids = outcome.item_ids;
                let locked_stream_target = active_lock
                    .as_ref()
                    .filter(|lock| lock.source_key == source_key && lock.stream_id == stream_id)
                    .map(|_| Self::locked_stream_ready_target(stream.items.len()));

                for item in &stream.items {
                    let item_id = item_ids.get(&item.post_no).copied().with_context(|| {
                        format!(
                            "missing upserted external item mapping source={} post_no={}",
                            source_key, item.post_no
                        )
                    })?;
                    if store.external_item_hidden(item_id)? {
                        continue;
                    }

                    let should_devour = if store
                        .external_item_frontier_ready(item_id, self.embedder.model_name())?
                    {
                        true
                    } else {
                        locked_stream_target.is_some_and(|target| {
                            ready_frontier.ready_in_stream(stream_id) < target
                        }) || !ready_frontier.source_saturated()
                            || !ready_frontier.stream_saturated(stream_id)
                    };
                    if !should_devour {
                        continue;
                    }
                    let outcome = self.devour_external_warm_candidate(
                        &store,
                        source,
                        &source_key,
                        stream_id,
                        item_id,
                        item,
                    )?;
                    needs_face_backfill |= outcome.needs_face_backfill;
                    if outcome.became_ready {
                        ready_frontier.note_ready(stream_id);
                    }
                }
            }
            chunk_start = chunk_end;
        }

        if needs_face_backfill {
            self.schedule_external_face_embedding_backfill(&source_key);
        }

        self.purge_duplicate_frontier();
        Ok(())
    }

    fn devour_external_warm_candidate(
        &self,
        store: &Store,
        source: &SourceConfig,
        source_key: &str,
        stream_id: i64,
        item_id: RemoteItemId,
        item: &crate::sources::RemoteItemSnapshot,
    ) -> anyhow::Result<ExternalWarmOutcome> {
        let already_ready =
            store.external_item_frontier_ready(item_id, self.embedder.model_name())?;
        let warm = store.external_item_warm_state(
            item_id,
            self.embedder.model_name(),
            self.embedder
                .clip_enabled()
                .then_some(self.embedder.clip_model_name()),
            self.embedder.recognition_model_name(),
            QUALITY_FEATURE_REVISION,
        )?;
        let mut outcome = ExternalWarmOutcome {
            became_ready: false,
            needs_face_backfill: warm.needs_face_embedding,
        };
        if already_ready
            && !warm.needs_inline_work()
            && store
                .external_item_cached_path(item_id)?
                .is_some_and(|path| path.exists())
        {
            return Ok(outcome);
        }

        let cached_path = match item.materialized_path.as_ref() {
            Some(path) if path.exists() => path.clone(),
            Some(_) if source.local_directory().is_some() => return Ok(outcome),
            Some(_) | None => self.source_scanner.cache_remote_image(source_key, item)?,
        };
        if !cached_path.exists() {
            return Ok(outcome);
        }

        if warm.needs_materialization && !warm.needs_identity {
            self.with_write_store("save_external_item_cached_path", {
                let cached_path = cached_path.clone();
                move |store| store.save_external_item_cached_path(item_id, &cached_path)
            })?;
        }

        if warm.needs_inline_work() {
            let bytes =
                if warm.needs_identity || warm.needs_quality_features {
                    Some(fs::read(&cached_path).with_context(|| {
                        format!("reading cached remote {}", cached_path.display())
                    })?)
                } else {
                    None
                };

            if warm.needs_identity {
                let bytes = bytes
                    .as_deref()
                    .context("missing cached bytes for identity")?;
                let identity = inspect_image_bytes(bytes).with_context(|| {
                    format!(
                        "inspecting cached remote identity {}",
                        cached_path.display()
                    )
                })?;
                let disposition = self.with_write_store("save_external_item_identity", {
                    let cached_path = cached_path.clone();
                    move |store| store.save_external_item_identity(item_id, &identity, &cached_path)
                })?;
                if disposition != crate::store::ExternalIdentityDisposition::Active {
                    return Ok(outcome);
                }
            }

            if warm.needs_quality_features
                && let Some(bytes) = bytes.as_deref()
                && let Ok(features) = crate::quality_features::extract_asset_quality_features(bytes)
            {
                self.with_write_store("save_external_item_quality_features", move |store| {
                    store.save_external_item_quality_features(
                        item_id,
                        QUALITY_FEATURE_REVISION,
                        &features,
                    )
                })?;
            }

            if warm.needs_embedding
                && let Some(embedding) = self.embedder.embed(&cached_path)?
            {
                let cached_path = cached_path.clone();
                self.with_write_store("save_external_embedding", move |store| {
                    store.save_external_embedding(item_id, &embedding, &cached_path)
                })?;
            }

            if warm.needs_clip_embedding
                && let Some(embedding) = self.embedder.clip_embed(&cached_path)?
            {
                self.with_write_store("save_external_clip_embedding", move |store| {
                    store.save_external_clip_embedding(item_id, &embedding)
                })?;
            }
        }

        if !already_ready
            && store.external_item_frontier_ready(item_id, self.embedder.model_name())?
        {
            outcome.became_ready = true;
        }
        if outcome.became_ready {
            debug!(
                source = %source_key,
                stream_id,
                item_id = item_id.0,
                "warmed locked external frontier item"
            );
        }
        Ok(outcome)
    }

    fn remote_source_materialization_profile(
        &self,
        store: &Store,
        source: &SourceConfig,
    ) -> anyhow::Result<ReadyTargetProfile> {
        let profile = ReadyTargetProfile::for_source(source);
        if source.local_directory().is_some() {
            return Ok(profile);
        }
        let source_key = source.source_key();
        let recent_since = OffsetDateTime::now_utc().unix_timestamp()
            - REMOTE_SOURCE_IDLE_SCAN_GRACE.whole_seconds();
        Ok(
            if store.external_source_recently_selected(
                self.active.session_id,
                &source_key,
                recent_since,
            )? {
                profile.capped(REMOTE_SOURCE_RECENT_READY_CAP)
            } else {
                profile.capped(profile.idle_floor())
            },
        )
    }

    pub(super) fn choose_pair_with_store(
        &self,
        store: &Store,
        field: &SessionField,
        assets: &[AssetRecord],
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        if assets.is_empty() {
            return Ok(None);
        }

        let effective_excluded =
            self.arena_effective_visual_exclusions(store, excluded_visual_keys, None)?;
        let explore = self.arena_explore();
        let external_probability = self.config.read().external_probability();
        if external_probability_is_certain(external_probability) {
            let external_pair =
                self.choose_external_pair(store, field, assets, explore, &effective_excluded)?;
            if external_pair.is_some() {
                return Ok(external_pair);
            }
            let mut local_pair =
                choose_local_pair(assets, field, store, explore, &effective_excluded)?;
            if local_pair.is_none() {
                local_pair =
                    choose_local_pair(assets, field, store, explore, excluded_visual_keys)?;
            }
            return Ok(local_pair);
        }
        let mut local_pair = choose_local_pair(assets, field, store, explore, &effective_excluded)?;
        if local_pair.is_none() && external_probability_is_zero(external_probability) {
            local_pair = choose_local_pair(assets, field, store, explore, excluded_visual_keys)?;
        }
        if external_probability_is_zero(external_probability) {
            return Ok(local_pair);
        }
        let external_pair =
            self.choose_external_pair(store, field, assets, explore, &effective_excluded)?;
        Ok(choose_pair_source(
            &mut rng(),
            external_probability,
            local_pair,
            external_pair,
        ))
    }

    pub(super) fn load_arena_card(
        &self,
        store: &Store,
        field: &SessionField,
        handle: &ArenaHandle,
    ) -> anyhow::Result<Option<ArenaCard>> {
        match handle {
            ArenaHandle::Local(asset_id) => {
                let Some(asset) = store
                    .corpus_asset(self.active.corpus_id, asset_id)?
                    .filter(|asset| !asset.hidden && asset.path.exists())
                else {
                    return Ok(None);
                };
                let domain = self.asset_domain_view(store, asset_id, field.embedding(asset_id))?;
                Ok(Some(ArenaCard::Local(ArenaLocalCard {
                    utility: field.utility(&asset),
                    hearted: field.hearted(asset_id),
                    quality: field.quality_summary_or_fallback(&asset),
                    domain,
                    asset,
                })))
            }
            ArenaHandle::Remote(item_id) => {
                let Some(item) = store.arena_remote_item(*item_id)? else {
                    return Ok(None);
                };
                if !item.path.exists() {
                    return Ok(None);
                }
                let Some(embedding) =
                    store.external_item_embedding(*item_id, self.embedder.model_name())?
                else {
                    return Ok(None);
                };
                let face_embedding =
                    self.ensure_remote_face_embedding(store, *item_id, &item.source_key)?;
                let quality_features = Self::lazy_remote_quality_features(
                    store.external_item_quality_features(*item_id, QUALITY_FEATURE_REVISION)?,
                );
                let quality_cache = store
                    .external_item_quality_cache(*item_id, field.quality_model)?
                    .map(|cache| cache.payload);
                let technical_head = store.active_quality_model().ok().and_then(|model| {
                    store
                        .load_linear_technical_prior_head(&model)
                        .ok()
                        .flatten()
                });
                let domain_oracle = self.asset_domain_oracle(store)?;
                let face_oracle = self.ensure_face_oracle(store)?;
                let quality = Self::remote_quality_summary(
                    field,
                    &embedding,
                    face_embedding.as_deref(),
                    quality_features.as_ref(),
                    quality_cache.as_ref(),
                    technical_head.as_ref(),
                    Some(&domain_oracle),
                    face_oracle.as_ref(),
                );
                let stream_locked = field.subsource_lock.as_ref().is_some_and(|lock| {
                    lock.source_key == item.source_key && lock.stream_id == item.stream_id
                });
                Ok(Some(ArenaCard::Remote(ArenaRemoteCard {
                    item,
                    hearted: false,
                    stream_locked,
                    utility: quality.asset.mean,
                    quality,
                })))
            }
        }
    }

    pub(super) fn cluster_around_remote(
        &self,
        store: &Store,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<DuplicateCluster>> {
        self.ensure_duplicate_frontier(store)?;
        let Some(query) = store.external_item_embedding(item_id, self.embedder.model_name())?
        else {
            return Ok(None);
        };
        let radius = self.dedup_radius();
        if radius <= 0.0 {
            return Ok(None);
        }
        let normalized = normalize_embedding(&query);
        let mut satellites = self
            .duplicate_frontier
            .read()
            .as_ref()
            .map(|frontier| frontier.tree.ransack(&normalized, radius))
            .unwrap_or_default()
            .into_iter()
            .filter(|(candidate_id, _)| *candidate_id != item_id)
            .filter_map(|(candidate_id, distance)| {
                store
                    .remote_item(candidate_id)
                    .ok()
                    .flatten()
                    .filter(|item| item.path.exists())
                    .map(|item| ClusterSatellite { item, distance })
            })
            .collect::<Vec<_>>();
        satellites.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.item.id.0.cmp(&right.item.id.0))
        });
        satellites.truncate(12);
        Ok((!satellites.is_empty()).then_some(DuplicateCluster { satellites }))
    }

    pub(super) fn choose_pair_preserving_local_anchor_with_store(
        &self,
        store: &Store,
        field: &SessionField,
        local_anchor: Option<&AssetId>,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        let assets = visible_assets(store, self.active.corpus_id)?;
        let pair = match local_anchor
            .and_then(|asset_id| assets.iter().find(|asset| asset.id == *asset_id).cloned())
        {
            Some(anchor) => self.choose_pair_against_local_anchor_with_store(
                store,
                field,
                &assets,
                &anchor,
                &self.arena_effective_visual_exclusions(
                    store,
                    excluded_visual_keys,
                    anchor.visual_key.as_ref(),
                )?,
            )?,
            None => self.choose_pair_with_store(store, field, &assets, excluded_visual_keys)?,
        };
        Ok(pair)
    }

    pub(super) fn choose_pair_in_locked_subsource_with_store(
        &self,
        store: &Store,
        field: &SessionField,
        assets: &[AssetRecord],
        local_anchor: Option<&AssetId>,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        let Some(lock) = field.subsource_lock.as_ref() else {
            return Ok(None);
        };
        let Some(source) = self.source_config_for_key(&lock.source_key) else {
            return Ok(None);
        };
        let mut effective_excluded =
            self.arena_effective_visual_exclusions(store, excluded_visual_keys, None)?;
        let anchor = if let Some(anchor) = local_anchor
            .and_then(|asset_id| assets.iter().find(|asset| asset.id == *asset_id).cloned())
        {
            effective_excluded = self.arena_effective_visual_exclusions(
                store,
                excluded_visual_keys,
                anchor.visual_key.as_ref(),
            )?;
            anchor
        } else {
            if assets.is_empty() {
                return Ok(None);
            }
            let explore = self.arena_explore();
            let recent = (assets.len() > ARENA_RECENT_REPEAT_EXCLUDE)
                .then(|| {
                    store.recent_arena_asset_ids(field.session.id, ARENA_RECENT_REPEAT_EXCLUDE)
                })
                .transpose()?
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>();
            let pool = assets
                .iter()
                .filter(|asset| !recent.contains(&asset.id))
                .filter(|asset| !asset_visual_key_excluded(asset, &effective_excluded))
                .cloned()
                .collect::<Vec<_>>();
            if pool.is_empty() {
                return Ok(None);
            }
            let mut rng = rng();
            let anchor_scores = pool
                .iter()
                .map(|asset| field.arena_anchor_score(asset, explore))
                .collect::<Vec<_>>();
            let Some(anchor_index) = sample_softmax_index(
                &mut rng,
                &anchor_scores,
                arena_sampling_temperature(explore),
                arena_uniform_mix(explore),
            ) else {
                return Ok(None);
            };
            pool[anchor_index].clone()
        };
        let explore = self.arena_explore();
        let Some(scored) = self.pick_remote_candidate_from_locked_stream(
            store,
            field,
            &anchor,
            &source,
            lock,
            explore,
            &effective_excluded,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(ArenaPair {
            left: ArenaCard::Local(ArenaLocalCard {
                utility: field.utility(&anchor),
                hearted: field.hearted(&anchor.id),
                quality: field.quality_summary_or_fallback(&anchor),
                domain: AssetDomainView::default(),
                asset: anchor,
            }),
            right: ArenaCard::Remote(ArenaRemoteCard {
                item: scored.candidate.item,
                hearted: false,
                stream_locked: true,
                utility: scored.utility,
                quality: scored.quality,
            }),
        }))
    }

    pub(super) fn enshrine_remote_item(
        &self,
        item_id: RemoteItemId,
        hearted: bool,
    ) -> anyhow::Result<()> {
        let asset_id = self.seal_imported_external_outcome(
            item_id,
            if hearted {
                ExternalEventKind::Hearted
            } else {
                ExternalEventKind::Kept
            },
        )?;
        if hearted {
            self.set_heart_asset(&asset_id, true)?;
        }
        Ok(())
    }

    fn queue_external_import_outcome(
        &self,
        item_id: RemoteItemId,
        outcome_kind: ExternalEventKind,
    ) -> anyhow::Result<()> {
        let session_id = self.active.session_id;
        let corpus_id = self.active.corpus_id;
        let queued = self.with_write_store("queue_external_import_outcome", move |store| {
            store.queue_external_import_outcome(session_id, corpus_id, item_id, outcome_kind)
        })?;
        if queued {
            self.purge_duplicate_frontier();
            self.maintenance_notify.notify_one();
        }
        Ok(())
    }

    pub(super) fn seal_imported_external_outcome(
        &self,
        item_id: RemoteItemId,
        outcome_kind: ExternalEventKind,
    ) -> anyhow::Result<AssetId> {
        let prepared = self.prepare_remote_import(item_id)?;
        let session_id = self.active.session_id;
        let corpus_id = self.active.corpus_id;
        let asset_id = self.with_write_store("seal_imported_external_outcome", move |store| {
            store.seal_imported_external_outcome_precomputed(
                session_id,
                corpus_id,
                item_id,
                &prepared.import_path,
                &prepared.identity,
                prepared.rotation_quarters,
                prepared.embedding.as_ref(),
                outcome_kind,
            )
        })?;
        if self.quality_refresh_is_inline()? {
            self.devour_quality_model_refresh()?;
        }
        self.purge_duplicate_frontier();
        Ok(asset_id)
    }

    pub(super) fn devour_pending_external_import_outcome(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<()> {
        let pending = {
            let store = Store::open_hot(&self.db_path)?;
            store.pending_external_import_outcome(item_id)?
        };
        let Some(pending) = pending else {
            return Ok(());
        };
        let prepared = self.prepare_remote_import(item_id)?;
        self.with_write_store("finalize_external_import_outcome", move |store| {
            store.seal_imported_external_outcome_precomputed(
                pending.session_id,
                pending.corpus_id,
                item_id,
                &prepared.import_path,
                &prepared.identity,
                prepared.rotation_quarters,
                prepared.embedding.as_ref(),
                pending.outcome_kind,
            )
        })?;
        if self.quality_refresh_is_inline()? {
            self.devour_quality_model_refresh()?;
        } else {
            self.schedule_quality_model_refresh();
        }
        self.purge_duplicate_frontier();
        Ok(())
    }

    fn prepare_remote_import(&self, item_id: RemoteItemId) -> anyhow::Result<PreparedRemoteImport> {
        let (item, item_snapshot, embedding_record) = {
            let store = Store::open_hot(&self.db_path)?;
            let item = store
                .remote_item(item_id)?
                .with_context(|| format!("missing remote item {}", item_id.0))?;
            let item_snapshot = store
                .remote_item_snapshot(item_id)?
                .with_context(|| format!("missing remote item snapshot {}", item_id.0))?;
            let embedding = store.external_item_embedding(item_id, self.embedder.model_name())?;
            let embedding_record = embedding
                .as_ref()
                .map(|vector| crate::model::EmbeddingRecord {
                    model_name: self.embedder.model_name().to_owned(),
                    vector: vector.clone(),
                });
            (item, item_snapshot, embedding_record)
        };
        let source_path = if item.image_url.starts_with("file://") {
            item.path.clone()
        } else {
            self.source_scanner
                .cache_remote_image(&item.source_key, &item_snapshot)?
        };
        let raw_bytes = fs::read(&source_path)
            .with_context(|| format!("reading remote import {}", source_path.display()))?;
        let import_dir = self
            .root_path
            .join(".picmash-imported")
            .join(sanitize_source_key(&item.source_key));
        fs::create_dir_all(&import_dir)
            .with_context(|| format!("creating import directory {}", import_dir.display()))?;

        let path_hint = import_dir.join(format!(
            "{}-{}.{}",
            item.thread_no,
            item.post_no,
            extension_or_fallback(&item.path)
        ));
        let crushed = crush_import_image(&path_hint, &raw_bytes)
            .context("normalizing remote import into canonical jxl")?;
        let extension = crushed.extension().to_owned();
        let import_bytes = crushed.into_bytes();
        let identity =
            inspect_image_bytes(&import_bytes).context("inspecting external import identity")?;
        let import_path =
            import_dir.join(format!("{}-{}.{}", item.thread_no, item.post_no, extension));
        fs::write(&import_path, &import_bytes)
            .with_context(|| format!("writing imported remote {}", import_path.display()))?;
        Ok(PreparedRemoteImport {
            import_path,
            identity,
            rotation_quarters: item.rotation_quarters,
            embedding: embedding_record,
        })
    }

    pub(super) fn vote_remote_duel(
        &self,
        local_asset_id: &AssetId,
        remote_item_id: RemoteItemId,
        winner: &ArenaHandle,
    ) -> anyhow::Result<()> {
        let (local_asset, source_policy) = {
            let store = self.read_store()?;
            let Some(local_asset) = store.corpus_asset(self.active.corpus_id, local_asset_id)?
            else {
                return Ok(());
            };
            if local_asset.hidden || !local_asset.path.exists() {
                return Ok(());
            }
            let Some(remote_item) = store.remote_item(remote_item_id)? else {
                return Ok(());
            };
            if !remote_item.path.exists() {
                return Ok(());
            }
            let source_policy = self
                .source_config_for_key(&remote_item.source_key)
                .map_or(ImportPolicy::NotX, |source| source.import_policy);
            (local_asset, source_policy)
        };

        match winner {
            ArenaHandle::Local(winner_id) if *winner_id == local_asset.id => {
                if source_policy == ImportPolicy::NotX {
                    self.queue_external_import_outcome(
                        remote_item_id,
                        ExternalEventKind::LocalWin,
                    )?;
                } else {
                    let session_id = self.active.session_id;
                    let corpus_id = self.active.corpus_id;
                    let local_asset_id = local_asset.id;
                    self.with_write_store("record_external_result", move |store| {
                        store.record_external_result_and_touch_session(
                            session_id,
                            corpus_id,
                            remote_item_id,
                            &local_asset_id,
                            ExternalEventKind::LocalWin,
                        )
                    })?;
                    if self.quality_refresh_is_inline()? {
                        self.devour_quality_model_refresh()?;
                    }
                }
                Ok(())
            }
            ArenaHandle::Remote(winner_id) if *winner_id == remote_item_id => {
                self.queue_external_import_outcome(remote_item_id, ExternalEventKind::RemoteWin)?;
                Ok(())
            }
            _ => bail!("remote duel winner is not one of the compared assets"),
        }
    }

    fn devour_remote_face_embedding(
        &self,
        store: &Store,
        item_id: RemoteItemId,
        cached_path: &Path,
        bytes: &[u8],
    ) -> anyhow::Result<Option<crate::model::EmbeddingRecord>> {
        let image = canonical_embedding_image(bytes).with_context(|| {
            format!("canonicalizing remote face probe {}", cached_path.display())
        })?;
        let faces = self.embedder.detect_faces(&image)?;
        let largest = faces
            .into_iter()
            .filter(|face| {
                face.bbox.w.min(face.bbox.h) >= crate::store::FACE_DETECTION_MIN_FACE_SIDE
            })
            .filter(|face| face.bbox.w > 0.0 && face.bbox.h > 0.0)
            .filter(|face| {
                !store
                    .face_is_tombstoned_for_detection(None, Some(item_id), face)
                    .unwrap_or(false)
            })
            .max_by(|left, right| {
                let left_area = left.bbox.w * left.bbox.h;
                let right_area = right.bbox.w * right.bbox.h;
                left_area
                    .total_cmp(&right_area)
                    .then_with(|| left.confidence.total_cmp(&right.confidence))
            });
        let Some(face) = largest else {
            return Ok(None);
        };
        let crop = align_face_for_embedding(&image, &face);
        self.embedder.recognize_face_image(&crop.crop)
    }

    fn lazy_remote_quality_features(
        features: Option<crate::quality_features::AssetQualityFeatures>,
    ) -> Option<crate::quality_features::AssetQualityFeatures> {
        features
    }

    fn remote_quality_summary(
        field: &SessionField,
        embedding: &[f32],
        face_embedding: Option<&[f32]>,
        quality_features: Option<&crate::quality_features::AssetQualityFeatures>,
        quality_cache: Option<&crate::quality::AssetQualityCachePayload>,
        technical_head: Option<&LinearTechnicalPriorHead>,
        domain_oracle: Option<&AssetDomainOracle>,
        face_oracle: Option<&FaceOracle>,
    ) -> AssetQualitySummary {
        let face = face_embedding
            .and_then(|embedding| face_oracle.and_then(|oracle| oracle.predict(embedding)))
            .map(|prediction| PosteriorSummary {
                mean: prediction.mean,
                sigma: prediction.sigma,
            });
        match field.quality_model {
            QualityFormalVersion::LegacyIndependentV1 => {
                let utility = field.residual_score_for_embedding(embedding);
                AssetQualitySummary {
                    asset: PosteriorSummary {
                        mean: utility,
                        sigma: crate::quality::legacy_cache_variance(0).sqrt(),
                    },
                    baseline: PosteriorSummary {
                        mean: utility,
                        sigma: crate::quality::legacy_cache_variance(0).sqrt(),
                    },
                    semantic: None,
                    vibe: None,
                    technical: None,
                    face,
                }
            }
            QualityFormalVersion::HierarchicalGaussianV1 => {
                let Some(session_quality) = field.hierarchical_session else {
                    let baseline = remote_baseline_summary();
                    return AssetQualitySummary {
                        asset: baseline,
                        baseline,
                        semantic: None,
                        vibe: None,
                        technical: None,
                        face,
                    };
                };
                if let Some(cache) = quality_cache
                    && let Some(posterior) =
                        crate::quality::HierarchicalAssetPosterior::decode(cache)
                {
                    return AssetQualitySummary {
                        asset: hierarchical_total_summary(&posterior, &session_quality, face),
                        baseline: PosteriorSummary {
                            mean: posterior.baseline_mean,
                            sigma: posterior.baseline_variance.max(0.0).sqrt(),
                        },
                        semantic: Some(hierarchical_semantic_summary(&posterior, &session_quality)),
                        vibe: hierarchical_vibe_summary(&posterior, &session_quality),
                        technical: posterior.technical_mean.map(|mean| PosteriorSummary {
                            mean,
                            sigma: posterior
                                .technical_variance
                                .unwrap_or(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                                .max(0.0)
                                .sqrt(),
                        }),
                        face,
                    };
                }
                let baseline = remote_baseline_summary();
                let gate_3d = matches!(
                    domain_oracle.and_then(|oracle| oracle
                        .predict(embedding)
                        .map(|prediction| prediction.label())),
                    Some(AssetDomainLabel::Real)
                );
                let tech_head = technical_head.copied().unwrap_or_default();
                let technical = gate_3d
                    .then(|| {
                        quality_features.map(|features| PosteriorSummary {
                            mean: technical_prior_mean_with_head(&tech_head, &features.technical),
                            sigma: technical_prior_variance_with_head(
                                &tech_head,
                                &features.technical,
                            )
                            .max(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                            .sqrt(),
                        })
                    })
                    .flatten();
                let vibe = gate_3d
                    .then(|| {
                        remote_vibe_summary(
                            quality_features.map(|features| &features.vibe),
                            &session_quality,
                        )
                    })
                    .flatten();
                AssetQualitySummary {
                    asset: remote_total_summary(
                        PosteriorSummary {
                            mean: 0.0,
                            sigma: 0.0,
                        },
                        baseline,
                        technical,
                        face,
                        vibe,
                    ),
                    baseline,
                    semantic: None,
                    vibe,
                    technical,
                    face,
                }
            }
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => {
                let Some(session_quality) = field.perturbative_session else {
                    let baseline = remote_baseline_summary();
                    return AssetQualitySummary {
                        asset: baseline,
                        baseline,
                        semantic: None,
                        vibe: None,
                        technical: None,
                        face,
                    };
                };
                if let Some(cache) = quality_cache
                    && let Some(posterior) =
                        crate::quality::PerturbativeAssetPosterior::decode(cache)
                {
                    return AssetQualitySummary {
                        asset: perturbative_total_summary(
                            &posterior,
                            &session_quality,
                            field.perturbative_hyper,
                            face,
                        ),
                        baseline: PosteriorSummary {
                            mean: posterior.baseline_mean,
                            sigma: posterior.baseline_variance.max(0.0).sqrt(),
                        },
                        semantic: Some(perturbative_semantic_summary(&posterior, &session_quality)),
                        vibe: perturbative_vibe_summary(
                            &posterior,
                            &session_quality,
                            field.perturbative_hyper,
                        ),
                        technical: posterior.technical_mean.map(|mean| PosteriorSummary {
                            mean,
                            sigma: posterior
                                .technical_variance
                                .unwrap_or(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                                .max(0.0)
                                .sqrt(),
                        }),
                        face,
                    };
                }
                let baseline = remote_baseline_summary();
                let gate_3d = matches!(
                    domain_oracle.and_then(|oracle| oracle
                        .predict(embedding)
                        .map(|prediction| prediction.label())),
                    Some(AssetDomainLabel::Real)
                );
                let tech_head = technical_head.copied().unwrap_or_default();
                let technical = gate_3d
                    .then(|| {
                        quality_features.map(|features| PosteriorSummary {
                            mean: technical_prior_mean_with_head(&tech_head, &features.technical),
                            sigma: technical_prior_variance_with_head(
                                &tech_head,
                                &features.technical,
                            )
                            .max(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                            .sqrt(),
                        })
                    })
                    .flatten();
                AssetQualitySummary {
                    asset: perturbative_remote_total_summary(
                        PosteriorSummary {
                            mean: 0.0,
                            sigma: 0.0,
                        },
                        baseline,
                        technical,
                        face,
                        None,
                        if gate_3d {
                            AssetDomainLabel::Real
                        } else {
                            AssetDomainLabel::Anime
                        },
                        field.perturbative_hyper,
                    ),
                    baseline,
                    semantic: None,
                    vibe: None,
                    technical,
                    face,
                }
            }
        }
    }

    fn choose_pair_against_local_anchor_with_store(
        &self,
        store: &Store,
        field: &SessionField,
        assets: &[AssetRecord],
        anchor: &AssetRecord,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        let explore = self.arena_explore();
        let external_probability = self.config.read().external_probability();
        if external_probability_is_certain(external_probability) {
            let external_pair = self.choose_external_pair_against_anchor(
                store,
                field,
                anchor,
                explore,
                excluded_visual_keys,
            )?;
            if external_pair.is_some() {
                return Ok(external_pair);
            }
            return choose_local_pair_against_anchor(
                anchor,
                assets,
                field,
                store,
                explore,
                excluded_visual_keys,
            );
        }
        let local_pair = choose_local_pair_against_anchor(
            anchor,
            assets,
            field,
            store,
            explore,
            excluded_visual_keys,
        )?;
        if external_probability_is_zero(external_probability) {
            return Ok(local_pair);
        }
        let external_pair = self.choose_external_pair_against_anchor(
            store,
            field,
            anchor,
            explore,
            excluded_visual_keys,
        )?;
        Ok(choose_pair_source(
            &mut rng(),
            external_probability,
            local_pair,
            external_pair,
        ))
    }

    fn choose_external_pair(
        &self,
        store: &Store,
        field: &SessionField,
        assets: &[AssetRecord],
        explore: f32,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        let recent = (assets.len() > ARENA_RECENT_REPEAT_EXCLUDE)
            .then(|| store.recent_arena_asset_ids(field.session.id, ARENA_RECENT_REPEAT_EXCLUDE))
            .transpose()?
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();
        let usable = assets
            .iter()
            .filter(|asset| !recent.contains(&asset.id))
            .filter(|asset| !asset_visual_key_excluded(asset, excluded_visual_keys))
            .cloned()
            .collect::<Vec<_>>();
        let pool = usable;
        if pool.is_empty() {
            return Ok(None);
        }
        let mut rng = rng();
        let anchor_scores = pool
            .iter()
            .map(|asset| field.arena_anchor_score(asset, explore))
            .collect::<Vec<_>>();
        let Some(anchor_index) = sample_softmax_index(
            &mut rng,
            &anchor_scores,
            arena_sampling_temperature(explore),
            arena_uniform_mix(explore),
        ) else {
            return Ok(None);
        };
        let anchor = pool[anchor_index].clone();
        self.choose_external_pair_against_anchor(
            store,
            field,
            &anchor,
            explore,
            excluded_visual_keys,
        )
    }

    fn choose_external_pair_against_anchor(
        &self,
        store: &Store,
        field: &SessionField,
        anchor: &AssetRecord,
        explore: f32,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        let Some(mut scored) =
            self.pick_remote_candidate(store, field, anchor, explore, excluded_visual_keys)?
        else {
            return Ok(None);
        };
        if scored.quality.technical.is_none() && scored.quality.vibe.is_none() {
            let face_oracle = self.ensure_face_oracle(store)?;
            let domain_oracle = self.asset_domain_oracle(store)?;
            let features = Self::lazy_remote_quality_features(scored.candidate.quality_features);
            let technical_head = store.active_quality_model().ok().and_then(|model| {
                store
                    .load_linear_technical_prior_head(&model)
                    .ok()
                    .flatten()
            });
            scored.quality = Self::remote_quality_summary(
                field,
                &scored.candidate.embedding,
                scored.candidate.face_embedding.as_deref(),
                features.as_ref(),
                scored.candidate.quality_cache.as_ref(),
                technical_head.as_ref(),
                Some(&domain_oracle),
                face_oracle.as_ref(),
            );
            scored.utility = scored.quality.asset.mean;
        }
        Ok(Some(ArenaPair {
            left: ArenaCard::Local(ArenaLocalCard {
                utility: field.utility(anchor),
                hearted: field.hearted(&anchor.id),
                quality: field.quality_summary_or_fallback(anchor),
                domain: AssetDomainView::default(),
                asset: anchor.clone(),
            }),
            right: ArenaCard::Remote(ArenaRemoteCard {
                item: scored.candidate.item,
                hearted: false,
                stream_locked: false,
                utility: scored.utility,
                quality: scored.quality,
            }),
        }))
    }

    fn pick_remote_candidate_from_locked_stream(
        &self,
        store: &Store,
        field: &SessionField,
        anchor: &AssetRecord,
        source: &SourceConfig,
        lock: &crate::model::SessionSubsourceLock,
        explore: f32,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ScoredRemoteCandidate>> {
        let mut candidates = store.remote_candidates(
            &lock.source_key,
            self.embedder.model_name(),
            self.embedder.recognition_model_name(),
            field.quality_model,
        )?;
        prune_unmaterialized_remote_candidates(source, &mut candidates)?;
        candidates.retain(|candidate| candidate.item.stream_id == lock.stream_id);
        if candidates.is_empty() {
            return Ok(None);
        }
        let visible_candidates = candidates
            .iter()
            .filter(|candidate| !remote_visual_key_excluded(candidate, excluded_visual_keys))
            .cloned()
            .collect::<Vec<_>>();
        let candidates = if visible_candidates.is_empty() {
            candidates
        } else {
            visible_candidates
        };
        let recent_item_ids = store
            .recent_selected_external_item_ids(self.active.session_id, EXTERNAL_RECENT_EXCLUDE)?;
        let recent_item_set = recent_item_ids.into_iter().collect::<HashSet<_>>();
        let filtered = candidates
            .iter()
            .filter(|candidate| !recent_item_set.contains(&candidate.item.id))
            .cloned()
            .collect::<Vec<_>>();
        let candidates = if filtered.is_empty() {
            candidates
        } else {
            filtered
        };

        let anchor_quality = field.quality_summary_or_fallback(anchor);
        let face_oracle = self.ensure_face_oracle(store)?;
        let domain_oracle = self.asset_domain_oracle(store)?;
        let technical_head = store.active_quality_model().ok().and_then(|model| {
            store
                .load_linear_technical_prior_head(&model)
                .ok()
                .flatten()
        });
        let recent_item_ranks = store
            .recent_selected_external_item_ids(self.active.session_id, EXTERNAL_RECENT_EXCLUDE)?
            .into_iter()
            .rev()
            .enumerate()
            .map(|(rank, item_id)| (item_id, rank))
            .collect::<HashMap<_, _>>();

        let mut contenders = Vec::<ScoredRemoteCandidate>::with_capacity(candidates.len());
        for candidate in candidates {
            let quality = Self::remote_quality_summary(
                field,
                &candidate.embedding,
                candidate.face_embedding.as_deref(),
                candidate.quality_features.as_ref(),
                candidate.quality_cache.as_ref(),
                technical_head.as_ref(),
                Some(&domain_oracle),
                face_oracle.as_ref(),
            );
            let item_recency =
                recency_discount(recent_item_ranks.get(&candidate.item.id).copied(), 0.12);
            let selection_score = remote_candidate_score(
                field,
                anchor_quality.asset,
                quality.asset,
                item_recency,
                explore,
            );
            contenders.push(ScoredRemoteCandidate {
                utility: quality.asset.mean,
                quality,
                candidate,
                selection_score,
            });
        }

        let mut rng = rng();
        let selection_scores = contenders
            .iter()
            .map(|candidate| candidate.selection_score)
            .collect::<Vec<_>>();
        let Some(selected_index) = sample_softmax_index(
            &mut rng,
            &selection_scores,
            arena_sampling_temperature(explore),
            arena_uniform_mix(explore),
        ) else {
            return Ok(None);
        };
        let selected = contenders.swap_remove(selected_index);
        if !selected.candidate.item.path.exists() {
            return Ok(None);
        }
        Ok(Some(selected))
    }

    fn pick_remote_candidate(
        &self,
        store: &Store,
        field: &SessionField,
        anchor: &AssetRecord,
        explore: f32,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ScoredRemoteCandidate>> {
        let sources = self
            .configured_sources()
            .into_iter()
            .filter(|source| source.weight > 0.0)
            .collect::<Vec<_>>();
        let enforce_item_repeat_exclusion =
            sources.iter().try_fold(0usize, |ready_total, source| {
                let (_, _, ready) = store.external_source_counts(&source.source_key())?;
                Ok::<_, anyhow::Error>(ready_total.saturating_add(ready))
            })? > ARENA_RECENT_REPEAT_EXCLUDE;
        let recent_sources = store.recent_selected_external_source_keys(
            self.active.session_id,
            source_recent_exclude_window(sources.len()),
        )?;
        let source_ranks = recent_sources
            .iter()
            .rev()
            .enumerate()
            .map(|(rank, key)| (key.clone(), rank))
            .collect::<HashMap<_, _>>();

        let mut contenders = Vec::<(f32, ScoredRemoteCandidate)>::new();
        for source in sources {
            let Some(candidate) = self.pick_remote_candidate_from_source(
                store,
                field,
                anchor,
                &source,
                explore,
                enforce_item_repeat_exclusion,
                excluded_visual_keys,
            )?
            else {
                continue;
            };
            let source_score = remote_source_score(
                source.weight,
                recency_discount(source_ranks.get(&source.source_key()).copied(), 0.28),
                explore,
            );
            contenders.push((source_score, candidate));
        }

        if contenders.is_empty() {
            return Ok(None);
        }
        let mut rng = rng();
        let weights = contenders
            .iter()
            .map(|(score, _)| *score)
            .collect::<Vec<_>>();
        let Some(index) = sample_softmax_index(
            &mut rng,
            &weights,
            arena_sampling_temperature(explore),
            arena_uniform_mix(explore),
        ) else {
            return Ok(None);
        };
        Ok(Some(contenders.swap_remove(index).1))
    }

    fn pick_remote_candidate_from_source(
        &self,
        store: &Store,
        field: &SessionField,
        anchor: &AssetRecord,
        source: &SourceConfig,
        explore: f32,
        enforce_item_repeat_exclusion: bool,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ScoredRemoteCandidate>> {
        let mut candidates = store.remote_candidates(
            &source.source_key(),
            self.embedder.model_name(),
            self.embedder.recognition_model_name(),
            field.quality_model,
        )?;
        prune_unmaterialized_remote_candidates(source, &mut candidates)?;
        candidates.retain(|candidate| !remote_visual_key_excluded(candidate, excluded_visual_keys));
        if candidates.is_empty() {
            return Ok(None);
        }
        if enforce_item_repeat_exclusion {
            let recent_items = store
                .recent_selected_external_item_ids(
                    self.active.session_id,
                    ARENA_RECENT_REPEAT_EXCLUDE,
                )?
                .into_iter()
                .collect::<HashSet<_>>();
            candidates.retain(|candidate| !recent_items.contains(&candidate.item.id));
            if candidates.is_empty() {
                return Ok(None);
            }
        }
        let recent_item_ids = store
            .recent_selected_external_item_ids(self.active.session_id, EXTERNAL_RECENT_EXCLUDE)?;
        let recent_stream_ids = store.recent_selected_external_stream_ids(
            self.active.session_id,
            EXTERNAL_STREAM_HARD_EXCLUDE,
        )?;
        let recent_stream_set = recent_stream_ids.iter().copied().collect::<HashSet<_>>();
        if !recent_stream_set.is_empty() {
            let candidate_stream_count = candidates
                .iter()
                .map(|candidate| candidate.item.stream_id)
                .collect::<HashSet<_>>()
                .len();
            if candidate_stream_count > recent_stream_set.len() {
                let filtered_count = candidates
                    .iter()
                    .filter(|candidate| !recent_stream_set.contains(&candidate.item.stream_id))
                    .count();
                if filtered_count > 0 {
                    candidates
                        .retain(|candidate| !recent_stream_set.contains(&candidate.item.stream_id));
                }
            }
        }

        let anchor_quality = field.quality_summary_or_fallback(anchor);
        let face_oracle = self.ensure_face_oracle(store)?;
        let domain_oracle = self.asset_domain_oracle(store)?;
        let technical_head = store.active_quality_model().ok().and_then(|model| {
            store
                .load_linear_technical_prior_head(&model)
                .ok()
                .flatten()
        });
        let now = OffsetDateTime::now_utc().unix_timestamp();
        let recent_stream_ids = store.recent_selected_external_stream_ids(
            self.active.session_id,
            EXTERNAL_STREAM_RECENT_EXCLUDE,
        )?;
        let item_ranks = recent_item_ids
            .iter()
            .rev()
            .enumerate()
            .map(|(rank, item_id)| (*item_id, rank))
            .collect::<HashMap<_, _>>();
        let stream_ranks = recent_stream_ids
            .iter()
            .rev()
            .enumerate()
            .map(|(rank, stream_id)| (*stream_id, rank))
            .collect::<HashMap<_, _>>();

        let mut contenders = Vec::<ScoredRemoteCandidate>::with_capacity(candidates.len());
        for candidate in candidates {
            let stream_id = candidate.item.stream_id;
            let stream_image_count = candidate.stream_image_count;
            let stream_last_modified = candidate.stream_last_modified;
            let source_bonuses =
                remote_source_arena_bonuses(source, now, stream_last_modified, stream_image_count);
            let quality = Self::remote_quality_summary(
                field,
                &candidate.embedding,
                candidate.face_embedding.as_deref(),
                candidate.quality_features.as_ref(),
                candidate.quality_cache.as_ref(),
                technical_head.as_ref(),
                Some(&domain_oracle),
                face_oracle.as_ref(),
            );
            let item_recency = recency_discount(item_ranks.get(&candidate.item.id).copied(), 0.12);
            let selection_score = remote_candidate_score(
                field,
                anchor_quality.asset,
                quality.asset,
                item_recency,
                explore,
            );
            contenders.push(ScoredRemoteCandidate {
                utility: quality.asset.mean,
                quality,
                candidate,
                selection_score: remote_stream_score(
                    selection_score,
                    source_bonuses.stream_size,
                    source_bonuses.freshness,
                    recency_discount(stream_ranks.get(&stream_id).copied(), 0.22),
                    explore,
                ),
            });
        }

        if contenders.is_empty() {
            return Ok(None);
        }
        let mut rng = rng();
        let selection_scores = contenders
            .iter()
            .map(|candidate| candidate.selection_score)
            .collect::<Vec<_>>();
        let Some(selected_index) = sample_softmax_index(
            &mut rng,
            &selection_scores,
            arena_sampling_temperature(explore),
            arena_uniform_mix(explore),
        ) else {
            return Ok(None);
        };
        let selected = contenders.swap_remove(selected_index);
        if !selected.candidate.item.path.exists() {
            return Ok(None);
        }
        Ok(Some(selected))
    }

    pub(super) fn note_remote_pair_selected(&self, pair: &ArenaPair) -> anyhow::Result<()> {
        let (local_asset_id, remote) = match (&pair.left, &pair.right) {
            (ArenaCard::Local(local), ArenaCard::Remote(remote))
            | (ArenaCard::Remote(remote), ArenaCard::Local(local)) => {
                (local.asset.id.clone(), remote)
            }
            _ => return Ok(()),
        };
        let remote_item_id = remote.item.id;
        let session_id = self.active.session_id;
        let corpus_id = self.active.corpus_id;
        self.with_write_store("note_external_selected", move |store| {
            store.note_external_selected(session_id, corpus_id, remote_item_id, &local_asset_id)
        })?;
        if remote.stream_locked {
            let store = self.read_store()?;
            if let Some(counts) = store.external_stream_frontier_counts(
                &remote.item.source_key,
                remote.item.stream_id,
                self.embedder.model_name(),
            )? && counts.ready_items < Self::locked_stream_ready_target(counts.live_items)
            {
                self.request_locked_stream_refresh(crate::model::SessionSubsourceLock {
                    source_key: remote.item.source_key.clone(),
                    stream_id: remote.item.stream_id,
                });
            }
        }
        Ok(())
    }
}

fn extension_or_fallback(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .filter(|ext| !ext.is_empty())
        .unwrap_or_else(|| "img".to_owned())
}

pub(super) fn choose_pair_source<R: rand::Rng + ?Sized, T>(
    rng: &mut R,
    external_probability: f32,
    local_pair: Option<T>,
    external_pair: Option<T>,
) -> Option<T> {
    match (local_pair, external_pair) {
        (None, None) => None,
        (Some(pair), None) | (None, Some(pair)) => Some(pair),
        (Some(local_pair), Some(external_pair)) => {
            if rng.random::<f32>() < external_probability {
                Some(external_pair)
            } else {
                Some(local_pair)
            }
        }
    }
}

fn external_probability_is_zero(value: f32) -> bool {
    value <= f32::EPSILON
}

fn external_probability_is_certain(value: f32) -> bool {
    value >= 1.0 - f32::EPSILON
}

fn sanitize_source_key(source_key: &str) -> String {
    source_key
        .chars()
        .map(|glyph| {
            if glyph.is_ascii_alphanumeric() {
                glyph
            } else {
                '-'
            }
        })
        .collect()
}

fn source_recent_exclude_window(source_count: usize) -> usize {
    source_count.clamp(1, EXTERNAL_SOURCE_RECENT_EXCLUDE_CAP)
}

fn recency_discount(rank: Option<usize>, floor: f32) -> f32 {
    rank.map_or(1.0, |rank| {
        floor + (1.0 - floor) * (rank as f32 / (rank as f32 + 1.0))
    })
}

fn remote_visual_key_excluded(
    candidate: &RemoteCandidate,
    excluded_visual_keys: &HashSet<VisualKey>,
) -> bool {
    candidate
        .item
        .visual_key
        .as_ref()
        .is_some_and(|visual_key| excluded_visual_keys.contains(visual_key))
}

fn pair_balance_pull(anchor_quality: f32, candidate_quality: f32) -> f32 {
    1.0 / (1.0 + ((anchor_quality - candidate_quality) / 1.3).abs())
}

fn remote_accept_probability(field: &SessionField, summary: PosteriorSummary) -> f32 {
    let (frontier_mean, frontier_variance) = match field.quality_model {
        QualityFormalVersion::HierarchicalPerturbativeV2
        | QualityFormalVersion::HierarchicalPerturbativeV3 => field.perturbative_session.map_or(
            (
                field.session.frontier,
                crate::quality::PERTURBATIVE_THRESHOLD_PRIOR_VARIANCE,
            ),
            |session| (session.threshold_mean, session.threshold_variance),
        ),
        _ => (
            field
                .hierarchical_session
                .map_or(field.session.frontier, |session| session.frontier_mean),
            field.hierarchical_session.map_or(
                crate::quality::HIERARCHICAL_FRONTIER_PRIOR_VARIANCE,
                |session| session.frontier_variance,
            ),
        ),
    };
    crate::quality::posterior_accept_probability(
        summary.mean,
        summary.sigma.powi(2),
        frontier_mean,
        frontier_variance,
        crate::quality::HIERARCHICAL_UNARY_ACCEPT_BETA,
    )
}

fn remote_candidate_score(
    field: &SessionField,
    anchor_quality: PosteriorSummary,
    candidate_quality: PosteriorSummary,
    item_recency: f32,
    explore: f32,
) -> f32 {
    candidate_quality.mean
        + explore
            * (candidate_quality.sigma * ARENA_EXPLORE_SIGMA_WEIGHT
                + pair_balance_pull(anchor_quality.mean, candidate_quality.mean)
                    * ARENA_REMOTE_PAIR_BALANCE_WEIGHT
                + remote_accept_probability(field, candidate_quality) * ARENA_REMOTE_ACCEPT_WEIGHT
                + item_recency * ARENA_REMOTE_ITEM_RECENCY_WEIGHT)
}

fn remote_stream_score(
    candidate_score: f32,
    stream_size: f32,
    freshness: f32,
    recency: f32,
    explore: f32,
) -> f32 {
    candidate_score
        + explore
            * (stream_size * ARENA_REMOTE_STREAM_SIZE_WEIGHT
                + freshness * ARENA_REMOTE_STREAM_FRESHNESS_WEIGHT
                + recency * ARENA_REMOTE_STREAM_RECENCY_WEIGHT)
}

fn remote_source_score(source_weight: f32, source_recency: f32, explore: f32) -> f32 {
    source_weight_bonus(source_weight)
        + explore * source_recency * ARENA_REMOTE_SOURCE_RECENCY_WEIGHT
}

pub(super) fn remote_source_arena_bonuses(
    source: &SourceConfig,
    now: i64,
    stream_last_modified: i64,
    stream_image_count: u32,
) -> RemoteSourceArenaBonuses {
    if source.local_directory().is_some() {
        return RemoteSourceArenaBonuses {
            stream_size: 0.0,
            freshness: 0.0,
        };
    }
    RemoteSourceArenaBonuses {
        stream_size: stream_size_bias(stream_image_count),
        freshness: freshness_pull(now, stream_last_modified),
    }
}

fn source_weight_bonus(source_weight: f32) -> f32 {
    ARENA_REMOTE_SOURCE_WEIGHT * source_weight.max(0.0).ln_1p()
}

fn stream_size_bias(image_count: u32) -> f32 {
    (image_count.clamp(1, 25) as f32).sqrt() / 5.0_f32.sqrt()
}

fn freshness_pull(now: i64, last_modified: i64) -> f32 {
    let age_days = (now - last_modified).max(0) as f32 / 86_400.0;
    1.0 / (1.0 + age_days / 5.0)
}

fn prune_unmaterialized_remote_candidates(
    _source: &SourceConfig,
    candidates: &mut Vec<RemoteCandidate>,
) -> anyhow::Result<()> {
    let stale = candidates
        .iter()
        .filter(|candidate| !candidate.item.path.exists())
        .map(|candidate| candidate.item.id)
        .collect::<Vec<_>>();
    if !stale.is_empty() {
        let stale = stale.into_iter().collect::<HashSet<_>>();
        candidates.retain(|candidate| !stale.contains(&candidate.item.id));
    }
    Ok(())
}
