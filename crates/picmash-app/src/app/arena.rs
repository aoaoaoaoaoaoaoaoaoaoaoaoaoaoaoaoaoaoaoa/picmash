use super::*;
use crate::identity::VisualKey;
use std::collections::HashSet;

impl AppState {
    pub fn arena_target(&self) -> anyhow::Result<RedirectTarget> {
        self.redirect_target_for_next_pair(&HashSet::new())
    }

    pub fn arena_prefetch_target(&self) -> anyhow::Result<RedirectTarget> {
        self.arena_prefetch_target_excluding(&HashSet::new())
    }

    pub fn arena_prefetch_target_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
    ) -> anyhow::Result<RedirectTarget> {
        self.arena_prefetch_target_preserving_local_anchor_excluding(local_anchor, &HashSet::new())
    }

    pub fn arena_empty(&self) -> anyhow::Result<ArenaView> {
        let store = self.read_store()?;
        let _field = self.session_field(&store)?;
        Ok(ArenaView {
            pair: None,
            cluster: None,
        })
    }

    pub fn arena_pair(
        &self,
        left: &ArenaHandle,
        right: &ArenaHandle,
    ) -> anyhow::Result<Option<ArenaView>> {
        let store = self.read_store()?;
        let field = self.session_field(&store)?;
        let Some(left) = self.load_arena_card(&store, &field, left)? else {
            return Ok(None);
        };
        let Some(right) = self.load_arena_card(&store, &field, right)? else {
            return Ok(None);
        };
        if left.handle() == right.handle() {
            return Ok(None);
        }
        let cluster = match &right {
            ArenaCard::Remote(remote_card) => {
                self.cluster_around_remote(&store, remote_card.item.id)?
            }
            ArenaCard::Local(_) => None,
        };

        Ok(Some(ArenaView {
            pair: Some(ArenaPair { left, right }),
            cluster,
        }))
    }

    pub fn board(&self) -> anyhow::Result<BoardView> {
        let store = self.read_store()?;
        let field = self.session_field(&store)?;
        let mut entries = visible_assets(&store, self.active.corpus_id)?
            .into_iter()
            .map(|asset| field.board_entry(asset))
            .collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| {
            rhs.sampling_pull
                .total_cmp(&lhs.sampling_pull)
                .then_with(|| rhs.session_focus.total_cmp(&lhs.session_focus))
                .then_with(|| rhs.global_score.total_cmp(&lhs.global_score))
                .then_with(|| lhs.asset.id.0.cmp(&rhs.asset.id.0))
        });
        Ok(BoardView { entries })
    }

    pub fn rotate_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        let asset_id = asset_id.clone();
        let session_id = self.active.session_id;
        self.with_write_store("rotate_asset", move |store| {
            store.rotate_asset(&asset_id, direction)?;
            store.touch_session(session_id)
        })
    }

    pub fn rotate_arena_handle(&self, handle: &ArenaHandle, direction: i32) -> anyhow::Result<()> {
        let handle = handle.clone();
        let corpus_id = self.active.corpus_id;
        let session_id = self.active.session_id;
        self.with_write_store("rotate_arena_handle", move |store| {
            match handle {
                ArenaHandle::Local(asset_id) => {
                    if store.corpus_asset(corpus_id, &asset_id)?.is_none() {
                        return Ok(());
                    }
                    store.rotate_asset(&asset_id, direction)?;
                }
                ArenaHandle::Remote(item_id) => {
                    if store.remote_item(item_id)?.is_none() {
                        return Ok(());
                    }
                    store.rotate_external_item(item_id, direction)?;
                }
            }
            store.touch_session(session_id)
        })
    }

    pub fn hide_asset(&self, asset_id: &AssetId, hidden: bool) -> anyhow::Result<RedirectTarget> {
        let asset_id = asset_id.clone();
        let corpus_id = self.active.corpus_id;
        let session_id = self.active.session_id;
        let present = self.with_write_store("hide_asset", move |store| {
            let present = store.corpus_asset(corpus_id, &asset_id)?.is_some();
            if !present {
                return Ok(false);
            }
            store.set_hidden(corpus_id, &asset_id, hidden)?;
            store.touch_session(session_id)?;
            Ok(true)
        })?;
        if !present {
            return redirect_target_for_pair(None);
        }
        self.purge_explore_vectors();
        self.purge_all_explore_layouts();
        self.invalidate_session_field_cache();
        self.redirect_target_for_next_pair(&HashSet::new())
    }

    pub fn hide_arena_handle(
        &self,
        handle: &ArenaHandle,
        hidden: bool,
        cluster_ids: &[RemoteItemId],
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Local(asset_id) => self.hide_asset(asset_id, hidden),
            ArenaHandle::Remote(_item_id) => {
                let local_anchor = surviving_local_anchor(handle, pair_left, pair_right).cloned();
                let lock_handle = handle.clone();
                let cluster_ids = cluster_ids.to_vec();
                let anchor_for_write = local_anchor.clone();
                let session_id = self.active.session_id;
                let corpus_id = self.active.corpus_id;
                self.with_write_store("hide_remote_handle", move |store| {
                    if hidden {
                        store.reject_external_item(
                            session_id,
                            corpus_id,
                            match lock_handle {
                                ArenaHandle::Remote(item_id) => item_id,
                                ArenaHandle::Local(_) => unreachable!("remote branch required"),
                            },
                            anchor_for_write.as_ref(),
                            ExternalEventKind::Rejected,
                        )?;
                        for satellite_id in &cluster_ids {
                            store.reject_external_item(
                                session_id,
                                corpus_id,
                                *satellite_id,
                                anchor_for_write.as_ref(),
                                ExternalEventKind::Rejected,
                            )?;
                        }
                    }
                    store.touch_session(session_id)?;
                    Ok(())
                })?;
                if hidden {
                    self.purge_duplicate_frontier();
                }
                self.invalidate_session_field_cache();
                self.redirect_target_preserving_local_anchor(local_anchor.as_ref())
            }
        }
    }

    pub fn veto_external_thread_for_handle(
        &self,
        handle: &ArenaHandle,
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        let ArenaHandle::Remote(item_id) = handle else {
            return self.arena_target();
        };
        let item = {
            let store = Store::open_hot(&self.db_path)?;
            store
                .remote_item(*item_id)?
                .with_context(|| format!("missing remote item {}", item_id.0))?
        };
        let local_anchor = surviving_local_anchor(handle, pair_left, pair_right).cloned();
        let item_id = *item_id;
        let anchor_for_write = local_anchor.clone();
        let session_id = self.active.session_id;
        let corpus_id = self.active.corpus_id;
        let locked_source_key = item.source_key.clone();
        let locked_source_key_for_check = locked_source_key.clone();
        let locked_stream_id = item.stream_id;
        let cleared_lock = self.with_write_store("veto_external_thread", move |store| {
            let cleared_lock = store
                .session_subsource_lock(session_id)?
                .is_some_and(|lock| {
                    lock.source_key == locked_source_key_for_check
                        && lock.stream_id == locked_stream_id
                });
            store.block_external_stream(
                session_id,
                corpus_id,
                item_id,
                anchor_for_write.as_ref(),
            )?;
            if cleared_lock {
                store.clear_session_subsource_lock(session_id)?;
                info!(
                    source = %item.source_key,
                    stream_id = item.stream_id,
                    "cleared arena external subsource lock after veto"
                );
            }
            info!(
                source = %item.source_key,
                thread_no = item.thread_no,
                title = %item.stream_title,
                "blocked external stream"
            );
            store.touch_session(session_id)?;
            Ok(cleared_lock)
        })?;
        self.purge_duplicate_frontier();
        self.invalidate_session_field_cache();
        if cleared_lock {
            info!(
                source = %locked_source_key,
                stream_id = locked_stream_id,
                "veto lifted the active subsource lock"
            );
        }
        self.redirect_target_preserving_local_anchor(local_anchor.as_ref())
    }

    pub fn set_external_subsource_lock_for_handle(
        &self,
        handle: &ArenaHandle,
        active: bool,
        pair_left: &ArenaHandle,
        pair_right: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Remote(item_id) => {
                self.set_external_subsource_lock(*item_id, active)?;
                Ok(RedirectTarget::ArenaPair {
                    left: pair_left.clone(),
                    right: pair_right.clone(),
                })
            }
            ArenaHandle::Local(_) => Ok(RedirectTarget::ArenaPair {
                left: pair_left.clone(),
                right: pair_right.clone(),
            }),
        }
    }

    pub fn hide_asset_from_board(&self, asset_id: &AssetId) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        let asset_id = asset_id.clone();
        let corpus_id = self.active.corpus_id;
        let session_id = self.active.session_id;
        self.with_write_store("hide_asset_from_board", move |store| {
            store.set_hidden(corpus_id, &asset_id, true)?;
            store.touch_session(session_id)
        })?;
        self.purge_explore_vectors();
        self.purge_all_explore_layouts();
        self.invalidate_session_field_cache();
        Ok(())
    }

    pub fn set_asset_domain_label(
        &self,
        asset_id: &AssetId,
        label: AssetDomainLabel,
    ) -> anyhow::Result<()> {
        if self.maybe_image_asset(asset_id)?.is_none() {
            return Ok(());
        }
        let asset_id = asset_id.clone();
        let asset_id_for_write = asset_id.clone();
        let session_id = self.active.session_id;
        self.with_write_store("set_asset_domain_label", move |store| {
            store.set_asset_domain_label(&asset_id_for_write, label)?;
            store.touch_session(session_id)
        })?;
        let status = self
            .retrain_asset_domain_oracle(&self.read_store()?)?
            .status();
        info!(
            asset_id = %asset_id.0,
            label = label.as_str(),
            real_labels = status.real_labels,
            anime_labels = status.anime_labels,
            trained = status.trained,
            "updated asset domain label"
        );
        Ok(())
    }

    pub fn nudge_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        self.apply_unary_feedback(asset_id, UnaryFeedback::from_nudge(direction)?)
    }

    pub fn set_heart_asset(&self, asset_id: &AssetId, active: bool) -> anyhow::Result<()> {
        let updated = self.with_locked_store_write(|store| {
            if !active {
                return Ok(None);
            }
            let mut field = self.session_field(store)?;
            let Some(mut asset) = store
                .corpus_assets(self.active.corpus_id)?
                .into_iter()
                .find(|candidate| candidate.id == *asset_id)
            else {
                return Ok(None);
            };

            if asset.is_hearted {
                return Ok(None);
            }

            asset.is_hearted = true;
            asset.heart_count = asset.heart_count.saturating_add(1);
            field.session.hearts = field.session.hearts.saturating_add(1);
            field.hearted_assets.insert(asset_id.clone());

            store.persist_heart_step(&field.session, &asset, asset_id, active)?;
            Ok(Some(field))
        })?;
        if let Some(field) = updated {
            self.replace_session_field_cache(field);
        }
        Ok(())
    }

    pub fn set_heart_arena_handle(
        &self,
        handle: &ArenaHandle,
        active: bool,
    ) -> anyhow::Result<RedirectTarget> {
        match handle {
            ArenaHandle::Local(asset_id) => {
                self.set_heart_asset(asset_id, active)?;
                Ok(RedirectTarget::ArenaRoot)
            }
            ArenaHandle::Remote(item_id) => {
                if active {
                    self.enshrine_remote_item(*item_id, true)?;
                }
                self.arena_target()
            }
        }
    }

    fn apply_unary_feedback(
        &self,
        asset_id: &AssetId,
        feedback: UnaryFeedback,
    ) -> anyhow::Result<()> {
        let updated = self.with_locked_store_write(|store| {
            let mut field = self.session_field(store)?;
            let Some(mut asset) = store
                .corpus_assets(self.active.corpus_id)?
                .into_iter()
                .find(|candidate| candidate.id == *asset_id)
            else {
                return Ok(None);
            };

            let exact_offset = field.exact_offset(asset_id);
            let utility_before = field.utility(&asset);
            let frontier_before = match field.quality_model {
                QualityFormalVersion::HierarchicalPerturbativeV2
                | QualityFormalVersion::HierarchicalPerturbativeV3 => field
                    .perturbative_session
                    .map_or(field.session.frontier, |session| session.threshold_mean),
                _ => field
                    .hierarchical_session
                    .map_or(field.session.frontier, |session| session.frontier_mean),
            };
            let legacy_feedback = feedback.into_legacy();
            let tuning = legacy_feedback.tuning(utility_before, frontier_before);

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalPerturbativeV2
                    | QualityFormalVersion::HierarchicalPerturbativeV3
            ) {
                let next_offset =
                    exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);
                field.session.nudges += 1;
                store.persist_nudge_step(
                    &field.session,
                    &asset,
                    asset_id,
                    legacy_feedback.direction(),
                    utility_before,
                    frontier_before,
                    tuning.signal,
                    next_offset,
                    None,
                    None,
                )?;
                return Ok(Some(field));
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalGaussianV1
            ) {
                let mut session_quality = field.hierarchical_session.with_context(|| {
                    format!(
                        "missing hierarchical session posterior {}",
                        field.session.id.0
                    )
                })?;
                let next_offset =
                    exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);
                session_quality.frontier_mean += tuning.frontier_rate
                    * (-tuning.signal - LEGACY_L2_FRONTIER * session_quality.frontier_mean);
                field.session.frontier = session_quality.frontier_mean;
                field.session.nudges += 1;

                let embedding = field.embedding(asset_id).map(ToOwned::to_owned);
                let mut embedding_head = field.embedding_head.clone().or_else(|| {
                    embedding.as_ref().map(|vector| {
                        SessionEmbeddingHead::zero(
                            self.embedder.model_name().to_owned(),
                            vector.len(),
                        )
                    })
                });
                if let (Some(head), Some(vector)) = (&mut embedding_head, embedding.as_deref()) {
                    head.unary_step(
                        vector,
                        tuning.signal,
                        tuning.head_rate,
                        LEGACY_L2_SESSION_HEAD,
                    );
                }

                let asset_cache = field
                    .hierarchical_assets
                    .get(asset_id)
                    .copied()
                    .with_context(|| {
                        format!("missing hierarchical posterior for {}", asset_id.0)
                    })?;
                store.persist_hierarchical_nudge_step(
                    &field.session,
                    &asset,
                    asset_id,
                    legacy_feedback.direction(),
                    utility_before,
                    frontier_before,
                    tuning.signal,
                    next_offset,
                    embedding_head.as_ref(),
                    &hierarchical_asset_cache(
                        &asset_cache,
                        hierarchical_face_summary(field.dominant_faces.get(asset_id)),
                    ),
                    &hierarchical_session_cache(&session_quality),
                )?;
                if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
                    field.exact_offsets.remove(asset_id);
                } else {
                    field.exact_offsets.insert(asset_id.clone(), next_offset);
                }
                field.embedding_head = embedding_head;
                field.hierarchical_session = Some(session_quality);
                return Ok(Some(field));
            }

            let embedding = field.embedding(asset_id).map(ToOwned::to_owned);
            let mut projection =
                projection_state(store, self.embedder.model_name(), embedding.as_deref())?;
            let prior = legacy_projection_prior(projection.as_ref(), embedding.as_deref());
            let asset_before = asset.clone();

            legacy_batter_asset(
                &mut asset.alpha,
                &mut asset.coords,
                &field.session.mood,
                &prior,
                tuning.signal,
                tuning.alpha_rate,
                tuning.coord_rate,
            );
            legacy_shove_mood(
                &mut field.session.mood,
                &asset_before.coords,
                tuning.signal,
                tuning.mood_rate,
            );
            field.session.frontier += tuning.frontier_rate
                * (-tuning.signal - LEGACY_L2_FRONTIER * field.session.frontier);

            let next_offset =
                exact_offset + tuning.offset_rate * (tuning.signal - L2_OFFSET * exact_offset);

            let mut embedding_head = field.embedding_head.clone().or_else(|| {
                embedding.as_ref().map(|vector| {
                    SessionEmbeddingHead::zero(self.embedder.model_name().to_owned(), vector.len())
                })
            });
            if let (Some(head), Some(vector)) = (&mut embedding_head, embedding.as_deref()) {
                head.unary_step(
                    vector,
                    tuning.signal,
                    tuning.head_rate,
                    LEGACY_L2_SESSION_HEAD,
                );
            }
            if let (Some(model), Some(vector)) = (&mut projection, embedding.as_deref()) {
                model.gradient_step(
                    vector,
                    &asset.coords,
                    tuning.projection_rate,
                    LEGACY_PROJECTION_WEIGHT_DECAY,
                );
            }

            field.session.nudges += 1;
            store.persist_nudge_step(
                &field.session,
                &asset,
                asset_id,
                legacy_feedback.direction(),
                utility_before,
                frontier_before,
                tuning.signal,
                next_offset,
                projection.as_ref(),
                embedding_head.as_ref(),
            )?;

            if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
                field.exact_offsets.remove(asset_id);
            } else {
                field.exact_offsets.insert(asset_id.clone(), next_offset);
            }
            field.embedding_head = embedding_head;
            Ok(Some(field))
        })?;
        if let Some(field) = updated {
            self.replace_session_field_cache(field);
        }
        Ok(())
    }

    pub fn image_asset(&self, asset_id: &AssetId) -> anyhow::Result<AssetRecord> {
        self.read_store()?
            .corpus_asset(self.active.corpus_id, asset_id)?
            .with_context(|| {
                format!(
                    "asset {} not found in corpus {}",
                    asset_id.0, self.active.corpus_id.0
                )
            })
    }

    pub fn maybe_image_asset(&self, asset_id: &AssetId) -> anyhow::Result<Option<AssetRecord>> {
        self.read_store()?
            .corpus_asset(self.active.corpus_id, asset_id)
    }

    pub fn arena_handle_is_live(&self, handle: &ArenaHandle) -> anyhow::Result<bool> {
        let store = self.read_store()?;
        Ok(match handle {
            ArenaHandle::Local(asset_id) => store
                .corpus_asset(self.active.corpus_id, asset_id)?
                .filter(|asset| !asset.hidden && asset.path.exists())
                .is_some(),
            ArenaHandle::Remote(item_id) => store
                .arena_remote_item(*item_id)?
                .is_some_and(|item| item.path.exists()),
        })
    }

    pub fn maybe_remote_item(
        &self,
        item_id: RemoteItemId,
    ) -> anyhow::Result<Option<crate::model::RemoteItemRecord>> {
        self.read_store()?.remote_item(item_id)
    }

    pub fn hearted_assets(&self) -> anyhow::Result<HashSet<AssetId>> {
        self.read_store()?.hearted_assets()
    }

    pub fn vote(
        &self,
        left: &ArenaHandle,
        right: &ArenaHandle,
        winner: &ArenaHandle,
    ) -> anyhow::Result<RedirectTarget> {
        match (left, right) {
            (ArenaHandle::Local(left_id), ArenaHandle::Local(right_id)) => {
                if let ArenaHandle::Local(winner_id) = winner {
                    return self.vote_local(left_id, right_id, winner_id);
                }
                bail!("remote winner is impossible in a local duel");
            }
            (ArenaHandle::Local(local_id), ArenaHandle::Remote(remote_id))
            | (ArenaHandle::Remote(remote_id), ArenaHandle::Local(local_id)) => {
                self.vote_remote_duel(local_id, *remote_id, winner)
            }
            (ArenaHandle::Remote(_), ArenaHandle::Remote(_)) => {
                bail!("remote-vs-remote arena duels are forbidden");
            }
        }
    }

    fn vote_local(
        &self,
        left_id: &AssetId,
        right_id: &AssetId,
        winner_id: &AssetId,
    ) -> anyhow::Result<RedirectTarget> {
        let updated = self.with_locked_store_write(|store| {
            let corpus_id = self.active.corpus_id;
            let mut field = self.session_field(store)?;
            let assets = store.corpus_assets(corpus_id)?;
            let mut left = assets
                .iter()
                .find(|asset| asset.id == *left_id)
                .cloned()
                .with_context(|| format!("missing left asset {}", left_id.0))?;
            let mut right = assets
                .iter()
                .find(|asset| asset.id == *right_id)
                .cloned()
                .with_context(|| format!("missing right asset {}", right_id.0))?;

            if left.id == right.id {
                bail!("cannot compare an asset against itself");
            }
            if winner_id != &left.id && winner_id != &right.id {
                bail!("winner is not one of the compared assets");
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalPerturbativeV2
                    | QualityFormalVersion::HierarchicalPerturbativeV3
            ) {
                let left_before = left.clone();
                let right_before = right.clone();
                let left_utility = field.utility(&left);
                let right_utility = field.utility(&right);
                let left_won = winner_id == &left.id;
                left.compare_count += 1;
                right.compare_count += 1;
                if left_won {
                    left.win_count += 1;
                } else {
                    right.win_count += 1;
                }
                field.session.comparisons += 1;
                store.persist_duel_step(
                    &field.session,
                    &left_before,
                    &right_before,
                    &left,
                    &right,
                    winner_id,
                    left_utility,
                    right_utility,
                    None,
                    None,
                )?;
                return Ok(Some(field));
            }

            if matches!(
                field.quality_model,
                QualityFormalVersion::HierarchicalGaussianV1
            ) {
                let mut left_quality = field
                    .hierarchical_assets
                    .get(&left.id)
                    .copied()
                    .with_context(|| format!("missing hierarchical posterior for {}", left.id.0))?;
                let mut right_quality = field
                    .hierarchical_assets
                    .get(&right.id)
                    .copied()
                    .with_context(|| {
                        format!("missing hierarchical posterior for {}", right.id.0)
                    })?;
                let mut session_quality = field.hierarchical_session.with_context(|| {
                    format!(
                        "missing hierarchical session posterior {}",
                        field.session.id.0
                    )
                })?;
                let left_face = hierarchical_face_summary(field.dominant_faces.get(&left.id));
                let right_face = hierarchical_face_summary(field.dominant_faces.get(&right.id));
                let left_before = left.clone();
                let right_before = right.clone();
                let left_utility = field.utility(&left);
                let right_utility = field.utility(&right);
                let left_won = winner_id == &left.id;
                let outcome = if left_won { 1.0 } else { -1.0 };
                let delta_mean = left_utility - right_utility;
                let delta_variance = hierarchical_session_utility_variance(
                    &left_quality,
                    &session_quality,
                    left_face,
                ) + hierarchical_session_utility_variance(
                    &right_quality,
                    &session_quality,
                    right_face,
                );
                let moments = crate::quality::gaussian_duel_moment_match(
                    delta_mean,
                    delta_variance,
                    outcome,
                    crate::quality::HIERARCHICAL_DUEL_BETA,
                )
                .context("moment-matching hierarchical duel update")?;
                crate::quality::diagonal_adf_update(
                    &mut left_quality.baseline_mean,
                    &mut left_quality.baseline_variance,
                    1.0,
                    outcome,
                    moments,
                );
                crate::quality::diagonal_adf_update(
                    &mut right_quality.baseline_mean,
                    &mut right_quality.baseline_variance,
                    -1.0,
                    outcome,
                    moments,
                );
                for axis in 0..LATENT_DIM {
                    crate::quality::diagonal_adf_update(
                        &mut left_quality.mood_loading_mean[axis],
                        &mut left_quality.mood_loading_variance[axis],
                        session_quality.semantic_mood_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut right_quality.mood_loading_mean[axis],
                        &mut right_quality.mood_loading_variance[axis],
                        -session_quality.semantic_mood_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut session_quality.semantic_mood_mean[axis],
                        &mut session_quality.semantic_mood_variance[axis],
                        left_quality.mood_loading_mean[axis]
                            - right_quality.mood_loading_mean[axis],
                        outcome,
                        moments,
                    );
                }
                if let (Some(mean), Some(variance)) = (
                    &mut left_quality.technical_mean,
                    &mut left_quality.technical_variance,
                ) {
                    crate::quality::diagonal_adf_update(
                        mean,
                        variance,
                        crate::quality::HIERARCHICAL_TECH_WEIGHT,
                        outcome,
                        moments,
                    );
                }
                if let (Some(mean), Some(variance)) = (
                    &mut right_quality.technical_mean,
                    &mut right_quality.technical_variance,
                ) {
                    crate::quality::diagonal_adf_update(
                        mean,
                        variance,
                        -crate::quality::HIERARCHICAL_TECH_WEIGHT,
                        outcome,
                        moments,
                    );
                }
                for axis in 0..crate::quality_features::VIBE_DESCRIPTOR_DIM {
                    crate::quality::diagonal_adf_update(
                        &mut left_quality.vibe_mean[axis],
                        &mut left_quality.vibe_variance[axis],
                        session_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut right_quality.vibe_mean[axis],
                        &mut right_quality.vibe_variance[axis],
                        -session_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                    crate::quality::diagonal_adf_update(
                        &mut session_quality.vibe_mean[axis],
                        &mut session_quality.vibe_variance[axis],
                        left_quality.vibe_mean[axis] - right_quality.vibe_mean[axis],
                        outcome,
                        moments,
                    );
                }
                let left_face_id = field
                    .dominant_faces
                    .get(&left.id)
                    .map(|identity| identity.id);
                let right_face_id = field
                    .dominant_faces
                    .get(&right.id)
                    .map(|identity| identity.id);
                let mut touched_subjects = BTreeSet::new();
                match (left_face_id, right_face_id) {
                    (Some(left_id), Some(right_id)) if left_id != right_id => {
                        if let Some(identity) = field.dominant_faces.get(&left.id) {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                crate::quality::hierarchical_face_backflow_coeff(),
                                outcome,
                                moments,
                            );
                            for entry in Arc::make_mut(&mut field.dominant_faces)
                                .values_mut()
                                .filter(|entry| entry.id == left_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(left_id);
                        }
                        if let Some(identity) = field.dominant_faces.get(&right.id) {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                -crate::quality::hierarchical_face_backflow_coeff(),
                                outcome,
                                moments,
                            );
                            for entry in Arc::make_mut(&mut field.dominant_faces)
                                .values_mut()
                                .filter(|entry| entry.id == right_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(right_id);
                        }
                    }
                    (Some(identity_id), None) | (None, Some(identity_id)) => {
                        let coefficient = if left_face_id == Some(identity_id) {
                            crate::quality::hierarchical_face_backflow_coeff()
                        } else {
                            -crate::quality::hierarchical_face_backflow_coeff()
                        };
                        if let Some(identity) = field
                            .dominant_faces
                            .values()
                            .find(|entry| entry.id == identity_id)
                            .cloned()
                        {
                            let mut mean = identity.beauty.mean;
                            let mut variance = identity
                                .beauty
                                .sigma
                                .powi(2)
                                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
                            crate::quality::diagonal_adf_update(
                                &mut mean,
                                &mut variance,
                                coefficient,
                                outcome,
                                moments,
                            );
                            for entry in Arc::make_mut(&mut field.dominant_faces)
                                .values_mut()
                                .filter(|entry| entry.id == identity_id)
                            {
                                entry.beauty.mean = mean;
                                entry.beauty.sigma = variance.sqrt();
                            }
                            touched_subjects.insert(identity_id);
                        }
                    }
                    _ => {}
                }
                if !touched_subjects.is_empty() {
                    let snapshot = touched_subjects
                        .into_iter()
                        .filter_map(|identity_id| {
                            field
                                .dominant_faces
                                .values()
                                .find(|entry| entry.id == identity_id)
                                .map(|entry| (entry.id, entry.beauty, entry.duel_count))
                        })
                        .collect::<Vec<_>>();
                    store.save_identity_beauty_snapshot(&snapshot)?;
                }
                let left_face = hierarchical_face_summary(field.dominant_faces.get(&left.id));
                let right_face = hierarchical_face_summary(field.dominant_faces.get(&right.id));

                left.compare_count += 1;
                right.compare_count += 1;
                if left_won {
                    left.win_count += 1;
                } else {
                    right.win_count += 1;
                }
                left.alpha = hierarchical_canonical_mean(&left_quality, left_face);
                right.alpha = hierarchical_canonical_mean(&right_quality, right_face);
                left.coords = left_quality.mood_loading_mean;
                right.coords = right_quality.mood_loading_mean;
                field.session.comparisons += 1;
                field.session.mood = session_quality.semantic_mood_mean;
                field.session.frontier = session_quality.frontier_mean;

                let left_embedding = field.embedding(&left.id).map(ToOwned::to_owned);
                let right_embedding = field.embedding(&right.id).map(ToOwned::to_owned);
                let mut embedding_head = field.embedding_head.clone().or_else(|| {
                    left_embedding
                        .as_ref()
                        .or(right_embedding.as_ref())
                        .map(|vector| {
                            SessionEmbeddingHead::zero(
                                self.embedder.model_name().to_owned(),
                                vector.len(),
                            )
                        })
                });
                let logistic_err = if left_won { 1.0 } else { 0.0 } - sigmoid(delta_mean);
                if let (Some(head), Some(lhs), Some(rhs)) = (
                    &mut embedding_head,
                    left_embedding.as_deref(),
                    right_embedding.as_deref(),
                ) {
                    head.contrast_step(
                        lhs,
                        rhs,
                        logistic_err,
                        LEGACY_LR_DUEL_HEAD,
                        LEGACY_L2_SESSION_HEAD,
                    );
                }

                store.persist_hierarchical_duel_step(
                    &field.session,
                    &left_before,
                    &right_before,
                    &left,
                    &right,
                    winner_id,
                    left_utility,
                    right_utility,
                    embedding_head.as_ref(),
                    &hierarchical_asset_cache(&left_quality, left_face),
                    &hierarchical_asset_cache(&right_quality, right_face),
                    &hierarchical_session_cache(&session_quality),
                )?;
                field.embedding_head = embedding_head;
                Arc::make_mut(&mut field.hierarchical_assets).insert(left.id.clone(), left_quality);
                Arc::make_mut(&mut field.hierarchical_assets)
                    .insert(right.id.clone(), right_quality);
                field.hierarchical_session = Some(session_quality);
                return Ok(Some(field));
            }

            let left_before = left.clone();
            let right_before = right.clone();
            let left_utility = field.utility(&left);
            let right_utility = field.utility(&right);
            let left_won = winner_id == &left.id;
            let y = if left_won { 1.0 } else { 0.0 };
            let err = y - sigmoid(left_utility - right_utility);
            let coord_gap = subtract(&left.coords, &right.coords);

            let left_embedding = field.embedding(&left.id).map(ToOwned::to_owned);
            let right_embedding = field.embedding(&right.id).map(ToOwned::to_owned);
            let mut projection = projection_state(
                store,
                self.embedder.model_name(),
                left_embedding.as_deref().or(right_embedding.as_deref()),
            )?;
            let left_prior =
                legacy_projection_prior(projection.as_ref(), left_embedding.as_deref());
            let right_prior =
                legacy_projection_prior(projection.as_ref(), right_embedding.as_deref());

            legacy_batter_asset(
                &mut left.alpha,
                &mut left.coords,
                &field.session.mood,
                &left_prior,
                err,
                LEGACY_LR_DUEL_ALPHA,
                LEGACY_LR_DUEL_COORD,
            );
            legacy_batter_asset(
                &mut right.alpha,
                &mut right.coords,
                &field.session.mood,
                &right_prior,
                -err,
                LEGACY_LR_DUEL_ALPHA,
                LEGACY_LR_DUEL_COORD,
            );
            for (mood, gap) in field.session.mood.iter_mut().zip(coord_gap.iter().copied()) {
                *mood += LEGACY_LR_DUEL_MOOD * (err * gap - crate::quality::LEGACY_L2_MOOD * *mood);
            }

            left.compare_count += 1;
            right.compare_count += 1;
            if left_won {
                left.win_count += 1;
            } else {
                right.win_count += 1;
            }
            field.session.comparisons += 1;

            let mut embedding_head = field.embedding_head.clone().or_else(|| {
                left_embedding
                    .as_ref()
                    .or(right_embedding.as_ref())
                    .map(|vector| {
                        SessionEmbeddingHead::zero(
                            self.embedder.model_name().to_owned(),
                            vector.len(),
                        )
                    })
            });
            if let (Some(head), Some(lhs), Some(rhs)) = (
                &mut embedding_head,
                left_embedding.as_deref(),
                right_embedding.as_deref(),
            ) {
                head.contrast_step(lhs, rhs, err, LEGACY_LR_DUEL_HEAD, LEGACY_L2_SESSION_HEAD);
            }
            if let Some(model) = &mut projection {
                if let Some(embedding) = &left_embedding {
                    model.gradient_step(
                        embedding,
                        &left.coords,
                        LEGACY_LR_PROJECTION,
                        LEGACY_PROJECTION_WEIGHT_DECAY,
                    );
                }
                if let Some(embedding) = &right_embedding {
                    model.gradient_step(
                        embedding,
                        &right.coords,
                        LEGACY_LR_PROJECTION,
                        LEGACY_PROJECTION_WEIGHT_DECAY,
                    );
                }
            }

            store.persist_duel_step(
                &field.session,
                &left_before,
                &right_before,
                &left,
                &right,
                winner_id,
                left_utility,
                right_utility,
                projection.as_ref(),
                embedding_head.as_ref(),
            )?;
            field.embedding_head = embedding_head;
            Ok(Some(field))
        })?;
        if let Some(field) = updated {
            self.replace_session_field_cache(field);
        }
        self.redirect_target_for_next_pair(&HashSet::new())
    }

    fn session_with_store(&self, store: &Store) -> anyhow::Result<SessionRecord> {
        store.session(self.active.session_id)
    }

    fn rebuild_session_field(&self, store: &Store) -> anyhow::Result<SessionField> {
        let quality_model = store.active_quality_model()?.formal_version;
        let hierarchical_assets = match quality_model {
            QualityFormalVersion::LegacyIndependentV1 => HashMap::new(),
            QualityFormalVersion::HierarchicalGaussianV1 => store
                .corpus_asset_quality_caches(self.active.corpus_id, quality_model)?
                .into_iter()
                .filter_map(|(asset_id, cache)| {
                    HierarchicalAssetPosterior::decode(&cache.payload)
                        .map(|payload| (asset_id, payload))
                })
                .collect(),
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => HashMap::new(),
        };
        let hierarchical_session = match quality_model {
            QualityFormalVersion::LegacyIndependentV1 => None,
            QualityFormalVersion::HierarchicalGaussianV1 => store
                .session_quality_cache(self.active.session_id, quality_model)?
                .and_then(|cache| HierarchicalSessionPosterior::decode(&cache.payload)),
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => None,
        };
        let perturbative_assets = match quality_model {
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => store
                .corpus_asset_quality_caches(self.active.corpus_id, quality_model)?
                .into_iter()
                .filter_map(|(asset_id, cache)| {
                    PerturbativeAssetPosterior::decode(&cache.payload)
                        .map(|payload| (asset_id, payload))
                })
                .collect(),
            _ => HashMap::new(),
        };
        let perturbative_session = match quality_model {
            QualityFormalVersion::HierarchicalPerturbativeV2
            | QualityFormalVersion::HierarchicalPerturbativeV3 => store
                .session_quality_cache(self.active.session_id, quality_model)?
                .and_then(|cache| PerturbativeSessionPosterior::decode(&cache.payload)),
            _ => None,
        };
        let perturbative_hyper = store
            .load_perturbative_hyper_params(&store.active_quality_model()?)?
            .unwrap_or_default();
        Ok(SessionField {
            session: self.session_with_store(store)?,
            quality_model,
            exact_offsets: store.session_asset_offsets(self.active.session_id)?,
            hearted_assets: store.hearted_assets()?,
            subsource_lock: store.session_subsource_lock(self.active.session_id)?,
            embeddings: Arc::new(
                store.corpus_embeddings(self.active.corpus_id, self.embedder.model_name())?,
            ),
            embedding_head: store
                .session_embedding_head(self.active.session_id, self.embedder.model_name())?,
            hierarchical_assets: Arc::new(hierarchical_assets),
            dominant_faces: Arc::new(store.dominant_local_face_identities(self.active.corpus_id)?),
            hierarchical_session,
            perturbative_assets: Arc::new(perturbative_assets),
            perturbative_session,
            perturbative_hyper,
        })
    }

    fn choose_next_pair(
        &self,
        lock_exhaustion: LockExhaustionPolicy,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        for _ in 0..2 {
            let store = self.read_store()?;
            let field = self.session_field(&store)?;
            let assets = visible_assets(&store, self.active.corpus_id)?;
            if field.subsource_lock.is_some() {
                let pair = self.choose_pair_in_locked_subsource_with_store(
                    &store,
                    &field,
                    &assets,
                    None,
                    excluded_visual_keys,
                )?;
                if pair.is_some() {
                    return Ok(pair);
                }
                if lock_exhaustion == LockExhaustionPolicy::PreserveAndStop {
                    return Ok(None);
                }
                self.clear_external_subsource_lock()?;
                continue;
            }
            return self.choose_pair_with_store(&store, &field, &assets, excluded_visual_keys);
        }
        Ok(None)
    }

    fn choose_next_pair_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
        lock_exhaustion: LockExhaustionPolicy,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaPair>> {
        for _ in 0..2 {
            let store = self.read_store()?;
            let field = self.session_field(&store)?;
            if field.subsource_lock.is_some() {
                let assets = visible_assets(&store, self.active.corpus_id)?;
                let pair = self.choose_pair_in_locked_subsource_with_store(
                    &store,
                    &field,
                    &assets,
                    local_anchor,
                    excluded_visual_keys,
                )?;
                if pair.is_some() {
                    return Ok(pair);
                }
                if lock_exhaustion == LockExhaustionPolicy::PreserveAndStop {
                    return Ok(None);
                }
                self.clear_external_subsource_lock()?;
                continue;
            }
            return self.choose_pair_preserving_local_anchor_with_store(
                &store,
                &field,
                local_anchor,
                excluded_visual_keys,
            );
        }
        Ok(None)
    }

    fn redirect_target_for_next_pair(
        &self,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<RedirectTarget> {
        let pair =
            self.choose_next_pair(LockExhaustionPolicy::ClearAndRetry, excluded_visual_keys)?;
        if let Some(pair_ref) = pair.as_ref() {
            self.note_remote_pair_selected(pair_ref)?;
        }
        redirect_target_for_pair(pair)
    }

    pub(super) fn redirect_target_preserving_local_anchor(
        &self,
        local_anchor: Option<&AssetId>,
    ) -> anyhow::Result<RedirectTarget> {
        let pair = self.choose_next_pair_preserving_local_anchor(
            local_anchor,
            LockExhaustionPolicy::ClearAndRetry,
            &HashSet::new(),
        )?;
        if let Some(pair_ref) = pair.as_ref() {
            self.note_remote_pair_selected(pair_ref)?;
        }
        redirect_target_for_pair(pair)
    }

    pub(crate) fn arena_prefetch_target_excluding(
        &self,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<RedirectTarget> {
        redirect_target_for_pair(
            self.choose_next_pair(LockExhaustionPolicy::PreserveAndStop, excluded_visual_keys)?,
        )
    }

    pub(crate) fn arena_prefetch_target_preserving_local_anchor_excluding(
        &self,
        local_anchor: Option<&AssetId>,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<RedirectTarget> {
        redirect_target_for_pair(self.choose_next_pair_preserving_local_anchor(
            local_anchor,
            LockExhaustionPolicy::PreserveAndStop,
            excluded_visual_keys,
        )?)
    }

    pub(super) fn session_field(&self, store: &Store) -> anyhow::Result<SessionField> {
        if let Some(field) = self.session_field_cache.read().clone() {
            return Ok(field);
        }
        let field = self.rebuild_session_field(store)?;
        self.replace_session_field_cache(field.clone());
        Ok(field)
    }
}
