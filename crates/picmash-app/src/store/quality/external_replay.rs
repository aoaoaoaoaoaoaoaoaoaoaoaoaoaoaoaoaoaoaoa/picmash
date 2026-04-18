use super::*;

impl Store {
    pub(super) fn apply_hierarchical_replay_external(
        &self,
        state: &mut HierarchicalReplayState,
        event: HierarchicalReplayExternalEvent,
    ) -> anyhow::Result<()> {
        match event.kind {
            ExternalEventKind::LocalWin | ExternalEventKind::RemoteWin => {
                self.apply_hierarchical_replay_external_duel(state, event)
            }
            ExternalEventKind::Rejected | ExternalEventKind::Kept | ExternalEventKind::Hearted => {
                self.apply_hierarchical_replay_external_unary(state, event)
            }
            ExternalEventKind::Selected
            | ExternalEventKind::StreamBlocked
            | ExternalEventKind::Imported => Ok(()),
        }
    }

    fn apply_hierarchical_replay_external_duel(
        &self,
        state: &mut HierarchicalReplayState,
        event: HierarchicalReplayExternalEvent,
    ) -> anyhow::Result<()> {
        let Some(local_asset_id) = event.local_asset_id.as_ref() else {
            return Ok(());
        };
        let Some(session) = state.sessions.get_mut(&event.session_id) else {
            return Ok(());
        };
        let Some(mut local) = state.assets.remove(local_asset_id) else {
            return Ok(());
        };
        let Some(mut remote) = state.external_items.remove(&event.item_id) else {
            state.assets.insert(local.asset.id.clone(), local);
            return Ok(());
        };
        refresh_asset_face_anchor(&mut local, &state.subjects);
        let outcome = if matches!(event.kind, ExternalEventKind::LocalWin) {
            1.0
        } else {
            -1.0
        };
        let local_mean = hierarchical_session_utility_mean(&local, session, local_asset_id);
        let remote_mean = hierarchical_external_session_utility_mean(&remote, session);
        let delta_mean = local_mean - remote_mean;
        let delta_variance = hierarchical_session_utility_variance(&local, session)
            + hierarchical_external_session_utility_variance(&remote, session);
        let Some(moments) =
            gaussian_duel_moment_match(delta_mean, delta_variance, outcome, HIERARCHICAL_DUEL_BETA)
        else {
            state.assets.insert(local.asset.id.clone(), local);
            state.external_items.insert(event.item_id, remote);
            return Ok(());
        };

        diagonal_adf_update(
            &mut local.baseline_mean,
            &mut local.baseline_variance,
            1.0,
            outcome,
            moments,
        );
        diagonal_adf_update(
            &mut remote.baseline_mean,
            &mut remote.baseline_variance,
            -1.0,
            outcome,
            moments,
        );
        for axis in 0..crate::model::LATENT_DIM {
            diagonal_adf_update(
                &mut local.mood_loading_mean[axis],
                &mut local.mood_loading_variance[axis],
                session.session.mood[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut remote.mood_loading_mean[axis],
                &mut remote.mood_loading_variance[axis],
                -session.session.mood[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut session.session.mood[axis],
                &mut session.semantic_mood_variance[axis],
                local.mood_loading_mean[axis] - remote.mood_loading_mean[axis],
                outcome,
                moments,
            );
        }
        if let (Some(mean), Some(variance)) =
            (&mut local.technical_mean, &mut local.technical_variance)
        {
            diagonal_adf_update(mean, variance, HIERARCHICAL_TECH_WEIGHT, outcome, moments);
        }
        if let (Some(mean), Some(variance)) =
            (&mut remote.technical_mean, &mut remote.technical_variance)
        {
            diagonal_adf_update(mean, variance, -HIERARCHICAL_TECH_WEIGHT, outcome, moments);
        }
        for axis in 0..VIBE_DESCRIPTOR_DIM {
            diagonal_adf_update(
                &mut local.vibe_mean[axis],
                &mut local.vibe_variance[axis],
                session.vibe_mean[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut remote.vibe_mean[axis],
                &mut remote.vibe_variance[axis],
                -session.vibe_mean[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut session.vibe_mean[axis],
                &mut session.vibe_variance[axis],
                local.vibe_mean[axis] - remote.vibe_mean[axis],
                outcome,
                moments,
            );
        }

        local.asset.compare_count += 1;
        if matches!(event.kind, ExternalEventKind::LocalWin) {
            local.asset.win_count += 1;
        }
        sync_hierarchical_asset_record(&mut local);
        session.session.comparisons += 1;
        state.assets.insert(local.asset.id.clone(), local);
        state.external_items.insert(event.item_id, remote);
        Ok(())
    }

    fn apply_hierarchical_replay_external_unary(
        &self,
        state: &mut HierarchicalReplayState,
        event: HierarchicalReplayExternalEvent,
    ) -> anyhow::Result<()> {
        let feedback = match event.kind {
            ExternalEventKind::Rejected => ExternalUnaryFeedback::Reject,
            ExternalEventKind::Kept => ExternalUnaryFeedback::Accept,
            ExternalEventKind::Hearted => ExternalUnaryFeedback::Heart,
            other => bail!("unsupported external unary event {}", other.as_str()),
        };
        let Some(session) = state.sessions.get_mut(&event.session_id) else {
            return Ok(());
        };
        let Some(mut remote) = state.external_items.remove(&event.item_id) else {
            return Ok(());
        };
        let utility_mean = hierarchical_external_session_utility_mean(&remote, session);
        let utility_variance = hierarchical_external_session_utility_variance(&remote, session);
        let Some(moments) = gaussian_duel_moment_match(
            utility_mean - session.session.frontier,
            utility_variance + session.frontier_variance,
            feedback.outcome(),
            feedback.beta(),
        ) else {
            state.external_items.insert(event.item_id, remote);
            return Ok(());
        };
        diagonal_adf_update(
            &mut remote.baseline_mean,
            &mut remote.baseline_variance,
            1.0,
            feedback.outcome(),
            moments,
        );
        for axis in 0..crate::model::LATENT_DIM {
            diagonal_adf_update(
                &mut remote.mood_loading_mean[axis],
                &mut remote.mood_loading_variance[axis],
                session.session.mood[axis],
                feedback.outcome(),
                moments,
            );
            diagonal_adf_update(
                &mut session.session.mood[axis],
                &mut session.semantic_mood_variance[axis],
                remote.mood_loading_mean[axis],
                feedback.outcome(),
                moments,
            );
        }
        if let (Some(mean), Some(variance)) =
            (&mut remote.technical_mean, &mut remote.technical_variance)
        {
            diagonal_adf_update(
                mean,
                variance,
                HIERARCHICAL_TECH_WEIGHT,
                feedback.outcome(),
                moments,
            );
        }
        for axis in 0..VIBE_DESCRIPTOR_DIM {
            diagonal_adf_update(
                &mut remote.vibe_mean[axis],
                &mut remote.vibe_variance[axis],
                session.vibe_mean[axis],
                feedback.outcome(),
                moments,
            );
            diagonal_adf_update(
                &mut session.vibe_mean[axis],
                &mut session.vibe_variance[axis],
                remote.vibe_mean[axis],
                feedback.outcome(),
                moments,
            );
        }
        diagonal_adf_update(
            &mut session.session.frontier,
            &mut session.frontier_variance,
            -1.0,
            feedback.outcome(),
            moments,
        );
        if matches!(feedback, ExternalUnaryFeedback::Heart) {
            session.session.hearts = session.session.hearts.saturating_add(1);
        } else {
            session.session.nudges = session.session.nudges.saturating_add(1);
        }
        state.external_items.insert(event.item_id, remote);
        Ok(())
    }

    pub(super) fn persist_legacy_replay_state(
        &mut self,
        state: &LegacyReplayState,
        subject_snapshot: &[(FaceIdentityId, FaceBeauty, u32)],
    ) -> anyhow::Result<()> {
        let updated_at = now_ts();
        let tx = self
            .conn
            .transaction()
            .context("opening legacy quality replay transaction")?;

        tx.execute(
            "DELETE FROM quality_asset_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::LegacyIndependentV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_session_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::LegacyIndependentV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_subject_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::LegacyIndependentV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_replay_cursors WHERE formal_version = ?1",
            params![QualityFormalVersion::LegacyIndependentV1.as_str()],
        )?;
        tx.execute("DELETE FROM session_asset_offsets", [])?;
        tx.execute("DELETE FROM session_asset_hearts", [])?;
        tx.execute(
            "DELETE FROM session_embedding_heads WHERE model_name = ?1",
            params![state.projection_model_name],
        )?;
        tx.execute(
            "DELETE FROM projection_models WHERE model_name = ?1",
            params![state.projection_model_name],
        )?;

        let mut update_asset_stmt = tx.prepare(
            r"
            UPDATE assets
            SET alpha = ?2,
                c0 = ?3,
                c1 = ?4,
                c2 = ?5,
                heart_count = ?6,
                hearted = ?7,
                compare_count = ?8,
                win_count = ?9
            WHERE id = ?1
            ",
        )?;
        let mut update_session_stmt = tx.prepare(
            r"
            UPDATE sessions
            SET z0 = ?2,
                z1 = ?3,
                z2 = ?4,
                frontier = ?5,
                comparisons = ?6,
                nudges = ?7,
                hearts = ?8
            WHERE id = ?1
            ",
        )?;
        let mut insert_asset_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_asset_cache (
                formal_version,
                asset_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_session_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_session_cache (
                formal_version,
                session_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_subject_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_subject_cache (
                formal_version,
                identity_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut update_subject_stmt = tx.prepare(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = ?4
            WHERE id = ?1
            ",
        )?;

        for asset in state.assets.values() {
            update_asset_stmt.execute(params![
                asset.id.0,
                asset.alpha,
                asset.coords[0],
                asset.coords[1],
                asset.coords[2],
                asset.heart_count,
                i64::from(asset.is_hearted),
                asset.compare_count,
                asset.win_count,
            ])?;
            insert_asset_cache_stmt.execute(params![
                QualityFormalVersion::LegacyIndependentV1.as_str(),
                asset.id.0,
                encode_quality_payload(&legacy_asset_quality_payload(asset))?,
                updated_at,
            ])?;
        }

        for session in state.sessions.values() {
            update_session_stmt.execute(params![
                session.session.id.0,
                session.session.mood[0],
                session.session.mood[1],
                session.session.mood[2],
                session.session.frontier,
                session.session.comparisons,
                session.session.nudges,
                session.session.hearts,
            ])?;
            insert_session_cache_stmt.execute(params![
                QualityFormalVersion::LegacyIndependentV1.as_str(),
                session.session.id.0,
                encode_quality_payload(&legacy_session_quality_payload(&session.session))?,
                updated_at,
            ])?;

            for (asset_id, offset) in &session.exact_offsets {
                write_session_offset(&tx, session.session.id, asset_id, *offset)?;
            }
            for asset_id in &session.hearted_assets {
                enshrine_session_heart(&tx, session.session.id, asset_id)?;
            }
            upsert_session_embedding_head(
                &tx,
                session.session.id,
                session.embedding_head.as_ref(),
            )?;
        }

        upsert_projection(&tx, state.projection.as_ref())?;

        for &(identity_id, beauty, duel_count) in subject_snapshot {
            update_subject_stmt.execute(params![
                identity_id.0,
                beauty.mean,
                beauty.sigma,
                duel_count
            ])?;
            insert_subject_cache_stmt.execute(params![
                QualityFormalVersion::LegacyIndependentV1.as_str(),
                identity_id.0,
                encode_quality_payload(&legacy_subject_quality_payload(beauty, duel_count))?,
                updated_at,
            ])?;
        }

        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::LegacyIndependentV1,
            "comparison_max_id",
            state.max_comparison_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::LegacyIndependentV1,
            "nudge_max_id",
            state.max_nudge_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::LegacyIndependentV1,
            "heart_max_id",
            state.max_heart_id,
            updated_at,
        )?;

        drop(update_asset_stmt);
        drop(update_session_stmt);
        drop(insert_asset_cache_stmt);
        drop(insert_session_cache_stmt);
        drop(insert_subject_cache_stmt);
        drop(update_subject_stmt);
        tx.commit()
            .context("committing legacy quality replay transaction")?;
        Ok(())
    }

    pub(super) fn persist_hierarchical_replay_state(
        &mut self,
        state: &HierarchicalReplayState,
        subject_snapshot: &[(FaceIdentityId, FaceBeauty, u32)],
        model: &QualityModelRecord,
        technical_head: &LinearTechnicalPriorHead,
    ) -> anyhow::Result<()> {
        let updated_at = now_ts();
        let tx = self
            .conn
            .transaction()
            .context("opening hierarchical quality replay transaction")?;
        tx.execute(
            "DELETE FROM quality_asset_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_session_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_subject_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_external_item_cache WHERE formal_version = ?1",
            params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_replay_cursors WHERE formal_version = ?1",
            params![QualityFormalVersion::HierarchicalGaussianV1.as_str()],
        )?;
        tx.execute("DELETE FROM session_asset_offsets", [])?;
        tx.execute("DELETE FROM session_asset_hearts", [])?;
        tx.execute("DELETE FROM session_embedding_heads", [])?;
        tx.execute("DELETE FROM projection_models", [])?;

        let mut update_asset_stmt = tx.prepare(
            r"
            UPDATE assets
            SET alpha = ?2,
                c0 = ?3,
                c1 = ?4,
                c2 = ?5,
                heart_count = ?6,
                hearted = ?7,
                compare_count = ?8,
                win_count = ?9
            WHERE id = ?1
            ",
        )?;
        let mut update_session_stmt = tx.prepare(
            r"
            UPDATE sessions
            SET z0 = ?2,
                z1 = ?3,
                z2 = ?4,
                frontier = ?5,
                comparisons = ?6,
                nudges = ?7,
                hearts = ?8
            WHERE id = ?1
            ",
        )?;
        let mut insert_asset_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_asset_cache (
                formal_version,
                asset_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_session_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_session_cache (
                formal_version,
                session_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_subject_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_subject_cache (
                formal_version,
                identity_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_external_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_external_item_cache (
                formal_version,
                item_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut update_subject_stmt = tx.prepare(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = ?4
            WHERE id = ?1
            ",
        )?;

        for state_asset in state.assets.values() {
            update_asset_stmt.execute(params![
                state_asset.asset.id.0,
                state_asset.asset.alpha,
                state_asset.asset.coords[0],
                state_asset.asset.coords[1],
                state_asset.asset.coords[2],
                state_asset.asset.heart_count,
                i64::from(state_asset.asset.is_hearted),
                state_asset.asset.compare_count,
                state_asset.asset.win_count,
            ])?;
            insert_asset_cache_stmt.execute(params![
                QualityFormalVersion::HierarchicalGaussianV1.as_str(),
                state_asset.asset.id.0,
                encode_quality_payload(&AssetQualityCachePayload::HierarchicalGaussianV1(
                    HierarchicalAssetQualityCacheV1 {
                        baseline_mean: state_asset.baseline_mean,
                        baseline_variance: state_asset.baseline_variance,
                        canonical_mean: hierarchical_asset_canonical_mean(state_asset),
                        canonical_variance: hierarchical_asset_canonical_variance(state_asset),
                        mood_loading_mean: state_asset.mood_loading_mean.to_vec(),
                        mood_loading_variance: state_asset.mood_loading_variance.to_vec(),
                        technical_mean: state_asset.technical_mean,
                        technical_variance: state_asset.technical_variance,
                        vibe_mean: state_asset.vibe_mean.to_vec(),
                        vibe_variance: state_asset.vibe_variance.to_vec(),
                    }
                ))?,
                updated_at,
            ])?;
        }

        for session in state.sessions.values() {
            update_session_stmt.execute(params![
                session.session.id.0,
                session.session.mood[0],
                session.session.mood[1],
                session.session.mood[2],
                session.session.frontier,
                session.session.comparisons,
                session.session.nudges,
                session.session.hearts,
            ])?;
            insert_session_cache_stmt.execute(params![
                QualityFormalVersion::HierarchicalGaussianV1.as_str(),
                session.session.id.0,
                encode_quality_payload(&SessionQualityCachePayload::HierarchicalGaussianV1(
                    HierarchicalSessionQualityCacheV1 {
                        semantic_mood_mean: session.session.mood.to_vec(),
                        semantic_mood_variance: session.semantic_mood_variance.to_vec(),
                        vibe_mean: session.vibe_mean.to_vec(),
                        vibe_variance: session.vibe_variance.to_vec(),
                        frontier_mean: session.session.frontier,
                        frontier_variance: session.frontier_variance,
                    }
                ))?,
                updated_at,
            ])?;
            for (asset_id, offset) in &session.exact_offsets {
                write_session_offset(&tx, session.session.id, asset_id, *offset)?;
            }
            for asset_id in &session.hearted_assets {
                enshrine_session_heart(&tx, session.session.id, asset_id)?;
            }
            upsert_session_embedding_head(
                &tx,
                session.session.id,
                session.embedding_head.as_ref(),
            )?;
        }

        for &(identity_id, beauty, duel_count) in subject_snapshot {
            update_subject_stmt.execute(params![
                identity_id.0,
                beauty.mean,
                beauty.sigma,
                duel_count
            ])?;
            insert_subject_cache_stmt.execute(params![
                QualityFormalVersion::HierarchicalGaussianV1.as_str(),
                identity_id.0,
                encode_quality_payload(&SubjectQualityCachePayload::HierarchicalGaussianV1(
                    HierarchicalSubjectQualityCacheV1 {
                        beauty_mean: beauty.mean,
                        beauty_variance: beauty.sigma * beauty.sigma,
                        duel_count,
                    }
                ))?,
                updated_at,
            ])?;
        }

        for (&item_id, state_item) in &state.external_items {
            insert_external_cache_stmt.execute(params![
                QualityFormalVersion::HierarchicalGaussianV1.as_str(),
                item_id.0,
                encode_quality_payload(&AssetQualityCachePayload::HierarchicalGaussianV1(
                    HierarchicalAssetQualityCacheV1 {
                        baseline_mean: state_item.baseline_mean,
                        baseline_variance: state_item.baseline_variance,
                        canonical_mean: hierarchical_external_canonical_mean(state_item),
                        canonical_variance: hierarchical_external_canonical_variance(state_item),
                        mood_loading_mean: state_item.mood_loading_mean.to_vec(),
                        mood_loading_variance: state_item.mood_loading_variance.to_vec(),
                        technical_mean: state_item.technical_mean,
                        technical_variance: state_item.technical_variance,
                        vibe_mean: state_item.vibe_mean.to_vec(),
                        vibe_variance: state_item.vibe_variance.to_vec(),
                    }
                ))?,
                updated_at,
            ])?;
        }

        tx.execute(
            r"
            INSERT INTO quality_prior_artifacts (
                formal_version,
                prior_family,
                prior_revision,
                artifact_key,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(formal_version, prior_family, prior_revision, artifact_key) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            ",
            params![
                model.formal_version.as_str(),
                model.prior_family.as_str(),
                model.prior_revision.as_str(),
                technical_prior_artifact_key(),
                encode_quality_payload(technical_head)?,
                updated_at,
            ],
        )?;

        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::HierarchicalGaussianV1,
            "comparison_max_id",
            state.max_comparison_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::HierarchicalGaussianV1,
            "nudge_max_id",
            state.max_nudge_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::HierarchicalGaussianV1,
            "heart_max_id",
            state.max_heart_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            QualityFormalVersion::HierarchicalGaussianV1,
            "external_max_id",
            state.max_external_id,
            updated_at,
        )?;

        drop(update_asset_stmt);
        drop(update_session_stmt);
        drop(insert_asset_cache_stmt);
        drop(insert_session_cache_stmt);
        drop(insert_subject_cache_stmt);
        drop(insert_external_cache_stmt);
        drop(update_subject_stmt);
        tx.commit()
            .context("committing hierarchical quality replay transaction")?;
        Ok(())
    }

    pub(super) fn global_hierarchical_domain_gates(
        &self,
        model_name: &str,
        embeddings: &HashMap<AssetId, Vec<f32>>,
    ) -> anyhow::Result<HashMap<AssetId, bool>> {
        let labels = self
            .conn
            .prepare(
                r"
            SELECT ad.asset_id, ad.label, e.vector
            FROM asset_domain_labels ad
            JOIN embeddings e
              ON e.asset_id = ad.asset_id
             AND e.model_name = ?1
            ",
            )?
            .query_map(params![model_name], |row| {
                let asset_id = AssetId(row.get::<_, String>(0)?);
                let label = row
                    .get::<_, String>(1)?
                    .parse::<AssetDomainLabel>()
                    .map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::other(error.to_string())),
                        )
                    })?;
                Ok(crate::asset_domain::AssetDomainTrainingRow {
                    asset_id,
                    label,
                    embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(2)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let oracle = AssetDomainOracle::train(&labels);
        let manual = labels
            .iter()
            .map(|row| (row.asset_id.clone(), row.label))
            .collect::<HashMap<_, _>>();
        Ok(embeddings
            .iter()
            .map(|(asset_id, embedding)| {
                let gate = manual
                    .get(asset_id)
                    .copied()
                    .or_else(|| {
                        oracle
                            .predict(embedding)
                            .map(|prediction| prediction.label())
                    })
                    .is_some_and(|label| matches!(label, AssetDomainLabel::Real));
                (asset_id.clone(), gate)
            })
            .collect())
    }

    pub(super) fn global_hierarchical_external_domain_gates(
        &self,
        model_name: &str,
        embeddings: &HashMap<RemoteItemId, Vec<f32>>,
    ) -> anyhow::Result<HashMap<RemoteItemId, bool>> {
        let labels = self
            .conn
            .prepare(
                r"
            SELECT ad.asset_id, ad.label, e.vector
            FROM asset_domain_labels ad
            JOIN embeddings e
              ON e.asset_id = ad.asset_id
             AND e.model_name = ?1
            ",
            )?
            .query_map(params![model_name], |row| {
                let asset_id = AssetId(row.get::<_, String>(0)?);
                let label = row
                    .get::<_, String>(1)?
                    .parse::<AssetDomainLabel>()
                    .map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::other(error.to_string())),
                        )
                    })?;
                Ok(crate::asset_domain::AssetDomainTrainingRow {
                    asset_id,
                    label,
                    embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(2)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let oracle = AssetDomainOracle::train(&labels);
        Ok(embeddings
            .iter()
            .map(|(item_id, embedding)| {
                let gate = oracle
                    .predict(embedding)
                    .map(|prediction| prediction.label())
                    .is_some_and(|label| matches!(label, AssetDomainLabel::Real));
                (*item_id, gate)
            })
            .collect())
    }

    pub(super) fn technical_prior_samples_from_state(
        &self,
        state: &HierarchicalReplayState,
    ) -> Vec<TechnicalPriorSample> {
        let mut samples = Vec::new();
        for (asset_id, asset) in &state.assets {
            if let (Some(mean), Some(variance)) = (asset.technical_mean, asset.technical_variance)
                && let Ok(Some(features)) =
                    self.asset_quality_features(asset_id, QUALITY_FEATURE_REVISION)
            {
                samples.push(TechnicalPriorSample {
                    descriptor: features.features.technical,
                    target_mean: mean,
                    target_variance: variance,
                });
            }
        }
        for (item_id, item) in &state.external_items {
            if let (Some(mean), Some(variance)) = (item.technical_mean, item.technical_variance)
                && let Ok(Some(features)) =
                    self.external_item_quality_features(*item_id, QUALITY_FEATURE_REVISION)
            {
                samples.push(TechnicalPriorSample {
                    descriptor: features.technical,
                    target_mean: mean,
                    target_variance: variance,
                });
            }
        }
        samples
    }

    pub(super) fn dominant_face_beauty_by_asset(
        &self,
    ) -> anyhow::Result<HashMap<AssetId, SubjectBeautyAnchor>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT f.asset_id, f.identity_id, fi.rating_mu, fi.rating_sigma, (f.bbox_w * f.bbox_h) AS face_area
            FROM faces f
            JOIN face_identities fi ON fi.id = f.identity_id
            WHERE f.asset_id IS NOT NULL
              AND f.hidden = 0
            ORDER BY f.asset_id ASC, face_area DESC, f.id ASC
            ",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    AssetId(row.get::<_, String>(0)?),
                    SubjectBeautyAnchor {
                        identity_id: FaceIdentityId(row.get::<_, i64>(1)?),
                        mean: row.get::<_, f32>(2)?,
                        variance: row.get::<_, f32>(3)?.powi(2).max(HIERARCHICAL_MIN_VARIANCE),
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = HashMap::new();
        for (asset_id, beauty) in rows {
            out.entry(asset_id).or_insert(beauty);
        }
        Ok(out)
    }

    pub(super) fn load_active_quality_model(&self) -> anyhow::Result<Option<QualityModelRecord>> {
        self.conn
            .query_row(
                r"
                SELECT formal_version, prior_family, prior_revision, updated_at
                FROM quality_model_registry
                WHERE slot = ?1
                ",
                params![ACTIVE_QUALITY_MODEL_SLOT],
                |row| {
                    Ok(QualityModelRecord {
                        formal_version: row.get::<_, String>(0)?.parse().map_err(into_rusqlite)?,
                        prior_family: crate::quality::QualityPriorFamily::forge(
                            row.get::<_, String>(1)?,
                        )
                        .map_err(into_rusqlite)?,
                        prior_revision: crate::quality::QualityPriorRevision::forge(
                            row.get::<_, String>(2)?,
                        )
                        .map_err(into_rusqlite)?,
                        updated_at: decode_ts(row.get::<_, i64>(3)?).map_err(into_rusqlite)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }
}

pub(super) fn replay_session_utility(
    asset: &AssetRecord,
    session: &LegacyReplaySessionState,
    embedding: Option<&[f32]>,
    asset_id: &AssetId,
) -> f32 {
    let residual = match (&session.embedding_head, embedding) {
        (Some(head), Some(embedding)) => head.score(embedding),
        _ => 0.0,
    };
    let exact_offset = session
        .exact_offsets
        .get(asset_id)
        .copied()
        .unwrap_or_default();
    session_utility(
        asset,
        &session.session,
        residual,
        exact_offset,
        legacy_heart_bias(asset, asset.is_hearted),
    )
}

pub(super) fn face_anchor_for_subject(
    face: SubjectBeautyAnchor,
    subjects: &HashMap<FaceIdentityId, HierarchicalSubjectState>,
) -> SubjectBeautyAnchor {
    subjects
        .get(&face.identity_id)
        .map(|subject| SubjectBeautyAnchor {
            identity_id: face.identity_id,
            mean: subject.beauty.mean,
            variance: subject.beauty.sigma.powi(2).max(HIERARCHICAL_MIN_VARIANCE),
        })
        .unwrap_or(face)
}

pub(super) fn refresh_asset_face_anchor(
    asset: &mut HierarchicalReplayAssetState,
    subjects: &HashMap<FaceIdentityId, HierarchicalSubjectState>,
) {
    asset.face = asset
        .face
        .map(|face| face_anchor_for_subject(face, subjects));
}

fn hierarchical_asset_canonical_mean(asset: &HierarchicalReplayAssetState) -> f32 {
    asset.baseline_mean
        + asset.face.map_or(0.0, |face| {
            HIERARCHICAL_FACE_WEIGHT
                * hierarchical_face_latent_mean(FaceBeauty::forge(
                    face.mean,
                    face.variance.max(HIERARCHICAL_MIN_VARIANCE).sqrt(),
                ))
        })
        + asset
            .technical_mean
            .map_or(0.0, |technical| HIERARCHICAL_TECH_WEIGHT * technical)
}

fn hierarchical_asset_canonical_variance(asset: &HierarchicalReplayAssetState) -> f32 {
    asset.baseline_variance
        + asset.face.map_or(0.0, |face| {
            HIERARCHICAL_FACE_WEIGHT.powi(2)
                * hierarchical_face_latent_variance(FaceBeauty::forge(
                    face.mean,
                    face.variance.max(HIERARCHICAL_MIN_VARIANCE).sqrt(),
                ))
        })
        + asset
            .technical_variance
            .map_or(0.0, |variance| HIERARCHICAL_TECH_WEIGHT.powi(2) * variance)
}

fn hierarchical_external_canonical_mean(item: &HierarchicalReplayExternalState) -> f32 {
    item.baseline_mean
        + item
            .technical_mean
            .map_or(0.0, |technical| HIERARCHICAL_TECH_WEIGHT * technical)
}

fn hierarchical_external_canonical_variance(item: &HierarchicalReplayExternalState) -> f32 {
    item.baseline_variance
        + item
            .technical_variance
            .map_or(0.0, |variance| HIERARCHICAL_TECH_WEIGHT.powi(2) * variance)
}

pub(super) fn hierarchical_session_utility_mean(
    asset: &HierarchicalReplayAssetState,
    session: &HierarchicalReplaySessionState,
    asset_id: &AssetId,
) -> f32 {
    let exact_offset = session
        .exact_offsets
        .get(asset_id)
        .copied()
        .unwrap_or_default();
    hierarchical_asset_canonical_mean(asset)
        + crate::model::dot(&asset.mood_loading_mean, &session.session.mood)
        + asset
            .technical_mean
            .map(|_| {
                asset
                    .vibe_mean
                    .iter()
                    .zip(session.vibe_mean.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>()
            })
            .unwrap_or_default()
        + exact_offset
        + legacy_heart_bias(&asset.asset, asset.asset.is_hearted)
}

pub(super) fn hierarchical_session_utility_variance(
    asset: &HierarchicalReplayAssetState,
    session: &HierarchicalReplaySessionState,
) -> f32 {
    let semantic = asset
        .mood_loading_mean
        .iter()
        .zip(asset.mood_loading_variance.iter())
        .zip(
            session
                .session
                .mood
                .iter()
                .zip(session.semantic_mood_variance.iter()),
        )
        .map(|((loading_mean, loading_var), (mood_mean, mood_var))| {
            mood_mean.powi(2) * *loading_var
                + loading_mean.powi(2) * *mood_var
                + loading_var * mood_var
        })
        .sum::<f32>();
    let vibe = asset
        .technical_mean
        .map(|_| {
            asset
                .vibe_mean
                .iter()
                .zip(asset.vibe_variance.iter())
                .zip(session.vibe_mean.iter().zip(session.vibe_variance.iter()))
                .map(|((asset_mean, asset_var), (session_mean, session_var))| {
                    session_mean.powi(2) * *asset_var
                        + asset_mean.powi(2) * *session_var
                        + asset_var * session_var
                })
                .sum::<f32>()
        })
        .unwrap_or_default();
    hierarchical_asset_canonical_variance(asset) + semantic + vibe
}

fn hierarchical_external_session_utility_mean(
    item: &HierarchicalReplayExternalState,
    session: &HierarchicalReplaySessionState,
) -> f32 {
    hierarchical_external_canonical_mean(item)
        + crate::model::dot(&item.mood_loading_mean, &session.session.mood)
        + item
            .technical_mean
            .map(|_| {
                item.vibe_mean
                    .iter()
                    .zip(session.vibe_mean.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>()
            })
            .unwrap_or_default()
}

fn hierarchical_external_session_utility_variance(
    item: &HierarchicalReplayExternalState,
    session: &HierarchicalReplaySessionState,
) -> f32 {
    let semantic = item
        .mood_loading_mean
        .iter()
        .zip(item.mood_loading_variance.iter())
        .zip(
            session
                .session
                .mood
                .iter()
                .zip(session.semantic_mood_variance.iter()),
        )
        .map(|((loading_mean, loading_var), (mood_mean, mood_var))| {
            mood_mean.powi(2) * *loading_var
                + loading_mean.powi(2) * *mood_var
                + loading_var * mood_var
        })
        .sum::<f32>();
    let vibe = item
        .technical_mean
        .map(|_| {
            item.vibe_mean
                .iter()
                .zip(item.vibe_variance.iter())
                .zip(session.vibe_mean.iter().zip(session.vibe_variance.iter()))
                .map(|((item_mean, item_var), (session_mean, session_var))| {
                    session_mean.powi(2) * *item_var
                        + item_mean.powi(2) * *session_var
                        + item_var * session_var
                })
                .sum::<f32>()
        })
        .unwrap_or_default();
    hierarchical_external_canonical_variance(item) + semantic + vibe
}

pub(super) fn sync_hierarchical_asset_record(asset: &mut HierarchicalReplayAssetState) {
    asset.asset.alpha = hierarchical_asset_canonical_mean(asset);
    asset.asset.coords = asset.mood_loading_mean;
}

pub(super) fn ensure_projection_dim(state: &mut LegacyReplayState, dim: Option<usize>) {
    if state.projection.is_none()
        && let Some(dim) = dim
    {
        state.projection = Some(ProjectionModel::zero(
            state.projection_model_name.clone(),
            dim,
        ));
    }
}
