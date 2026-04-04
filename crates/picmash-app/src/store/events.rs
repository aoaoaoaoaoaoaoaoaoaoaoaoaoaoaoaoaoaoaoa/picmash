use super::*;

impl Store {
    pub fn rotate_asset(&self, asset_id: &AssetId, direction: i32) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE assets
            SET rotation_quarters = ((rotation_quarters + ?2) % 4 + 4) % 4
            WHERE id = ?1
            ",
            params![asset_id.0, direction],
        )?;
        Ok(())
    }

    pub fn set_hidden(
        &self,
        corpus_id: CorpusId,
        asset_id: &AssetId,
        hidden: bool,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            UPDATE corpus_assets
            SET hidden = ?3
            WHERE corpus_id = ?1 AND asset_id = ?2
            ",
            params![corpus_id.0, asset_id.0, i64::from(hidden)],
        )?;
        Ok(())
    }

    pub fn persist_duel_step(
        &mut self,
        session: &SessionRecord,
        left_before: &AssetRecord,
        right_before: &AssetRecord,
        left_after: &AssetRecord,
        right_after: &AssetRecord,
        winner: &AssetId,
        left_utility: f32,
        right_utility: f32,
        projection: Option<&ProjectionModel>,
        embedding_head: Option<&SessionEmbeddingHead>,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening comparison transaction")?;

        update_asset(&tx, left_after)?;
        update_asset(&tx, right_after)?;
        update_session(&tx, session)?;
        tx.execute(
            r"
            INSERT INTO comparisons (
                session_id,
                corpus_id,
                left_asset_id,
                right_asset_id,
                winner_asset_id,
                created_at,
                left_utility,
                right_utility
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                left_before.id.0,
                right_before.id.0,
                winner.0,
                now_ts(),
                left_utility,
                right_utility,
            ],
        )?;
        upsert_projection(&tx, projection)?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit().context("committing comparison transaction")?;
        self.mirror_legacy_quality_cache(session, [left_after, right_after]);
        Ok(())
    }

    pub fn persist_hierarchical_duel_step(
        &mut self,
        session: &SessionRecord,
        left_before: &AssetRecord,
        right_before: &AssetRecord,
        left_after: &AssetRecord,
        right_after: &AssetRecord,
        winner: &AssetId,
        left_utility: f32,
        right_utility: f32,
        embedding_head: Option<&SessionEmbeddingHead>,
        left_cache: &HierarchicalAssetQualityCacheV1,
        right_cache: &HierarchicalAssetQualityCacheV1,
        session_cache: &HierarchicalSessionQualityCacheV1,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening hierarchical comparison transaction")?;
        update_asset(&tx, left_after)?;
        update_asset(&tx, right_after)?;
        update_session(&tx, session)?;
        tx.execute(
            r"
            INSERT INTO comparisons (
                session_id,
                corpus_id,
                left_asset_id,
                right_asset_id,
                winner_asset_id,
                created_at,
                left_utility,
                right_utility
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                left_before.id.0,
                right_before.id.0,
                winner.0,
                now_ts(),
                left_utility,
                right_utility,
            ],
        )?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit()
            .context("committing hierarchical comparison transaction")?;
        let updated_at = now();
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: left_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                left_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: right_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                right_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: crate::quality::SessionQualityCachePayload::HierarchicalGaussianV1(
                session_cache.clone(),
            ),
            updated_at,
        })?;
        Ok(())
    }

    pub fn persist_nudge_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        direction: f32,
        utility: f32,
        frontier: f32,
        signal: f32,
        exact_offset: f32,
        projection: Option<&ProjectionModel>,
        embedding_head: Option<&SessionEmbeddingHead>,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening nudge transaction")?;

        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        write_session_offset(&tx, session.id, asset_id, exact_offset)?;
        tx.execute(
            r"
            INSERT INTO nudge_events (
                session_id,
                corpus_id,
                asset_id,
                direction,
                created_at,
                utility,
                frontier,
                signal
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                asset_id.0,
                direction,
                now_ts(),
                utility,
                frontier,
                signal,
            ],
        )?;
        upsert_projection(&tx, projection)?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit().context("committing nudge transaction")?;
        self.mirror_legacy_quality_cache(session, [asset_after]);
        Ok(())
    }

    pub fn persist_hierarchical_nudge_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        direction: f32,
        utility: f32,
        frontier: f32,
        signal: f32,
        exact_offset: f32,
        embedding_head: Option<&SessionEmbeddingHead>,
        asset_cache: &HierarchicalAssetQualityCacheV1,
        session_cache: &HierarchicalSessionQualityCacheV1,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening hierarchical nudge transaction")?;
        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        write_session_offset(&tx, session.id, asset_id, exact_offset)?;
        tx.execute(
            r"
            INSERT INTO nudge_events (
                session_id,
                corpus_id,
                asset_id,
                direction,
                created_at,
                utility,
                frontier,
                signal
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                session.id.0,
                session.corpus_id.0,
                asset_id.0,
                direction,
                now_ts(),
                utility,
                frontier,
                signal,
            ],
        )?;
        upsert_session_embedding_head(&tx, session.id, embedding_head)?;
        tx.commit()
            .context("committing hierarchical nudge transaction")?;
        let updated_at = now();
        self.save_asset_quality_cache(&StoredAssetQualityCache {
            asset_id: asset_after.id.clone(),
            payload: crate::quality::AssetQualityCachePayload::HierarchicalGaussianV1(
                asset_cache.clone(),
            ),
            updated_at,
        })?;
        self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: crate::quality::SessionQualityCachePayload::HierarchicalGaussianV1(
                session_cache.clone(),
            ),
            updated_at,
        })?;
        Ok(())
    }

    pub fn persist_heart_step(
        &mut self,
        session: &SessionRecord,
        asset_after: &AssetRecord,
        asset_id: &AssetId,
        active: bool,
    ) -> anyhow::Result<()> {
        if !active {
            return Ok(());
        }
        let has_utility = self.has_column("heart_events", "utility")?;
        let has_signal = self.has_column("heart_events", "signal")?;
        let tx = self
            .conn
            .transaction()
            .context("opening heart transaction")?;

        update_asset(&tx, asset_after)?;
        update_session(&tx, session)?;
        enshrine_session_heart(&tx, session.id, asset_id)?;
        let legacy_signal = 1.0;
        match (has_utility, has_signal) {
            (true, true) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    utility,
                    signal,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (true, false) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    utility,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (false, true) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    signal,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    legacy_signal,
                    i64::from(active),
                ],
            )?,
            (false, false) => tx.execute(
                r"
                INSERT INTO heart_events (
                    session_id,
                    corpus_id,
                    asset_id,
                    created_at,
                    active
                ) VALUES (?1, ?2, ?3, ?4, ?5)
                ",
                params![
                    session.id.0,
                    session.corpus_id.0,
                    asset_id.0,
                    now_ts(),
                    i64::from(active),
                ],
            )?,
        };
        tx.commit().context("committing heart transaction")?;
        self.mirror_legacy_quality_cache(session, [asset_after]);
        Ok(())
    }

    fn mirror_legacy_quality_cache<const N: usize>(
        &self,
        session: &SessionRecord,
        assets: [&AssetRecord; N],
    ) {
        let Ok(model) = self.active_quality_model() else {
            return;
        };
        if model.formal_version != QualityFormalVersion::LegacyIndependentV1 {
            return;
        }

        let updated_at = now();
        for asset in assets {
            if let Err(error) = self.save_asset_quality_cache(&StoredAssetQualityCache {
                asset_id: asset.id.clone(),
                payload: legacy_asset_quality_payload(asset),
                updated_at,
            }) {
                warn!(
                    error = %format!("{error:#}"),
                    asset_id = %asset.id.0,
                    "failed to mirror legacy asset quality cache"
                );
            }
        }
        if let Err(error) = self.save_session_quality_cache(&StoredSessionQualityCache {
            session_id: session.id,
            payload: legacy_session_quality_payload(session),
            updated_at,
        }) {
            warn!(
                error = %format!("{error:#}"),
                session_id = session.id.0,
                "failed to mirror legacy session quality cache"
            );
        }
    }

    pub fn persist_similarity_step(
        &mut self,
        corpus_id: CorpusId,
        model: &SimilarityModel,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        choice: SimilarityChoice,
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening similarity transaction")?;
        let (dim, mean, weights) = match model {
            SimilarityModel::Linear(model) => (model.dim, &model.mean, &model.weights),
            SimilarityModel::Ordinal(model) => {
                (model.prior.dim, &model.prior.mean, &model.prior.weights)
            }
        };

        tx.execute(
            r"
            INSERT OR REPLACE INTO similarity_models (
                corpus_id,
                model_name,
                dim,
                mean,
                weights,
                kind,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                corpus_id.0,
                model.model_name(),
                i64::try_from(dim)?,
                encode_vec_f32(mean),
                encode_vec_f32(weights),
                match model {
                    SimilarityModel::Linear(_) => "linear",
                    SimilarityModel::Ordinal(_) => "ordinal",
                },
                serde_json::to_vec(model)?,
                now_ts(),
            ],
        )?;
        tx.execute(
            r"
            INSERT INTO similarity_triads (
                corpus_id,
                model_name,
                asset_a_id,
                asset_b_id,
                asset_c_id,
                chosen_pair,
                created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                corpus_id.0,
                model.model_name(),
                asset_a.0,
                asset_b.0,
                asset_c.0,
                choice.as_str(),
                now_ts(),
            ],
        )?;
        tx.commit().context("committing similarity transaction")?;
        Ok(())
    }
}
