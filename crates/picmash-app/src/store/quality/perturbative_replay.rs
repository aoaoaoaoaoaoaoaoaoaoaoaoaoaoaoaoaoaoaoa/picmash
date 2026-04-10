use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, bail};
use rusqlite::params;

use super::*;
use crate::asset_domain::AssetDomainLabel;
mod kernels;
mod state;
#[cfg(test)]
mod tests;

use self::{kernels::*, state::*};

impl Store {
    pub(super) fn rebuild_hierarchical_perturbative_v3(
        &mut self,
        projection_model_name: &str,
    ) -> anyhow::Result<crate::quality::QualityReplayStats> {
        let prepared = self.prepare_perturbative_replay_v3(projection_model_name)?;
        self.commit_prepared_perturbative_replay(prepared)?
            .context("prepared perturbative replay went stale before commit")
    }

    pub(crate) fn prepare_perturbative_replay_v3(
        &self,
        projection_model_name: &str,
    ) -> anyhow::Result<PreparedPerturbativeReplay> {
        let model = self.active_quality_model()?;
        let replay = self.read_replay_events()?;
        let frontier = replay.frontier();
        let bootstrap_head = self
            .load_linear_technical_prior_head(&model)?
            .unwrap_or_default();
        let bootstrap_hyper = self
            .load_perturbative_hyper_params(&model)?
            .unwrap_or_default();
        let mut hyper = bootstrap_hyper;
        let mut state =
            self.seed_perturbative_replay_state(projection_model_name, &bootstrap_head, hyper)?;
        self.apply_perturbative_replay_stream(&mut state, &replay, hyper)?;
        gauge_fix_perturbative_technical(&mut state, hyper);
        let fitted_head =
            fit_linear_technical_prior_head(&self.perturbative_technical_prior_samples(&state))
                .unwrap_or(bootstrap_head);
        let fitted_hyper = fit_perturbative_hyper_params(&state, &replay, hyper);
        if fitted_head.weights != bootstrap_head.weights
            || fitted_head.bias != bootstrap_head.bias
            || fitted_head.residual_variance != bootstrap_head.residual_variance
            || fitted_hyper != bootstrap_hyper
        {
            hyper = fitted_hyper;
            state =
                self.seed_perturbative_replay_state(projection_model_name, &fitted_head, hyper)?;
            self.apply_perturbative_replay_stream(&mut state, &replay, hyper)?;
            gauge_fix_perturbative_technical(&mut state, hyper);
        }
        let subject_snapshot = state
            .subjects
            .iter()
            .map(|(identity_id, subject)| (*identity_id, subject.beauty, subject.duel_count))
            .collect::<Vec<_>>();
        let asset_cache_payloads = state
            .assets
            .iter()
            .map(|(asset_id, state_asset)| {
                Ok((
                    asset_id.clone(),
                    encode_quality_payload(
                        &crate::quality::AssetQualityCachePayload::HierarchicalPerturbativeV3(
                            crate::quality::PerturbativeAssetQualityCacheV3 {
                                baseline_mean: state_asset.baseline_mean,
                                baseline_variance: state_asset.baseline_variance,
                                canonical_mean: perturbative_asset_canonical_mean(
                                    state_asset,
                                    hyper,
                                ),
                                canonical_variance: perturbative_asset_canonical_variance(
                                    state_asset,
                                    hyper,
                                ),
                                perturbation_basis: state_asset.perturbation_basis.to_vec(),
                                technical_mean: state_asset.technical_mean,
                                technical_variance: state_asset.technical_variance,
                                domain_label: state_asset.domain_label,
                            },
                        ),
                    )?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let session_cache_payloads = state
            .sessions
            .iter()
            .map(|(session_id, session)| {
                Ok((
                    *session_id,
                    encode_quality_payload(
                        &crate::quality::SessionQualityCachePayload::HierarchicalPerturbativeV3(
                            crate::quality::PerturbativeSessionQualityCacheV3 {
                                perturbation_weight_mean: session.perturbation_weight_mean.to_vec(),
                                perturbation_weight_variance: session
                                    .perturbation_weight_variance
                                    .to_vec(),
                                importance_raw_mean: session.importance_raw_mean,
                                importance_raw_variance: session.importance_raw_variance,
                                threshold_mean: session.threshold_mean,
                                threshold_variance: session.threshold_variance,
                            },
                        ),
                    )?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let subject_cache_payloads = subject_snapshot
            .iter()
            .map(|(identity_id, beauty, duel_count)| {
                Ok((
                    *identity_id,
                    encode_quality_payload(
                        &crate::quality::SubjectQualityCachePayload::HierarchicalPerturbativeV3(
                            crate::quality::HierarchicalSubjectQualityCacheV1 {
                                beauty_mean: beauty.mean,
                                beauty_variance: beauty.sigma * beauty.sigma,
                                duel_count: *duel_count,
                            },
                        ),
                    )?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let external_cache_payloads = state
            .external_items
            .iter()
            .map(|(item_id, state_item)| {
                Ok((
                    *item_id,
                    encode_quality_payload(
                        &crate::quality::AssetQualityCachePayload::HierarchicalPerturbativeV3(
                            crate::quality::PerturbativeAssetQualityCacheV3 {
                                baseline_mean: state_item.baseline_mean,
                                baseline_variance: state_item.baseline_variance,
                                canonical_mean: perturbative_external_canonical_mean(
                                    state_item, hyper,
                                ),
                                canonical_variance: perturbative_external_canonical_variance(
                                    state_item, hyper,
                                ),
                                perturbation_basis: state_item.perturbation_basis.to_vec(),
                                technical_mean: state_item.technical_mean,
                                technical_variance: state_item.technical_variance,
                                domain_label: state_item.domain_label,
                            },
                        ),
                    )?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let technical_head_payload = encode_quality_payload(&fitted_head)?;
        let hyper_payload = encode_quality_payload(&hyper)?;
        let stats = crate::quality::QualityReplayStats {
            formal_version: crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3,
            asset_count: state.assets.len(),
            session_count: state.sessions.len(),
            subject_count: subject_snapshot.len(),
            comparison_events: state.comparison_events,
            nudge_events: state.nudge_events,
            heart_events: state.heart_events,
        };
        Ok(PreparedPerturbativeReplay {
            frontier,
            model,
            state,
            subject_snapshot,
            asset_cache_payloads,
            session_cache_payloads,
            subject_cache_payloads,
            external_cache_payloads,
            technical_head_payload,
            hyper_payload,
            stats,
        })
    }

    pub(crate) fn commit_prepared_perturbative_replay(
        &mut self,
        prepared: PreparedPerturbativeReplay,
    ) -> anyhow::Result<Option<crate::quality::QualityReplayStats>> {
        if self.current_replay_frontier()? != prepared.frontier {
            return Ok(None);
        }
        self.persist_perturbative_replay_state(&prepared)?;
        Ok(Some(prepared.stats))
    }

    fn seed_perturbative_replay_state(
        &self,
        projection_model_name: &str,
        technical_head: &crate::quality_features::LinearTechnicalPriorHead,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<PerturbativeReplayState> {
        let embeddings = self
            .conn
            .prepare(
                r"
                SELECT asset_id, vector
                FROM embeddings
                WHERE model_name = ?1
                ",
            )?
            .query_map(params![projection_model_name], |row| {
                Ok((
                    AssetId(row.get::<_, String>(0)?),
                    decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
                ))
            })?
            .collect::<Result<HashMap<_, _>, _>>()?;
        let external_embeddings = self
            .conn
            .prepare(
                r"
                SELECT id, embedding
                FROM external_items
                WHERE embedding_model = ?1
                  AND embedding IS NOT NULL
                ",
            )?
            .query_map(params![projection_model_name], |row| {
                Ok((
                    RemoteItemId(row.get::<_, i64>(0)?),
                    decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
                ))
            })?
            .collect::<Result<HashMap<_, _>, _>>()?;
        let feature_rows = self
            .conn
            .prepare(
                r"
                SELECT asset_id, technical_payload, vibe_payload
                FROM asset_quality_features
                WHERE extractor_revision = ?1
                ",
            )?
            .query_map(
                params![crate::quality_features::QUALITY_FEATURE_REVISION],
                |row| {
                    Ok((
                        AssetId(row.get::<_, String>(0)?),
                        crate::quality_features::AssetQualityFeatures {
                            technical: decode_quality_payload(&row.get::<_, String>(1)?)
                                .map_err(into_rusqlite)?,
                            vibe: decode_quality_payload(&row.get::<_, String>(2)?)
                                .map_err(into_rusqlite)?,
                        },
                    ))
                },
            )?
            .collect::<Result<HashMap<_, _>, _>>()?;
        let external_feature_rows = self
            .conn
            .prepare(
                r"
                SELECT item_id, technical_payload, vibe_payload
                FROM external_item_quality_features
                WHERE extractor_revision = ?1
                ",
            )?
            .query_map(
                params![crate::quality_features::QUALITY_FEATURE_REVISION],
                |row| {
                    Ok((
                        RemoteItemId(row.get::<_, i64>(0)?),
                        crate::quality_features::AssetQualityFeatures {
                            technical: decode_quality_payload(&row.get::<_, String>(1)?)
                                .map_err(into_rusqlite)?,
                            vibe: decode_quality_payload(&row.get::<_, String>(2)?)
                                .map_err(into_rusqlite)?,
                        },
                    ))
                },
            )?
            .collect::<Result<HashMap<_, _>, _>>()?;
        let domain_gates =
            self.global_hierarchical_domain_gates(projection_model_name, &embeddings)?;
        let external_domain_gates = self.global_hierarchical_external_domain_gates(
            projection_model_name,
            &external_embeddings,
        )?;
        let semantic_basis = SemanticPriorBasis::fit(&embeddings, &external_embeddings);
        let vibe_standardization = fit_vibe_standardization(
            &feature_rows,
            &external_feature_rows,
            &domain_gates,
            &external_domain_gates,
        );
        let center_seed = self.perturbative_center_seed()?;
        let subjects = self
            .compute_identity_beauty_snapshot()?
            .into_iter()
            .map(|(identity_id, beauty, duel_count)| {
                (identity_id, HierarchicalSubjectState { beauty, duel_count })
            })
            .collect::<HashMap<_, _>>();
        let dominant_faces = self.dominant_face_beauty_by_asset()?;

        let assets = self
            .conn
            .prepare(
                r"
                SELECT id, rotation_quarters
                FROM assets
                ",
            )?
            .query_map([], |row| {
                Ok((AssetId(row.get::<_, String>(0)?), row.get::<_, i32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(id, rotation_quarters)| -> anyhow::Result<_> {
                let feature = feature_rows
                    .get(&id)
                    .copied()
                    .unwrap_or_else(crate::quality_features::AssetQualityFeatures::neutral);
                let gate_3d = domain_gates.get(&id).copied().unwrap_or(false);
                let domain_label = if gate_3d {
                    AssetDomainLabel::Real
                } else {
                    AssetDomainLabel::Anime
                };
                let technical_mean = gate_3d.then(|| {
                    crate::quality_features::technical_prior_mean_with_head(
                        technical_head,
                        &feature.technical,
                    )
                });
                let technical_variance = gate_3d.then(|| {
                    crate::quality_features::technical_prior_variance_with_head(
                        technical_head,
                        &feature.technical,
                    )
                    .max(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                });
                let perturbation_basis = perturbation_basis(
                    semantic_basis.as_ref().and_then(|basis| {
                        embeddings
                            .get(&id)
                            .map(|embedding| basis.project(embedding))
                    }),
                    standardized_vibe(&feature, gate_3d, &vibe_standardization, hyper),
                );
                let (
                    baseline_mean,
                    baseline_variance,
                    seeded_technical_mean,
                    seeded_technical_variance,
                ) = self.seed_perturbative_asset_anchor(
                    &id,
                    center_seed,
                    technical_mean,
                    technical_variance,
                )?;
                let mut state = PerturbativeReplayAssetState {
                    asset: AssetRecord {
                        id: id.clone(),
                        path: PathBuf::new(),
                        visual_key: None,
                        width: 0,
                        height: 0,
                        alpha: 0.0,
                        coords: [0.0; crate::model::LATENT_DIM],
                        rotation_quarters,
                        compare_count: 0,
                        win_count: 0,
                        heart_count: 0,
                        is_hearted: false,
                        hidden: false,
                    },
                    domain_label,
                    baseline_mean,
                    baseline_variance,
                    perturbation_basis,
                    technical_mean: seeded_technical_mean,
                    technical_variance: seeded_technical_variance,
                    face: dominant_faces
                        .get(&id)
                        .copied()
                        .map(|face| super::face_anchor_for_subject(face, &subjects)),
                };
                sync_perturbative_asset_record(&mut state, hyper);
                Ok((id, state))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let external_items = external_embeddings
            .keys()
            .copied()
            .map(|item_id| -> anyhow::Result<_> {
                let feature = external_feature_rows
                    .get(&item_id)
                    .copied()
                    .unwrap_or_else(crate::quality_features::AssetQualityFeatures::neutral);
                let gate_3d = external_domain_gates
                    .get(&item_id)
                    .copied()
                    .unwrap_or(false);
                let domain_label = if gate_3d {
                    AssetDomainLabel::Real
                } else {
                    AssetDomainLabel::Anime
                };
                let technical_mean = gate_3d.then(|| {
                    crate::quality_features::technical_prior_mean_with_head(
                        technical_head,
                        &feature.technical,
                    )
                });
                let technical_variance = gate_3d.then(|| {
                    crate::quality_features::technical_prior_variance_with_head(
                        technical_head,
                        &feature.technical,
                    )
                    .max(crate::quality::HIERARCHICAL_TECH_PRIOR_FLOOR)
                });
                let perturbation_basis = perturbation_basis(
                    semantic_basis.as_ref().and_then(|basis| {
                        external_embeddings
                            .get(&item_id)
                            .map(|embedding| basis.project(embedding))
                    }),
                    standardized_vibe(&feature, gate_3d, &vibe_standardization, hyper),
                );
                let (
                    baseline_mean,
                    baseline_variance,
                    seeded_technical_mean,
                    seeded_technical_variance,
                ) = self.seed_perturbative_external_anchor(
                    item_id,
                    center_seed,
                    technical_mean,
                    technical_variance,
                )?;
                Ok((
                    item_id,
                    PerturbativeReplayExternalState {
                        domain_label,
                        baseline_mean,
                        baseline_variance,
                        perturbation_basis,
                        technical_mean: seeded_technical_mean,
                        technical_variance: seeded_technical_variance,
                    },
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let threshold_center = center_seed
            .map_or(crate::quality::PERTURBATIVE_THRESHOLD_CENTER, |seed| {
                seed.threshold
            });
        let sessions = self
            .conn
            .prepare(
                r"
                SELECT id, corpus_id
                FROM sessions
                ",
            )?
            .query_map([], |row| {
                Ok((
                    SessionId(row.get::<_, i64>(0)?),
                    CorpusId(row.get::<_, i64>(1)?),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(id, corpus_id)| {
                let mut state = PerturbativeReplaySessionState {
                    session: SessionRecord {
                        id,
                        corpus_id,
                        mood: [0.0; crate::model::LATENT_DIM],
                        frontier: threshold_center,
                        comparisons: 0,
                        nudges: 0,
                        hearts: 0,
                    },
                    perturbation_weight_mean: [0.0; crate::quality::PERTURBATIVE_DIM],
                    perturbation_weight_variance:
                        [crate::quality::PERTURBATIVE_SESSION_WEIGHT_PRIOR_VARIANCE;
                            crate::quality::PERTURBATIVE_DIM],
                    importance_raw_mean:
                        crate::quality::PERTURBATIVE_SESSION_IMPORTANCE_RAW_PRIOR_MEAN,
                    importance_raw_variance:
                        crate::quality::PERTURBATIVE_SESSION_IMPORTANCE_RAW_PRIOR_VARIANCE,
                    threshold_mean: threshold_center,
                    threshold_variance: crate::quality::PERTURBATIVE_THRESHOLD_PRIOR_VARIANCE,
                    exact_offsets: HashMap::new(),
                    hearted_assets: HashSet::new(),
                };
                sync_perturbative_session_record(&mut state);
                (id, state)
            })
            .collect::<HashMap<_, _>>();

        Ok(PerturbativeReplayState {
            assets,
            external_items,
            subjects,
            sessions,
            comparison_events: 0,
            nudge_events: 0,
            heart_events: 0,
            external_events: 0,
            max_comparison_id: 0,
            max_nudge_id: 0,
            max_heart_id: 0,
            max_external_id: 0,
        })
    }

    fn perturbative_center_seed(&self) -> anyhow::Result<Option<PerturbativeCenterSeed>> {
        let center_session_id = self
            .conn
            .query_row(
                r"
                SELECT id
                FROM sessions
                ORDER BY last_touched_at DESC, started_at DESC, id DESC
                LIMIT 1
                ",
                [],
                |row| row.get::<_, i64>(0).map(SessionId),
            )
            .optional()?;
        let Some(center_session_id) = center_session_id else {
            return Ok(None);
        };
        let seed = self
            .session_quality_cache(
                center_session_id,
                crate::quality::QualityFormalVersion::HierarchicalGaussianV1,
            )?
            .and_then(|cache| crate::quality::HierarchicalSessionPosterior::decode(&cache.payload))
            .map(|seed| PerturbativeCenterSeed {
                semantic: seed.semantic_mood_mean,
                vibe: seed.vibe_mean,
                threshold: seed.frontier_mean,
            });
        Ok(seed)
    }

    fn seed_perturbative_asset_anchor(
        &self,
        asset_id: &AssetId,
        center_seed: Option<PerturbativeCenterSeed>,
        technical_mean: Option<f32>,
        technical_variance: Option<f32>,
    ) -> anyhow::Result<(f32, f32, Option<f32>, Option<f32>)> {
        let seeded = self
            .asset_quality_cache(
                asset_id,
                crate::quality::QualityFormalVersion::HierarchicalGaussianV1,
            )?
            .and_then(|cache| crate::quality::HierarchicalAssetPosterior::decode(&cache.payload));
        let Some(seed) = seeded else {
            return Ok((
                0.0,
                crate::quality::HIERARCHICAL_BASELINE_PRIOR_VARIANCE,
                technical_mean,
                technical_variance,
            ));
        };
        let center_shift = center_seed.map_or(0.0, |center| {
            crate::model::dot(&seed.mood_loading_mean, &center.semantic)
                + seed
                    .technical_mean
                    .map(|_| {
                        seed.vibe_mean
                            .iter()
                            .zip(center.vibe.iter())
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum::<f32>()
                    })
                    .unwrap_or_default()
        });
        Ok((
            seed.baseline_mean + center_shift,
            seed.baseline_variance,
            seed.technical_mean.or(technical_mean),
            seed.technical_variance.or(technical_variance),
        ))
    }

    fn seed_perturbative_external_anchor(
        &self,
        item_id: RemoteItemId,
        center_seed: Option<PerturbativeCenterSeed>,
        technical_mean: Option<f32>,
        technical_variance: Option<f32>,
    ) -> anyhow::Result<(f32, f32, Option<f32>, Option<f32>)> {
        let seeded = self
            .external_item_quality_cache(
                item_id,
                crate::quality::QualityFormalVersion::HierarchicalGaussianV1,
            )?
            .and_then(|cache| crate::quality::HierarchicalAssetPosterior::decode(&cache.payload));
        let Some(seed) = seeded else {
            return Ok((
                0.0,
                crate::quality::HIERARCHICAL_BASELINE_PRIOR_VARIANCE,
                technical_mean,
                technical_variance,
            ));
        };
        let center_shift = center_seed.map_or(0.0, |center| {
            crate::model::dot(&seed.mood_loading_mean, &center.semantic)
                + seed
                    .technical_mean
                    .map(|_| {
                        seed.vibe_mean
                            .iter()
                            .zip(center.vibe.iter())
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum::<f32>()
                    })
                    .unwrap_or_default()
        });
        Ok((
            seed.baseline_mean + center_shift,
            seed.baseline_variance,
            seed.technical_mean.or(technical_mean),
            seed.technical_variance.or(technical_variance),
        ))
    }

    fn apply_perturbative_replay_stream(
        &self,
        state: &mut PerturbativeReplayState,
        replay: &ReplayEventStream,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        state.comparison_events = replay.comparison_events;
        state.nudge_events = replay.nudge_events;
        state.heart_events = replay.heart_events;
        state.external_events = replay.external_events;
        state.max_comparison_id = replay.max_comparison_id;
        state.max_nudge_id = replay.max_nudge_id;
        state.max_heart_id = replay.max_heart_id;
        state.max_external_id = replay.max_external_id;
        for event in replay.events.iter().cloned() {
            match event {
                LegacyReplayEvent::Comparison(event) => {
                    self.apply_perturbative_replay_comparison(state, event, hyper)?;
                }
                LegacyReplayEvent::Nudge(event) => {
                    self.apply_perturbative_replay_nudge(state, event, hyper)?;
                }
                LegacyReplayEvent::Heart(event) => {
                    self.apply_perturbative_replay_heart(state, event, hyper)?;
                }
                LegacyReplayEvent::External(event) => {
                    self.apply_perturbative_replay_external(state, event, hyper)?;
                }
            }
        }
        Ok(())
    }

    fn apply_perturbative_replay_external(
        &self,
        state: &mut PerturbativeReplayState,
        event: HierarchicalReplayExternalEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        match event.kind {
            ExternalEventKind::LocalWin | ExternalEventKind::RemoteWin => {
                self.apply_perturbative_replay_external_duel(state, event, hyper)
            }
            ExternalEventKind::Rejected | ExternalEventKind::Kept | ExternalEventKind::Hearted => {
                self.apply_perturbative_replay_external_unary(state, event, hyper)
            }
            ExternalEventKind::Selected
            | ExternalEventKind::StreamBlocked
            | ExternalEventKind::Imported => Ok(()),
        }
    }

    fn apply_perturbative_replay_comparison(
        &self,
        state: &mut PerturbativeReplayState,
        event: LegacyReplayComparisonEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing perturbative session {}", event.session_id.0))?;
        if event.left_asset_id == event.right_asset_id {
            bail!(
                "perturbative replay encountered self duel for {}",
                event.left_asset_id.0
            );
        }
        let mut left = state
            .assets
            .remove(&event.left_asset_id)
            .with_context(|| format!("missing perturbative asset {}", event.left_asset_id.0))?;
        let mut right = state
            .assets
            .remove(&event.right_asset_id)
            .with_context(|| format!("missing perturbative asset {}", event.right_asset_id.0))?;
        refresh_perturbative_asset_face_anchor(&mut left, &state.subjects);
        refresh_perturbative_asset_face_anchor(&mut right, &state.subjects);
        let left_won = match event.winner_asset_id {
            _ if event.winner_asset_id == event.left_asset_id => true,
            _ if event.winner_asset_id == event.right_asset_id => false,
            _ => bail!(
                "comparison winner {} is not one of {} vs {}",
                event.winner_asset_id.0,
                event.left_asset_id.0,
                event.right_asset_id.0
            ),
        };
        let outcome = if left_won { 1.0 } else { -1.0 };
        let left_mean =
            perturbative_session_utility_mean(&left, session, &event.left_asset_id, hyper);
        let right_mean =
            perturbative_session_utility_mean(&right, session, &event.right_asset_id, hyper);
        let delta_mean = left_mean - right_mean;
        let delta_variance = perturbative_session_utility_variance(&left, session, hyper)
            + perturbative_session_utility_variance(&right, session, hyper);
        let Some(moments) = crate::quality::gaussian_duel_moment_match(
            delta_mean,
            delta_variance,
            outcome,
            crate::quality::HIERARCHICAL_DUEL_BETA,
        ) else {
            state.assets.insert(left.asset.id.clone(), left);
            state.assets.insert(right.asset.id.clone(), right);
            return Ok(());
        };
        crate::quality::diagonal_adf_update(
            &mut left.baseline_mean,
            &mut left.baseline_variance,
            1.0,
            outcome,
            moments,
        );
        crate::quality::diagonal_adf_update(
            &mut right.baseline_mean,
            &mut right.baseline_variance,
            -1.0,
            outcome,
            moments,
        );
        if let (Some(mean), Some(variance)) =
            (&mut left.technical_mean, &mut left.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                hyper.technical_weight(left.domain_label),
                outcome,
                moments,
            );
        }
        if let (Some(mean), Some(variance)) =
            (&mut right.technical_mean, &mut right.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                -hyper.technical_weight(right.domain_label),
                outcome,
                moments,
            );
        }
        self.apply_perturbative_face_backflow(
            &mut state.subjects,
            left.face.map(|face| (face.identity_id, left.domain_label)),
            right
                .face
                .map(|face| (face.identity_id, right.domain_label)),
            outcome,
            moments,
            hyper,
        );
        let basis_gap = subtract_basis(&left.perturbation_basis, &right.perturbation_basis);
        apply_perturbative_session_projection_update(session, &basis_gap, outcome, moments, hyper);

        left.asset.compare_count += 1;
        right.asset.compare_count += 1;
        if left_won {
            left.asset.win_count += 1;
        } else {
            right.asset.win_count += 1;
        }
        session.session.comparisons += 1;
        refresh_perturbative_asset_face_anchor(&mut left, &state.subjects);
        refresh_perturbative_asset_face_anchor(&mut right, &state.subjects);
        sync_perturbative_asset_record(&mut left, hyper);
        sync_perturbative_asset_record(&mut right, hyper);
        sync_perturbative_session_record(session);
        state.assets.insert(left.asset.id.clone(), left);
        state.assets.insert(right.asset.id.clone(), right);
        Ok(())
    }

    fn apply_perturbative_replay_nudge(
        &self,
        state: &mut PerturbativeReplayState,
        event: LegacyReplayNudgeEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing perturbative session {}", event.session_id.0))?;
        let mut asset = state
            .assets
            .remove(&event.asset_id)
            .with_context(|| format!("missing perturbative asset {}", event.asset_id.0))?;
        refresh_perturbative_asset_face_anchor(&mut asset, &state.subjects);
        let feedback = ExternalUnaryFeedback::from_direction(event.direction)?;
        let utility_mean =
            perturbative_session_utility_mean(&asset, session, &event.asset_id, hyper);
        let utility_variance = perturbative_session_utility_variance(&asset, session, hyper);
        let Some(moments) = crate::quality::gaussian_duel_moment_match(
            utility_mean - session.threshold_mean,
            utility_variance + session.threshold_variance,
            feedback.outcome(),
            feedback.beta(),
        ) else {
            state.assets.insert(asset.asset.id.clone(), asset);
            return Ok(());
        };
        crate::quality::diagonal_adf_update(
            &mut asset.baseline_mean,
            &mut asset.baseline_variance,
            1.0,
            feedback.outcome(),
            moments,
        );
        if let (Some(mean), Some(variance)) =
            (&mut asset.technical_mean, &mut asset.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                hyper.technical_weight(asset.domain_label),
                feedback.outcome(),
                moments,
            );
        }
        apply_perturbative_session_projection_update(
            session,
            &asset.perturbation_basis,
            feedback.outcome(),
            moments,
            hyper,
        );
        crate::quality::diagonal_adf_update(
            &mut session.threshold_mean,
            &mut session.threshold_variance,
            -1.0,
            feedback.outcome(),
            moments,
        );
        if let Some(identity_id) = asset.face.map(|face| face.identity_id) {
            self.apply_perturbative_face_unary_backflow(
                &mut state.subjects,
                identity_id,
                asset.domain_label,
                feedback.outcome(),
                moments,
                hyper,
            );
        }
        let exact_offset = session
            .exact_offsets
            .get(&event.asset_id)
            .copied()
            .unwrap_or_default();
        let next_offset = exact_offset
            + crate::quality::LEGACY_LR_NUDGE_OFFSET
                * (feedback.outcome() - crate::quality::LEGACY_L2_OFFSET * exact_offset);
        if next_offset.abs() < crate::quality::LEGACY_EXACT_OFFSET_EPSILON {
            session.exact_offsets.remove(&event.asset_id);
        } else {
            session.exact_offsets.insert(event.asset_id, next_offset);
        }
        session.session.nudges = session.session.nudges.saturating_add(1);
        refresh_perturbative_asset_face_anchor(&mut asset, &state.subjects);
        sync_perturbative_asset_record(&mut asset, hyper);
        sync_perturbative_session_record(session);
        state.assets.insert(asset.asset.id.clone(), asset);
        Ok(())
    }

    fn apply_perturbative_replay_heart(
        &self,
        state: &mut PerturbativeReplayState,
        event: LegacyReplayHeartEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing perturbative session {}", event.session_id.0))?;
        let mut asset = state
            .assets
            .remove(&event.asset_id)
            .with_context(|| format!("missing perturbative asset {}", event.asset_id.0))?;
        refresh_perturbative_asset_face_anchor(&mut asset, &state.subjects);
        if !event.active || asset.asset.is_hearted {
            state.assets.insert(asset.asset.id.clone(), asset);
            return Ok(());
        }
        let Some(moments) = crate::quality::gaussian_duel_moment_match(
            perturbative_session_utility_mean(&asset, session, &event.asset_id, hyper)
                - session.threshold_mean,
            perturbative_session_utility_variance(&asset, session, hyper)
                + session.threshold_variance,
            1.0,
            crate::quality::HIERARCHICAL_UNARY_HEART_BETA,
        ) else {
            state.assets.insert(asset.asset.id.clone(), asset);
            return Ok(());
        };
        crate::quality::diagonal_adf_update(
            &mut asset.baseline_mean,
            &mut asset.baseline_variance,
            1.0,
            1.0,
            moments,
        );
        if let (Some(mean), Some(variance)) =
            (&mut asset.technical_mean, &mut asset.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                hyper.technical_weight(asset.domain_label),
                1.0,
                moments,
            );
        }
        apply_perturbative_session_projection_update(
            session,
            &asset.perturbation_basis,
            1.0,
            moments,
            hyper,
        );
        crate::quality::diagonal_adf_update(
            &mut session.threshold_mean,
            &mut session.threshold_variance,
            -1.0,
            1.0,
            moments,
        );
        if let Some(identity_id) = asset.face.map(|face| face.identity_id) {
            self.apply_perturbative_face_unary_backflow(
                &mut state.subjects,
                identity_id,
                asset.domain_label,
                1.0,
                moments,
                hyper,
            );
        }
        asset.asset.is_hearted = true;
        asset.asset.heart_count = asset.asset.heart_count.saturating_add(1);
        session.session.hearts = session.session.hearts.saturating_add(1);
        session.hearted_assets.insert(event.asset_id);
        refresh_perturbative_asset_face_anchor(&mut asset, &state.subjects);
        sync_perturbative_asset_record(&mut asset, hyper);
        sync_perturbative_session_record(session);
        state.assets.insert(asset.asset.id.clone(), asset);
        Ok(())
    }

    fn apply_perturbative_replay_external_duel(
        &self,
        state: &mut PerturbativeReplayState,
        event: HierarchicalReplayExternalEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
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
        refresh_perturbative_asset_face_anchor(&mut local, &state.subjects);
        let outcome = if matches!(event.kind, ExternalEventKind::LocalWin) {
            1.0
        } else {
            -1.0
        };
        let local_mean = perturbative_session_utility_mean(&local, session, local_asset_id, hyper);
        let remote_mean = perturbative_external_session_utility_mean(&remote, session, hyper);
        let delta_mean = local_mean - remote_mean;
        let delta_variance = perturbative_session_utility_variance(&local, session, hyper)
            + perturbative_external_session_utility_variance(&remote, session, hyper);
        let Some(moments) = crate::quality::gaussian_duel_moment_match(
            delta_mean,
            delta_variance,
            outcome,
            crate::quality::HIERARCHICAL_DUEL_BETA,
        ) else {
            state.assets.insert(local.asset.id.clone(), local);
            state.external_items.insert(event.item_id, remote);
            return Ok(());
        };

        crate::quality::diagonal_adf_update(
            &mut local.baseline_mean,
            &mut local.baseline_variance,
            1.0,
            outcome,
            moments,
        );
        crate::quality::diagonal_adf_update(
            &mut remote.baseline_mean,
            &mut remote.baseline_variance,
            -1.0,
            outcome,
            moments,
        );
        if let (Some(mean), Some(variance)) =
            (&mut local.technical_mean, &mut local.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                hyper.technical_weight(local.domain_label),
                outcome,
                moments,
            );
        }
        if let (Some(mean), Some(variance)) =
            (&mut remote.technical_mean, &mut remote.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                -hyper.technical_weight(remote.domain_label),
                outcome,
                moments,
            );
        }
        if let Some(identity_id) = local.face.map(|face| face.identity_id) {
            self.apply_perturbative_face_unary_backflow(
                &mut state.subjects,
                identity_id,
                local.domain_label,
                outcome,
                moments,
                hyper,
            );
        }
        let basis_gap = subtract_basis(&local.perturbation_basis, &remote.perturbation_basis);
        apply_perturbative_session_projection_update(session, &basis_gap, outcome, moments, hyper);

        local.asset.compare_count += 1;
        if matches!(event.kind, ExternalEventKind::LocalWin) {
            local.asset.win_count += 1;
        }
        session.session.comparisons += 1;
        refresh_perturbative_asset_face_anchor(&mut local, &state.subjects);
        sync_perturbative_asset_record(&mut local, hyper);
        sync_perturbative_session_record(session);
        state.assets.insert(local.asset.id.clone(), local);
        state.external_items.insert(event.item_id, remote);
        Ok(())
    }

    fn apply_perturbative_replay_external_unary(
        &self,
        state: &mut PerturbativeReplayState,
        event: HierarchicalReplayExternalEvent,
        hyper: crate::quality::PerturbativeHyperParamsV3,
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
        let utility_mean = perturbative_external_session_utility_mean(&remote, session, hyper);
        let utility_variance =
            perturbative_external_session_utility_variance(&remote, session, hyper);
        let Some(moments) = crate::quality::gaussian_duel_moment_match(
            utility_mean - session.threshold_mean,
            utility_variance + session.threshold_variance,
            feedback.outcome(),
            feedback.beta(),
        ) else {
            state.external_items.insert(event.item_id, remote);
            return Ok(());
        };
        crate::quality::diagonal_adf_update(
            &mut remote.baseline_mean,
            &mut remote.baseline_variance,
            1.0,
            feedback.outcome(),
            moments,
        );
        if let (Some(mean), Some(variance)) =
            (&mut remote.technical_mean, &mut remote.technical_variance)
        {
            crate::quality::diagonal_adf_update(
                mean,
                variance,
                hyper.technical_weight(remote.domain_label),
                feedback.outcome(),
                moments,
            );
        }
        apply_perturbative_session_projection_update(
            session,
            &remote.perturbation_basis,
            feedback.outcome(),
            moments,
            hyper,
        );
        crate::quality::diagonal_adf_update(
            &mut session.threshold_mean,
            &mut session.threshold_variance,
            -1.0,
            feedback.outcome(),
            moments,
        );
        if matches!(feedback, ExternalUnaryFeedback::Heart) {
            session.session.hearts = session.session.hearts.saturating_add(1);
        } else {
            session.session.nudges = session.session.nudges.saturating_add(1);
        }
        sync_perturbative_session_record(session);
        state.external_items.insert(event.item_id, remote);
        Ok(())
    }

    fn apply_perturbative_face_backflow(
        &self,
        subjects: &mut HashMap<FaceIdentityId, HierarchicalSubjectState>,
        left: Option<(FaceIdentityId, AssetDomainLabel)>,
        right: Option<(FaceIdentityId, AssetDomainLabel)>,
        outcome: f32,
        moments: crate::quality::GaussianMomentMatch,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) {
        match (left, right) {
            (Some((left_id, left_domain)), Some((right_id, right_domain)))
                if left_id != right_id =>
            {
                self.apply_perturbative_face_unary_backflow(
                    subjects,
                    left_id,
                    left_domain,
                    outcome,
                    moments,
                    hyper,
                );
                self.apply_perturbative_face_unary_backflow(
                    subjects,
                    right_id,
                    right_domain,
                    -outcome,
                    moments,
                    hyper,
                );
            }
            (Some((identity_id, domain_label)), None) => {
                self.apply_perturbative_face_unary_backflow(
                    subjects,
                    identity_id,
                    domain_label,
                    outcome,
                    moments,
                    hyper,
                );
            }
            (None, Some((identity_id, domain_label))) => {
                self.apply_perturbative_face_unary_backflow(
                    subjects,
                    identity_id,
                    domain_label,
                    -outcome,
                    moments,
                    hyper,
                );
            }
            _ => {}
        }
    }

    fn apply_perturbative_face_unary_backflow(
        &self,
        subjects: &mut HashMap<FaceIdentityId, HierarchicalSubjectState>,
        identity_id: FaceIdentityId,
        domain_label: AssetDomainLabel,
        outcome: f32,
        moments: crate::quality::GaussianMomentMatch,
        hyper: crate::quality::PerturbativeHyperParamsV3,
    ) {
        if let Some(subject) = subjects.get_mut(&identity_id) {
            let mut variance = subject
                .beauty
                .sigma
                .powi(2)
                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE);
            crate::quality::diagonal_adf_update(
                &mut subject.beauty.mean,
                &mut variance,
                hyper.face_weight(domain_label)
                    * crate::quality::hierarchical_face_backflow_coeff(),
                outcome,
                moments,
            );
            subject.beauty.sigma = variance
                .max(crate::quality::HIERARCHICAL_MIN_VARIANCE)
                .sqrt();
        }
    }

    fn perturbative_technical_prior_samples(
        &self,
        state: &PerturbativeReplayState,
    ) -> Vec<crate::quality_features::TechnicalPriorSample> {
        let mut samples = state
            .assets
            .values()
            .filter_map(|asset| {
                let Some((mean, variance)) = asset.technical_mean.zip(asset.technical_variance)
                else {
                    return None;
                };
                let descriptor = self
                    .asset_quality_features(
                        &asset.asset.id,
                        crate::quality_features::QUALITY_FEATURE_REVISION,
                    )
                    .ok()
                    .flatten()?
                    .features
                    .technical;
                Some(crate::quality_features::TechnicalPriorSample {
                    descriptor,
                    target_mean: mean,
                    target_variance: variance,
                })
            })
            .collect::<Vec<_>>();
        samples.extend(state.external_items.iter().filter_map(|(item_id, item)| {
            let Some((mean, variance)) = item.technical_mean.zip(item.technical_variance) else {
                return None;
            };
            let descriptor = self
                .external_item_quality_features(
                    *item_id,
                    crate::quality_features::QUALITY_FEATURE_REVISION,
                )
                .ok()
                .flatten()?
                .technical;
            Some(crate::quality_features::TechnicalPriorSample {
                descriptor,
                target_mean: mean,
                target_variance: variance,
            })
        }));
        samples
    }

    fn persist_perturbative_replay_state(
        &mut self,
        prepared: &PreparedPerturbativeReplay,
    ) -> anyhow::Result<()> {
        let updated_at = now_ts();
        let state = &prepared.state;
        let tx = self
            .conn
            .transaction()
            .context("opening perturbative quality replay transaction")?;
        tx.execute(
            "DELETE FROM quality_asset_cache WHERE formal_version = ?1",
            params![crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_session_cache WHERE formal_version = ?1",
            params![crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_subject_cache WHERE formal_version = ?1",
            params![crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_external_item_cache WHERE formal_version = ?1",
            params![crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )?;
        tx.execute(
            "DELETE FROM quality_replay_cursors WHERE formal_version = ?1",
            params![crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str()],
        )?;
        tx.execute("DELETE FROM session_asset_offsets", [])?;
        tx.execute("DELETE FROM session_asset_hearts", [])?;
        tx.execute("DELETE FROM session_embedding_heads", [])?;

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
            INSERT INTO quality_asset_cache (formal_version, asset_id, payload, updated_at)
            VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_session_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_session_cache (formal_version, session_id, payload, updated_at)
            VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_subject_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_subject_cache (formal_version, identity_id, payload, updated_at)
            VALUES (?1, ?2, ?3, ?4)
            ",
        )?;
        let mut insert_external_cache_stmt = tx.prepare(
            r"
            INSERT INTO quality_external_item_cache (formal_version, item_id, payload, updated_at)
            VALUES (?1, ?2, ?3, ?4)
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
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str(),
                state_asset.asset.id.0,
                prepared.asset_cache_payloads[&state_asset.asset.id],
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
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str(),
                session.session.id.0,
                prepared.session_cache_payloads[&session.session.id],
                updated_at,
            ])?;
            for (asset_id, offset) in &session.exact_offsets {
                write_session_offset(&tx, session.session.id, asset_id, *offset)?;
            }
            for asset_id in &session.hearted_assets {
                enshrine_session_heart(&tx, session.session.id, asset_id)?;
            }
        }

        for &(identity_id, beauty, duel_count) in &prepared.subject_snapshot {
            update_subject_stmt.execute(params![
                identity_id.0,
                beauty.mean,
                beauty.sigma,
                duel_count
            ])?;
            insert_subject_cache_stmt.execute(params![
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str(),
                identity_id.0,
                prepared.subject_cache_payloads[&identity_id],
                updated_at,
            ])?;
        }

        for &item_id in state.external_items.keys() {
            insert_external_cache_stmt.execute(params![
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3.as_str(),
                item_id.0,
                prepared.external_cache_payloads[&item_id],
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
                prepared.model.formal_version.as_str(),
                prepared.model.prior_family.as_str(),
                prepared.model.prior_revision.as_str(),
                crate::quality::technical_prior_artifact_key(),
                prepared.technical_head_payload,
                updated_at,
            ],
        )?;
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
                prepared.model.formal_version.as_str(),
                prepared.model.prior_family.as_str(),
                prepared.model.prior_revision.as_str(),
                crate::quality::perturbative_hyper_artifact_key(),
                prepared.hyper_payload,
                updated_at,
            ],
        )?;
        insert_quality_replay_cursor(
            &tx,
            crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3,
            "comparison_max_id",
            state.max_comparison_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3,
            "nudge_max_id",
            state.max_nudge_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3,
            "heart_max_id",
            state.max_heart_id,
            updated_at,
        )?;
        insert_quality_replay_cursor(
            &tx,
            crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3,
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
            .context("committing perturbative quality replay transaction")?;
        Ok(())
    }
}
