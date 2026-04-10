use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

mod external_replay;
mod perturbative_replay;
mod semantic_prior;

use self::external_replay::{
    ensure_projection_dim, face_anchor_for_subject, hierarchical_session_utility_mean,
    hierarchical_session_utility_variance, refresh_asset_face_anchor, replay_session_utility,
    sync_hierarchical_asset_record,
};
use self::semantic_prior::SemanticPriorBasis;

use anyhow::{Context, bail};
use rusqlite::{OptionalExtension, params};
use time::OffsetDateTime;

use super::*;
use crate::{
    asset_domain::{AssetDomainLabel, AssetDomainOracle},
    facemash::FaceBeauty,
    model::{
        AssetId, AssetRecord, ExternalEventKind, FaceIdentityId, ProjectionModel, RemoteItemId,
        SessionEmbeddingHead, SessionId, SessionRecord, session_utility, sigmoid, subtract,
    },
    quality::{
        ACTIVE_QUALITY_MODEL_SLOT, AssetQualityCachePayload, HIERARCHICAL_BASELINE_PRIOR_VARIANCE,
        HIERARCHICAL_DUEL_BETA, HIERARCHICAL_FACE_WEIGHT, HIERARCHICAL_FRONTIER_PRIOR_VARIANCE,
        HIERARCHICAL_LOADING_PRIOR_VARIANCE, HIERARCHICAL_MIN_VARIANCE,
        HIERARCHICAL_SESSION_MOOD_PRIOR_VARIANCE, HIERARCHICAL_SESSION_VIBE_PRIOR_VARIANCE,
        HIERARCHICAL_TECH_PRIOR_FLOOR, HIERARCHICAL_TECH_WEIGHT, HIERARCHICAL_UNARY_ACCEPT_BETA,
        HIERARCHICAL_UNARY_HEART_BETA, HIERARCHICAL_UNARY_REJECT_BETA,
        HIERARCHICAL_VIBE_PRIOR_VARIANCE, HierarchicalAssetQualityCacheV1,
        HierarchicalSessionQualityCacheV1, HierarchicalSubjectQualityCacheV1,
        LEGACY_EXACT_OFFSET_EPSILON, LEGACY_L2_FRONTIER, LEGACY_L2_OFFSET, LEGACY_L2_SESSION_HEAD,
        LEGACY_LR_DUEL_ALPHA, LEGACY_LR_DUEL_COORD, LEGACY_LR_DUEL_HEAD, LEGACY_LR_DUEL_MOOD,
        LEGACY_LR_PROJECTION, LEGACY_PROJECTION_WEIGHT_DECAY, LegacyUnaryFeedback,
        PerturbativeHyperParamsV3, QualityFormalVersion, QualityModelRecord, QualityReplayStats,
        SessionQualityCachePayload, StoredAssetQualityCache, StoredExternalItemQualityCache,
        StoredSessionQualityCache, StoredSubjectQualityCache, SubjectQualityCachePayload,
        decode_quality_payload, diagonal_adf_update, encode_quality_payload,
        gaussian_duel_moment_match, hierarchical_face_backflow_coeff,
        hierarchical_face_latent_mean, hierarchical_face_latent_variance,
        legacy_asset_quality_payload, legacy_batter_asset, legacy_heart_bias,
        legacy_projection_prior, legacy_session_quality_payload, legacy_shove_mood,
        legacy_subject_quality_payload, perturbative_hyper_artifact_key,
        technical_prior_artifact_key,
    },
    quality_features::{
        AssetQualityFeatures, LinearTechnicalPriorHead, QUALITY_FEATURE_REVISION,
        StoredAssetQualityFeatures, TechnicalPriorSample, VIBE_DESCRIPTOR_DIM,
        fit_linear_technical_prior_head, technical_prior_mean_with_head,
        technical_prior_variance_with_head,
    },
};

#[derive(Debug, Clone)]
struct LegacyReplayComparisonEvent {
    id: i64,
    created_at: i64,
    session_id: SessionId,
    left_asset_id: AssetId,
    right_asset_id: AssetId,
    winner_asset_id: AssetId,
}

#[derive(Debug, Clone)]
struct LegacyReplayNudgeEvent {
    id: i64,
    created_at: i64,
    session_id: SessionId,
    asset_id: AssetId,
    direction: f32,
}

#[derive(Debug, Clone)]
struct LegacyReplayHeartEvent {
    id: i64,
    created_at: i64,
    session_id: SessionId,
    asset_id: AssetId,
    active: bool,
}

#[derive(Debug, Clone, Copy)]
enum ExternalUnaryFeedback {
    Reject,
    Accept,
    Heart,
}

impl ExternalUnaryFeedback {
    const fn outcome(self) -> f32 {
        match self {
            Self::Reject => -1.0,
            Self::Accept | Self::Heart => 1.0,
        }
    }

    const fn beta(self) -> f32 {
        match self {
            Self::Reject => HIERARCHICAL_UNARY_REJECT_BETA,
            Self::Accept => HIERARCHICAL_UNARY_ACCEPT_BETA,
            Self::Heart => HIERARCHICAL_UNARY_HEART_BETA,
        }
    }
}

#[derive(Debug, Clone)]
struct HierarchicalReplayExternalEvent {
    id: i64,
    created_at: i64,
    session_id: SessionId,
    item_id: RemoteItemId,
    local_asset_id: Option<AssetId>,
    kind: ExternalEventKind,
}

#[derive(Debug, Clone)]
enum LegacyReplayEvent {
    Comparison(LegacyReplayComparisonEvent),
    Nudge(LegacyReplayNudgeEvent),
    Heart(LegacyReplayHeartEvent),
    External(HierarchicalReplayExternalEvent),
}

impl LegacyReplayEvent {
    fn sort_key(&self) -> (i64, u8, i64) {
        match self {
            Self::Comparison(event) => (event.created_at, 0, event.id),
            Self::Nudge(event) => (event.created_at, 1, event.id),
            Self::Heart(event) => (event.created_at, 2, event.id),
            Self::External(event) => (event.created_at, 3, event.id),
        }
    }
}

#[derive(Debug, Clone)]
struct LegacyReplaySessionState {
    session: SessionRecord,
    exact_offsets: HashMap<AssetId, f32>,
    hearted_assets: HashSet<AssetId>,
    embedding_head: Option<SessionEmbeddingHead>,
}

#[derive(Debug, Clone)]
struct LegacyReplayState {
    projection_model_name: String,
    projection: Option<ProjectionModel>,
    embeddings: HashMap<AssetId, Vec<f32>>,
    assets: HashMap<AssetId, AssetRecord>,
    sessions: HashMap<SessionId, LegacyReplaySessionState>,
    comparison_events: usize,
    nudge_events: usize,
    heart_events: usize,
    max_comparison_id: i64,
    max_nudge_id: i64,
    max_heart_id: i64,
}

#[derive(Debug, Clone, Copy)]
struct SubjectBeautyAnchor {
    identity_id: FaceIdentityId,
    mean: f32,
    variance: f32,
}

#[derive(Debug, Clone, Copy)]
struct HierarchicalSubjectState {
    beauty: FaceBeauty,
    duel_count: u32,
}

#[derive(Debug, Clone)]
struct HierarchicalReplayAssetState {
    asset: AssetRecord,
    baseline_mean: f32,
    baseline_variance: f32,
    mood_loading_mean: [f32; crate::model::LATENT_DIM],
    mood_loading_variance: [f32; crate::model::LATENT_DIM],
    technical_mean: Option<f32>,
    technical_variance: Option<f32>,
    vibe_mean: [f32; VIBE_DESCRIPTOR_DIM],
    vibe_variance: [f32; VIBE_DESCRIPTOR_DIM],
    face: Option<SubjectBeautyAnchor>,
}

#[derive(Debug, Clone, Copy)]
struct HierarchicalReplayExternalState {
    baseline_mean: f32,
    baseline_variance: f32,
    mood_loading_mean: [f32; crate::model::LATENT_DIM],
    mood_loading_variance: [f32; crate::model::LATENT_DIM],
    technical_mean: Option<f32>,
    technical_variance: Option<f32>,
    vibe_mean: [f32; VIBE_DESCRIPTOR_DIM],
    vibe_variance: [f32; VIBE_DESCRIPTOR_DIM],
}

#[derive(Debug, Clone)]
struct HierarchicalReplaySessionState {
    session: SessionRecord,
    semantic_mood_variance: [f32; crate::model::LATENT_DIM],
    vibe_mean: [f32; VIBE_DESCRIPTOR_DIM],
    vibe_variance: [f32; VIBE_DESCRIPTOR_DIM],
    frontier_variance: f32,
    exact_offsets: HashMap<AssetId, f32>,
    hearted_assets: HashSet<AssetId>,
    embedding_head: Option<SessionEmbeddingHead>,
}

#[derive(Debug, Clone)]
struct HierarchicalReplayState {
    projection_model_name: String,
    embeddings: HashMap<AssetId, Vec<f32>>,
    assets: HashMap<AssetId, HierarchicalReplayAssetState>,
    external_items: HashMap<RemoteItemId, HierarchicalReplayExternalState>,
    subjects: HashMap<FaceIdentityId, HierarchicalSubjectState>,
    sessions: HashMap<SessionId, HierarchicalReplaySessionState>,
    comparison_events: usize,
    nudge_events: usize,
    heart_events: usize,
    external_events: usize,
    max_comparison_id: i64,
    max_nudge_id: i64,
    max_heart_id: i64,
    max_external_id: i64,
}

#[derive(Debug, Clone)]
struct ReplayEventStream {
    events: Vec<LegacyReplayEvent>,
    comparison_events: usize,
    nudge_events: usize,
    heart_events: usize,
    external_events: usize,
    max_comparison_id: i64,
    max_nudge_id: i64,
    max_heart_id: i64,
    max_external_id: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReplayFrontier {
    max_comparison_id: i64,
    max_nudge_id: i64,
    max_heart_id: i64,
    max_external_id: i64,
}

impl ReplayEventStream {
    fn frontier(&self) -> ReplayFrontier {
        ReplayFrontier {
            max_comparison_id: self.max_comparison_id,
            max_nudge_id: self.max_nudge_id,
            max_heart_id: self.max_heart_id,
            max_external_id: self.max_external_id,
        }
    }
}

impl Store {
    pub fn active_quality_model(&self) -> anyhow::Result<QualityModelRecord> {
        if let Some(model) = self.load_active_quality_model()? {
            return Ok(model);
        }
        let default = QualityModelRecord::runtime_default(now());
        self.set_active_quality_model(&default)?;
        Ok(default)
    }

    pub fn set_active_quality_model(&self, model: &QualityModelRecord) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO quality_model_registry (
                slot,
                formal_version,
                prior_family,
                prior_revision,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(slot) DO UPDATE SET
                formal_version = excluded.formal_version,
                prior_family = excluded.prior_family,
                prior_revision = excluded.prior_revision,
                updated_at = excluded.updated_at
            ",
            params![
                ACTIVE_QUALITY_MODEL_SLOT,
                model.formal_version.as_str(),
                model.prior_family.as_str(),
                model.prior_revision.as_str(),
                model.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn purge_quality_cache_except(
        &self,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<usize> {
        let mut purged = 0usize;
        for table in [
            "quality_asset_cache",
            "quality_session_cache",
            "quality_subject_cache",
            "quality_external_item_cache",
            "quality_replay_cursors",
        ] {
            purged += self.conn.execute(
                &format!("DELETE FROM {table} WHERE formal_version <> ?1"),
                params![formal_version.as_str()],
            )?;
        }
        Ok(purged)
    }

    pub fn asset_quality_features(
        &self,
        asset_id: &AssetId,
        extractor_revision: &str,
    ) -> anyhow::Result<Option<StoredAssetQualityFeatures>> {
        self.conn
            .query_row(
                r"
                SELECT technical_payload, vibe_payload, updated_at
                FROM asset_quality_features
                WHERE asset_id = ?1 AND extractor_revision = ?2
                ",
                params![asset_id.0, extractor_revision],
                |row| {
                    Ok(StoredAssetQualityFeatures {
                        asset_id: asset_id.clone(),
                        extractor_revision: extractor_revision.to_owned(),
                        features: AssetQualityFeatures {
                            technical: decode_quality_payload(&row.get::<_, String>(0)?)
                                .map_err(into_rusqlite)?,
                            vibe: decode_quality_payload(&row.get::<_, String>(1)?)
                                .map_err(into_rusqlite)?,
                        },
                        updated_at: decode_ts(row.get::<_, i64>(2)?).map_err(into_rusqlite)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_asset_quality_features(
        &self,
        stored: &StoredAssetQualityFeatures,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO asset_quality_features (
                asset_id,
                extractor_revision,
                technical_payload,
                vibe_payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(asset_id) DO UPDATE SET
                extractor_revision = excluded.extractor_revision,
                technical_payload = excluded.technical_payload,
                vibe_payload = excluded.vibe_payload,
                updated_at = excluded.updated_at
            ",
            params![
                stored.asset_id.0,
                stored.extractor_revision,
                encode_quality_payload(&stored.features.technical)?,
                encode_quality_payload(&stored.features.vibe)?,
                stored.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn save_asset_quality_features_batch(
        &mut self,
        rows: &[StoredAssetQualityFeatures],
    ) -> anyhow::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("opening asset quality feature batch transaction")?;
        let mut stmt = tx.prepare(
            r"
            INSERT INTO asset_quality_features (
                asset_id,
                extractor_revision,
                technical_payload,
                vibe_payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(asset_id) DO UPDATE SET
                extractor_revision = excluded.extractor_revision,
                technical_payload = excluded.technical_payload,
                vibe_payload = excluded.vibe_payload,
                updated_at = excluded.updated_at
            ",
        )?;
        for stored in rows {
            stmt.execute(params![
                stored.asset_id.0,
                stored.extractor_revision,
                encode_quality_payload(&stored.features.technical)?,
                encode_quality_payload(&stored.features.vibe)?,
                stored.updated_at.unix_timestamp(),
            ])?;
        }
        drop(stmt);
        tx.commit()
            .context("committing asset quality feature batch transaction")?;
        Ok(())
    }

    pub fn corpus_asset_quality_features(
        &self,
        corpus_id: CorpusId,
        extractor_revision: &str,
    ) -> anyhow::Result<HashMap<AssetId, AssetQualityFeatures>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT aqf.asset_id, aqf.technical_payload, aqf.vibe_payload
            FROM asset_quality_features aqf
            JOIN corpus_assets ca
              ON ca.asset_id = aqf.asset_id
             AND ca.corpus_id = ?1
            WHERE aqf.extractor_revision = ?2
            ",
        )?;
        let rows = stmt
            .query_map(params![corpus_id.0, extractor_revision], |row| {
                Ok((
                    AssetId(row.get::<_, String>(0)?),
                    AssetQualityFeatures {
                        technical: decode_quality_payload(&row.get::<_, String>(1)?)
                            .map_err(into_rusqlite)?,
                        vibe: decode_quality_payload(&row.get::<_, String>(2)?)
                            .map_err(into_rusqlite)?,
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.into_iter().collect())
    }

    pub fn assets_missing_quality_features(
        &self,
        corpus_id: CorpusId,
        extractor_revision: &str,
    ) -> anyhow::Result<Vec<AssetRecord>> {
        let present = self.corpus_asset_quality_features(corpus_id, extractor_revision)?;
        Ok(self
            .corpus_assets(corpus_id)?
            .into_iter()
            .filter(|asset| !present.contains_key(&asset.id))
            .collect())
    }

    pub fn rebuild_active_quality_state(
        &mut self,
        projection_model_name: &str,
    ) -> anyhow::Result<QualityReplayStats> {
        let model = self.active_quality_model()?;
        match model.formal_version {
            QualityFormalVersion::LegacyIndependentV1 => {
                self.rebuild_legacy_independent_v1(projection_model_name)
            }
            QualityFormalVersion::HierarchicalGaussianV1 => {
                self.rebuild_hierarchical_gaussian_v1(projection_model_name)
            }
            QualityFormalVersion::HierarchicalPerturbativeV2 => {
                bail!("hierarchical_perturbative_v2 is deprecated")
            }
            QualityFormalVersion::HierarchicalPerturbativeV3 => {
                self.rebuild_hierarchical_perturbative_v3(projection_model_name)
            }
        }
    }

    pub fn asset_quality_cache(
        &self,
        asset_id: &AssetId,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<Option<StoredAssetQualityCache>> {
        self.conn
            .query_row(
                r"
                SELECT payload, updated_at
                FROM quality_asset_cache
                WHERE formal_version = ?1 AND asset_id = ?2
                ",
                params![formal_version.as_str(), asset_id.0],
                |row| {
                    let payload = decode_quality_payload::<AssetQualityCachePayload>(
                        &row.get::<_, String>(0)?,
                    )
                    .map_err(into_rusqlite)?;
                    let updated_at = decode_ts(row.get::<_, i64>(1)?).map_err(into_rusqlite)?;
                    Ok(StoredAssetQualityCache {
                        asset_id: asset_id.clone(),
                        payload,
                        updated_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn corpus_asset_quality_caches(
        &self,
        corpus_id: CorpusId,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<HashMap<AssetId, StoredAssetQualityCache>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT qac.asset_id, qac.payload, qac.updated_at
            FROM quality_asset_cache qac
            JOIN corpus_assets ca
              ON ca.asset_id = qac.asset_id
             AND ca.corpus_id = ?1
            WHERE qac.formal_version = ?2
            ",
        )?;
        let rows = stmt
            .query_map(params![corpus_id.0, formal_version.as_str()], |row| {
                let asset_id = AssetId(row.get::<_, String>(0)?);
                let payload =
                    decode_quality_payload::<AssetQualityCachePayload>(&row.get::<_, String>(1)?)
                        .map_err(into_rusqlite)?;
                let updated_at = decode_ts(row.get::<_, i64>(2)?).map_err(into_rusqlite)?;
                Ok((
                    asset_id.clone(),
                    StoredAssetQualityCache {
                        asset_id,
                        payload,
                        updated_at,
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.into_iter().collect())
    }

    pub fn save_asset_quality_cache(&self, cache: &StoredAssetQualityCache) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO quality_asset_cache (
                formal_version,
                asset_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(formal_version, asset_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            ",
            params![
                cache.payload.formal_version().as_str(),
                cache.asset_id.0,
                encode_quality_payload(&cache.payload)?,
                cache.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn external_item_quality_cache(
        &self,
        item_id: RemoteItemId,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<Option<StoredExternalItemQualityCache>> {
        self.conn
            .query_row(
                r"
                SELECT payload, updated_at
                FROM quality_external_item_cache
                WHERE formal_version = ?1 AND item_id = ?2
                ",
                params![formal_version.as_str(), item_id.0],
                |row| {
                    let payload = decode_quality_payload::<AssetQualityCachePayload>(
                        &row.get::<_, String>(0)?,
                    )
                    .map_err(into_rusqlite)?;
                    let updated_at = decode_ts(row.get::<_, i64>(1)?).map_err(into_rusqlite)?;
                    Ok(StoredExternalItemQualityCache {
                        item_id,
                        payload,
                        updated_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_external_item_quality_cache(
        &self,
        cache: &StoredExternalItemQualityCache,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO quality_external_item_cache (
                formal_version,
                item_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(formal_version, item_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            ",
            params![
                cache.payload.formal_version().as_str(),
                cache.item_id.0,
                encode_quality_payload(&cache.payload)?,
                cache.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn load_linear_technical_prior_head(
        &self,
        model: &QualityModelRecord,
    ) -> anyhow::Result<Option<LinearTechnicalPriorHead>> {
        self.conn
            .query_row(
                r"
                SELECT payload
                FROM quality_prior_artifacts
                WHERE formal_version = ?1
                  AND prior_family = ?2
                  AND prior_revision = ?3
                  AND artifact_key = ?4
                ",
                params![
                    model.formal_version.as_str(),
                    model.prior_family.as_str(),
                    model.prior_revision.as_str(),
                    technical_prior_artifact_key(),
                ],
                |row| decode_quality_payload(&row.get::<_, String>(0)?).map_err(into_rusqlite),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_linear_technical_prior_head(
        &self,
        model: &QualityModelRecord,
        head: &LinearTechnicalPriorHead,
    ) -> anyhow::Result<()> {
        self.conn.execute(
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
                encode_quality_payload(head)?,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn load_perturbative_hyper_params(
        &self,
        model: &QualityModelRecord,
    ) -> anyhow::Result<Option<PerturbativeHyperParamsV3>> {
        self.conn
            .query_row(
                r"
                SELECT payload
                FROM quality_prior_artifacts
                WHERE formal_version = ?1
                  AND prior_family = ?2
                  AND prior_revision = ?3
                  AND artifact_key = ?4
                ",
                params![
                    model.formal_version.as_str(),
                    model.prior_family.as_str(),
                    model.prior_revision.as_str(),
                    perturbative_hyper_artifact_key(),
                ],
                |row| decode_quality_payload(&row.get::<_, String>(0)?).map_err(into_rusqlite),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_perturbative_hyper_params(
        &self,
        model: &QualityModelRecord,
        hyper: &PerturbativeHyperParamsV3,
    ) -> anyhow::Result<()> {
        self.conn.execute(
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
                perturbative_hyper_artifact_key(),
                encode_quality_payload(hyper)?,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn session_quality_cache(
        &self,
        session_id: SessionId,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<Option<StoredSessionQualityCache>> {
        self.conn
            .query_row(
                r"
                SELECT payload, updated_at
                FROM quality_session_cache
                WHERE formal_version = ?1 AND session_id = ?2
                ",
                params![formal_version.as_str(), session_id.0],
                |row| {
                    let payload = decode_quality_payload::<SessionQualityCachePayload>(
                        &row.get::<_, String>(0)?,
                    )
                    .map_err(into_rusqlite)?;
                    let updated_at = decode_ts(row.get::<_, i64>(1)?).map_err(into_rusqlite)?;
                    Ok(StoredSessionQualityCache {
                        session_id,
                        payload,
                        updated_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_session_quality_cache(
        &self,
        cache: &StoredSessionQualityCache,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO quality_session_cache (
                formal_version,
                session_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(formal_version, session_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            ",
            params![
                cache.payload.formal_version().as_str(),
                cache.session_id.0,
                encode_quality_payload(&cache.payload)?,
                cache.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    pub fn subject_quality_cache(
        &self,
        identity_id: FaceIdentityId,
        formal_version: QualityFormalVersion,
    ) -> anyhow::Result<Option<StoredSubjectQualityCache>> {
        self.conn
            .query_row(
                r"
                SELECT payload, updated_at
                FROM quality_subject_cache
                WHERE formal_version = ?1 AND identity_id = ?2
                ",
                params![formal_version.as_str(), identity_id.0],
                |row| {
                    let payload = decode_quality_payload::<SubjectQualityCachePayload>(
                        &row.get::<_, String>(0)?,
                    )
                    .map_err(into_rusqlite)?;
                    let updated_at = decode_ts(row.get::<_, i64>(1)?).map_err(into_rusqlite)?;
                    Ok(StoredSubjectQualityCache {
                        identity_id,
                        payload,
                        updated_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_subject_quality_cache(
        &self,
        cache: &StoredSubjectQualityCache,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO quality_subject_cache (
                formal_version,
                identity_id,
                payload,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(formal_version, identity_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            ",
            params![
                cache.payload.formal_version().as_str(),
                cache.identity_id.0,
                encode_quality_payload(&cache.payload)?,
                cache.updated_at.unix_timestamp(),
            ],
        )?;
        Ok(())
    }

    fn rebuild_legacy_independent_v1(
        &mut self,
        projection_model_name: &str,
    ) -> anyhow::Result<QualityReplayStats> {
        let mut state = self.seed_legacy_replay_state(projection_model_name)?;
        let replay = self.read_replay_events()?;
        state.comparison_events = replay.comparison_events;
        state.nudge_events = replay.nudge_events;
        state.heart_events = replay.heart_events;
        state.max_comparison_id = replay.max_comparison_id;
        state.max_nudge_id = replay.max_nudge_id;
        state.max_heart_id = replay.max_heart_id;
        for event in replay.events {
            self.apply_legacy_replay_event(&mut state, event)?;
        }
        let subject_snapshot = self.compute_identity_beauty_snapshot()?;
        self.persist_legacy_replay_state(&state, &subject_snapshot)?;
        Ok(QualityReplayStats {
            formal_version: QualityFormalVersion::LegacyIndependentV1,
            asset_count: state.assets.len(),
            session_count: state.sessions.len(),
            subject_count: subject_snapshot.len(),
            comparison_events: state.comparison_events,
            nudge_events: state.nudge_events,
            heart_events: state.heart_events,
        })
    }

    fn rebuild_hierarchical_gaussian_v1(
        &mut self,
        projection_model_name: &str,
    ) -> anyhow::Result<QualityReplayStats> {
        let model = self.active_quality_model()?;
        let replay = self.read_replay_events()?;
        let bootstrap_head = self
            .load_linear_technical_prior_head(&model)?
            .unwrap_or_default();
        let mut state =
            self.seed_hierarchical_replay_state(projection_model_name, &bootstrap_head)?;
        self.apply_hierarchical_replay_stream(&mut state, &replay)?;
        let fitted_head =
            fit_linear_technical_prior_head(&self.technical_prior_samples_from_state(&state))
                .unwrap_or(bootstrap_head);
        if fitted_head.weights != bootstrap_head.weights
            || fitted_head.bias != bootstrap_head.bias
            || fitted_head.residual_variance != bootstrap_head.residual_variance
        {
            state = self.seed_hierarchical_replay_state(projection_model_name, &fitted_head)?;
            self.apply_hierarchical_replay_stream(&mut state, &replay)?;
        }
        let subject_snapshot = state
            .subjects
            .iter()
            .map(|(identity_id, subject)| (*identity_id, subject.beauty, subject.duel_count))
            .collect::<Vec<_>>();
        self.persist_hierarchical_replay_state(&state, &subject_snapshot, &model, &fitted_head)?;
        Ok(QualityReplayStats {
            formal_version: QualityFormalVersion::HierarchicalGaussianV1,
            asset_count: state.assets.len(),
            session_count: state.sessions.len(),
            subject_count: subject_snapshot.len(),
            comparison_events: state.comparison_events,
            nudge_events: state.nudge_events,
            heart_events: state.heart_events,
        })
    }

    fn seed_legacy_replay_state(
        &self,
        projection_model_name: &str,
    ) -> anyhow::Result<LegacyReplayState> {
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
            .map(|(id, rotation_quarters)| {
                (
                    id.clone(),
                    AssetRecord {
                        id,
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
                )
            })
            .collect::<HashMap<_, _>>();

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
                (
                    id,
                    LegacyReplaySessionState {
                        session: SessionRecord {
                            id,
                            corpus_id,
                            mood: [0.0; crate::model::LATENT_DIM],
                            frontier: 0.0,
                            comparisons: 0,
                            nudges: 0,
                            hearts: 0,
                        },
                        exact_offsets: HashMap::new(),
                        hearted_assets: HashSet::new(),
                        embedding_head: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

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

        Ok(LegacyReplayState {
            projection_model_name: projection_model_name.to_owned(),
            projection: None,
            embeddings,
            assets,
            sessions,
            comparison_events: 0,
            nudge_events: 0,
            heart_events: 0,
            max_comparison_id: 0,
            max_nudge_id: 0,
            max_heart_id: 0,
        })
    }

    fn apply_hierarchical_replay_stream(
        &self,
        state: &mut HierarchicalReplayState,
        replay: &ReplayEventStream,
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
            self.apply_hierarchical_replay_event(state, event)?;
        }
        Ok(())
    }

    fn seed_hierarchical_replay_state(
        &self,
        projection_model_name: &str,
        technical_head: &LinearTechnicalPriorHead,
    ) -> anyhow::Result<HierarchicalReplayState> {
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
            .query_map(params![QUALITY_FEATURE_REVISION], |row| {
                Ok((
                    AssetId(row.get::<_, String>(0)?),
                    AssetQualityFeatures {
                        technical: decode_quality_payload(&row.get::<_, String>(1)?)
                            .map_err(into_rusqlite)?,
                        vibe: decode_quality_payload(&row.get::<_, String>(2)?)
                            .map_err(into_rusqlite)?,
                    },
                ))
            })?
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
            .query_map(params![QUALITY_FEATURE_REVISION], |row| {
                Ok((
                    RemoteItemId(row.get::<_, i64>(0)?),
                    AssetQualityFeatures {
                        technical: decode_quality_payload(&row.get::<_, String>(1)?)
                            .map_err(into_rusqlite)?,
                        vibe: decode_quality_payload(&row.get::<_, String>(2)?)
                            .map_err(into_rusqlite)?,
                    },
                ))
            })?
            .collect::<Result<HashMap<_, _>, _>>()?;
        let domain_gates =
            self.global_hierarchical_domain_gates(projection_model_name, &embeddings)?;
        let external_domain_gates = self.global_hierarchical_external_domain_gates(
            projection_model_name,
            &external_embeddings,
        )?;
        let semantic_basis = SemanticPriorBasis::fit(&embeddings, &external_embeddings);
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
            .map(|(id, rotation_quarters)| {
                let feature = feature_rows
                    .get(&id)
                    .copied()
                    .unwrap_or_else(AssetQualityFeatures::neutral);
                let gate_3d = domain_gates.get(&id).copied().unwrap_or(false);
                let technical_mean = gate_3d
                    .then(|| technical_prior_mean_with_head(technical_head, &feature.technical));
                let technical_variance = gate_3d.then(|| {
                    technical_prior_variance_with_head(technical_head, &feature.technical)
                        .max(HIERARCHICAL_TECH_PRIOR_FLOOR)
                });
                let mood_loading_mean = semantic_basis
                    .as_ref()
                    .and_then(|basis| {
                        embeddings
                            .get(&id)
                            .map(|embedding| basis.project(embedding))
                    })
                    .unwrap_or([0.0; crate::model::LATENT_DIM]);
                let vibe_mean = if gate_3d {
                    feature.vibe
                } else {
                    [0.0; VIBE_DESCRIPTOR_DIM]
                };
                let vibe_variance = if gate_3d {
                    [HIERARCHICAL_VIBE_PRIOR_VARIANCE; VIBE_DESCRIPTOR_DIM]
                } else {
                    [HIERARCHICAL_MIN_VARIANCE; VIBE_DESCRIPTOR_DIM]
                };
                let mut state = HierarchicalReplayAssetState {
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
                    baseline_mean: 0.0,
                    baseline_variance: HIERARCHICAL_BASELINE_PRIOR_VARIANCE,
                    mood_loading_mean,
                    mood_loading_variance: [HIERARCHICAL_LOADING_PRIOR_VARIANCE;
                        crate::model::LATENT_DIM],
                    technical_mean,
                    technical_variance,
                    vibe_mean,
                    vibe_variance,
                    face: dominant_faces
                        .get(&id)
                        .copied()
                        .map(|face| face_anchor_for_subject(face, &subjects)),
                };
                sync_hierarchical_asset_record(&mut state);
                (id, state)
            })
            .collect::<HashMap<_, _>>();
        let external_items = external_embeddings
            .keys()
            .copied()
            .map(|item_id| {
                let feature = external_feature_rows
                    .get(&item_id)
                    .copied()
                    .unwrap_or_else(AssetQualityFeatures::neutral);
                let gate_3d = external_domain_gates
                    .get(&item_id)
                    .copied()
                    .unwrap_or(false);
                let technical_mean = gate_3d
                    .then(|| technical_prior_mean_with_head(technical_head, &feature.technical));
                let technical_variance = gate_3d.then(|| {
                    technical_prior_variance_with_head(technical_head, &feature.technical)
                        .max(HIERARCHICAL_TECH_PRIOR_FLOOR)
                });
                let mood_loading_mean = semantic_basis
                    .as_ref()
                    .and_then(|basis| {
                        external_embeddings
                            .get(&item_id)
                            .map(|embedding| basis.project(embedding))
                    })
                    .unwrap_or([0.0; crate::model::LATENT_DIM]);
                let vibe_mean = if gate_3d {
                    feature.vibe
                } else {
                    [0.0; VIBE_DESCRIPTOR_DIM]
                };
                let vibe_variance = if gate_3d {
                    [HIERARCHICAL_VIBE_PRIOR_VARIANCE; VIBE_DESCRIPTOR_DIM]
                } else {
                    [HIERARCHICAL_MIN_VARIANCE; VIBE_DESCRIPTOR_DIM]
                };
                (
                    item_id,
                    HierarchicalReplayExternalState {
                        baseline_mean: 0.0,
                        baseline_variance: HIERARCHICAL_BASELINE_PRIOR_VARIANCE,
                        mood_loading_mean,
                        mood_loading_variance: [HIERARCHICAL_LOADING_PRIOR_VARIANCE;
                            crate::model::LATENT_DIM],
                        technical_mean,
                        technical_variance,
                        vibe_mean,
                        vibe_variance,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

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
                (
                    id,
                    HierarchicalReplaySessionState {
                        session: SessionRecord {
                            id,
                            corpus_id,
                            mood: [0.0; crate::model::LATENT_DIM],
                            frontier: 0.0,
                            comparisons: 0,
                            nudges: 0,
                            hearts: 0,
                        },
                        semantic_mood_variance: [HIERARCHICAL_SESSION_MOOD_PRIOR_VARIANCE;
                            crate::model::LATENT_DIM],
                        vibe_mean: [0.0; VIBE_DESCRIPTOR_DIM],
                        vibe_variance: [HIERARCHICAL_SESSION_VIBE_PRIOR_VARIANCE;
                            VIBE_DESCRIPTOR_DIM],
                        frontier_variance: HIERARCHICAL_FRONTIER_PRIOR_VARIANCE,
                        exact_offsets: HashMap::new(),
                        hearted_assets: HashSet::new(),
                        embedding_head: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        Ok(HierarchicalReplayState {
            projection_model_name: projection_model_name.to_owned(),
            embeddings,
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

    fn read_replay_events(&self) -> anyhow::Result<ReplayEventStream> {
        let comparisons = self
            .conn
            .prepare(
                r"
                SELECT id, created_at, session_id, left_asset_id, right_asset_id, winner_asset_id
                FROM comparisons
                ORDER BY created_at ASC, id ASC
                ",
            )?
            .query_map([], |row| {
                Ok(LegacyReplayEvent::Comparison(LegacyReplayComparisonEvent {
                    id: row.get::<_, i64>(0)?,
                    created_at: row.get::<_, i64>(1)?,
                    session_id: SessionId(row.get::<_, i64>(2)?),
                    left_asset_id: AssetId(row.get::<_, String>(3)?),
                    right_asset_id: AssetId(row.get::<_, String>(4)?),
                    winner_asset_id: AssetId(row.get::<_, String>(5)?),
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let comparison_events = comparisons.len();
        let max_comparison_id = comparisons
            .iter()
            .filter_map(|event| match event {
                LegacyReplayEvent::Comparison(event) => Some(event.id),
                _ => None,
            })
            .max()
            .unwrap_or_default();

        let nudges = self
            .conn
            .prepare(
                r"
                SELECT id, created_at, session_id, asset_id, direction
                FROM nudge_events
                ORDER BY created_at ASC, id ASC
                ",
            )?
            .query_map([], |row| {
                Ok(LegacyReplayEvent::Nudge(LegacyReplayNudgeEvent {
                    id: row.get::<_, i64>(0)?,
                    created_at: row.get::<_, i64>(1)?,
                    session_id: SessionId(row.get::<_, i64>(2)?),
                    asset_id: AssetId(row.get::<_, String>(3)?),
                    direction: row.get::<_, f32>(4)?,
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let nudge_events = nudges.len();
        let max_nudge_id = nudges
            .iter()
            .filter_map(|event| match event {
                LegacyReplayEvent::Nudge(event) => Some(event.id),
                _ => None,
            })
            .max()
            .unwrap_or_default();

        let hearts = self
            .conn
            .prepare(
                r"
                SELECT id, created_at, session_id, asset_id, active
                FROM heart_events
                ORDER BY created_at ASC, id ASC
                ",
            )?
            .query_map([], |row| {
                Ok(LegacyReplayEvent::Heart(LegacyReplayHeartEvent {
                    id: row.get::<_, i64>(0)?,
                    created_at: row.get::<_, i64>(1)?,
                    session_id: SessionId(row.get::<_, i64>(2)?),
                    asset_id: AssetId(row.get::<_, String>(3)?),
                    active: row.get::<_, i64>(4)? != 0,
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let heart_events = hearts.len();
        let max_heart_id = hearts
            .iter()
            .filter_map(|event| match event {
                LegacyReplayEvent::Heart(event) => Some(event.id),
                _ => None,
            })
            .max()
            .unwrap_or_default();

        let external = self
            .conn
            .prepare(
                r"
                SELECT id, created_at, session_id, item_id, local_asset_id, event_kind
                FROM external_events
                WHERE event_kind IN ('rejected', 'kept', 'hearted', 'local_win', 'remote_win')
                ORDER BY created_at ASC, id ASC
                ",
            )?
            .query_map([], |row| {
                let raw_kind = row.get::<_, String>(5)?;
                let kind = match raw_kind.as_str() {
                    "rejected" => ExternalEventKind::Rejected,
                    "kept" => ExternalEventKind::Kept,
                    "hearted" => ExternalEventKind::Hearted,
                    "local_win" => ExternalEventKind::LocalWin,
                    "remote_win" => ExternalEventKind::RemoteWin,
                    _ => {
                        return Err(rusqlite::Error::FromSqlConversionFailure(
                            5,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::other(format!(
                                "unsupported external replay event kind `{raw_kind}`"
                            ))),
                        ));
                    }
                };
                Ok(LegacyReplayEvent::External(
                    HierarchicalReplayExternalEvent {
                        id: row.get::<_, i64>(0)?,
                        created_at: row.get::<_, i64>(1)?,
                        session_id: SessionId(row.get::<_, i64>(2)?),
                        item_id: RemoteItemId(row.get::<_, i64>(3)?),
                        local_asset_id: row.get::<_, Option<String>>(4)?.map(AssetId),
                        kind,
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let external_events = external.len();
        let max_external_id = external
            .iter()
            .filter_map(|event| match event {
                LegacyReplayEvent::External(event) => Some(event.id),
                _ => None,
            })
            .max()
            .unwrap_or_default();

        let mut events = comparisons;
        events.extend(nudges);
        events.extend(hearts);
        events.extend(external);
        events.sort_by_key(LegacyReplayEvent::sort_key);
        Ok(ReplayEventStream {
            events,
            comparison_events,
            nudge_events,
            heart_events,
            external_events,
            max_comparison_id,
            max_nudge_id,
            max_heart_id,
            max_external_id,
        })
    }

    pub(super) fn current_replay_frontier(&self) -> anyhow::Result<ReplayFrontier> {
        let max_comparison_id =
            self.conn
                .query_row("SELECT COALESCE(MAX(id), 0) FROM comparisons", [], |row| {
                    row.get(0)
                })?;
        let max_nudge_id =
            self.conn
                .query_row("SELECT COALESCE(MAX(id), 0) FROM nudge_events", [], |row| {
                    row.get(0)
                })?;
        let max_heart_id =
            self.conn
                .query_row("SELECT COALESCE(MAX(id), 0) FROM heart_events", [], |row| {
                    row.get(0)
                })?;
        let max_external_id = self.conn.query_row(
            r"
            SELECT COALESCE(MAX(id), 0)
            FROM external_events
            WHERE event_kind IN ('rejected', 'kept', 'hearted', 'local_win', 'remote_win')
            ",
            [],
            |row| row.get(0),
        )?;
        Ok(ReplayFrontier {
            max_comparison_id,
            max_nudge_id,
            max_heart_id,
            max_external_id,
        })
    }

    fn apply_legacy_replay_event(
        &self,
        state: &mut LegacyReplayState,
        event: LegacyReplayEvent,
    ) -> anyhow::Result<()> {
        match event {
            LegacyReplayEvent::Comparison(event) => {
                self.apply_legacy_replay_comparison(state, event)
            }
            LegacyReplayEvent::Nudge(event) => self.apply_legacy_replay_nudge(state, event),
            LegacyReplayEvent::Heart(event) => self.apply_legacy_replay_heart(state, event),
            LegacyReplayEvent::External(_) => Ok(()),
        }
    }

    fn apply_hierarchical_replay_event(
        &self,
        state: &mut HierarchicalReplayState,
        event: LegacyReplayEvent,
    ) -> anyhow::Result<()> {
        match event {
            LegacyReplayEvent::Comparison(event) => {
                self.apply_hierarchical_replay_comparison(state, event)
            }
            LegacyReplayEvent::Nudge(event) => self.apply_hierarchical_replay_nudge(state, event),
            LegacyReplayEvent::Heart(event) => self.apply_hierarchical_replay_heart(state, event),
            LegacyReplayEvent::External(event) => {
                self.apply_hierarchical_replay_external(state, event)
            }
        }
    }

    fn apply_legacy_replay_comparison(
        &self,
        state: &mut LegacyReplayState,
        event: LegacyReplayComparisonEvent,
    ) -> anyhow::Result<()> {
        let projection_dim = state
            .embeddings
            .get(&event.left_asset_id)
            .or_else(|| state.embeddings.get(&event.right_asset_id))
            .map(Vec::len);
        ensure_projection_dim(state, projection_dim);
        let left_embedding = state
            .embeddings
            .get(&event.left_asset_id)
            .map(Vec::as_slice);
        let right_embedding = state
            .embeddings
            .get(&event.right_asset_id)
            .map(Vec::as_slice);
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing replay session {}", event.session_id.0))?;
        if event.left_asset_id == event.right_asset_id {
            bail!(
                "legacy replay encountered self duel for {}",
                event.left_asset_id.0
            );
        }
        let mut left = state
            .assets
            .remove(&event.left_asset_id)
            .with_context(|| format!("missing replay asset {}", event.left_asset_id.0))?;
        let mut right = state
            .assets
            .remove(&event.right_asset_id)
            .with_context(|| format!("missing replay asset {}", event.right_asset_id.0))?;
        let left_utility =
            replay_session_utility(&left, session, left_embedding, &event.left_asset_id);
        let right_utility =
            replay_session_utility(&right, session, right_embedding, &event.right_asset_id);
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
        let y = if left_won { 1.0 } else { 0.0 };
        let err = y - sigmoid(left_utility - right_utility);
        let coord_gap = subtract(&left.coords, &right.coords);
        let left_prior = legacy_projection_prior(state.projection.as_ref(), left_embedding);
        let right_prior = legacy_projection_prior(state.projection.as_ref(), right_embedding);

        legacy_batter_asset(
            &mut left.alpha,
            &mut left.coords,
            &session.session.mood,
            &left_prior,
            err,
            LEGACY_LR_DUEL_ALPHA,
            LEGACY_LR_DUEL_COORD,
        );
        legacy_batter_asset(
            &mut right.alpha,
            &mut right.coords,
            &session.session.mood,
            &right_prior,
            -err,
            LEGACY_LR_DUEL_ALPHA,
            LEGACY_LR_DUEL_COORD,
        );
        for (mood, gap) in session
            .session
            .mood
            .iter_mut()
            .zip(coord_gap.iter().copied())
        {
            *mood += LEGACY_LR_DUEL_MOOD * (err * gap - crate::quality::LEGACY_L2_MOOD * *mood);
        }

        left.compare_count += 1;
        right.compare_count += 1;
        if left_won {
            left.win_count += 1;
        } else {
            right.win_count += 1;
        }
        session.session.comparisons += 1;

        let mut embedding_head = session.embedding_head.take().or_else(|| {
            left_embedding.or(right_embedding).map(|vector| {
                SessionEmbeddingHead::zero(state.projection_model_name.clone(), vector.len())
            })
        });
        if let (Some(head), Some(lhs), Some(rhs)) =
            (&mut embedding_head, left_embedding, right_embedding)
        {
            head.contrast_step(lhs, rhs, err, LEGACY_LR_DUEL_HEAD, LEGACY_L2_SESSION_HEAD);
        }
        session.embedding_head = embedding_head;

        if let Some(model) = &mut state.projection {
            if let Some(embedding) = left_embedding {
                model.gradient_step(
                    embedding,
                    &left.coords,
                    LEGACY_LR_PROJECTION,
                    LEGACY_PROJECTION_WEIGHT_DECAY,
                );
            }
            if let Some(embedding) = right_embedding {
                model.gradient_step(
                    embedding,
                    &right.coords,
                    LEGACY_LR_PROJECTION,
                    LEGACY_PROJECTION_WEIGHT_DECAY,
                );
            }
        }
        state.assets.insert(left.id.clone(), left);
        state.assets.insert(right.id.clone(), right);

        Ok(())
    }

    fn apply_legacy_replay_nudge(
        &self,
        state: &mut LegacyReplayState,
        event: LegacyReplayNudgeEvent,
    ) -> anyhow::Result<()> {
        let projection_dim = state.embeddings.get(&event.asset_id).map(Vec::len);
        ensure_projection_dim(state, projection_dim);
        let embedding = state.embeddings.get(&event.asset_id).map(Vec::as_slice);
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing replay session {}", event.session_id.0))?;
        let asset = state
            .assets
            .get_mut(&event.asset_id)
            .with_context(|| format!("missing replay asset {}", event.asset_id.0))?;
        let feedback = LegacyUnaryFeedback::from_direction(event.direction)?;
        let utility_before = replay_session_utility(asset, session, embedding, &event.asset_id);
        let frontier_before = session.session.frontier;
        let tuning = feedback.tuning(utility_before, frontier_before);
        let prior = legacy_projection_prior(state.projection.as_ref(), embedding);
        let asset_before_coords = asset.coords;

        legacy_batter_asset(
            &mut asset.alpha,
            &mut asset.coords,
            &session.session.mood,
            &prior,
            tuning.signal,
            tuning.alpha_rate,
            tuning.coord_rate,
        );
        legacy_shove_mood(
            &mut session.session.mood,
            &asset_before_coords,
            tuning.signal,
            tuning.mood_rate,
        );
        session.session.frontier +=
            tuning.frontier_rate * (-tuning.signal - LEGACY_L2_FRONTIER * session.session.frontier);

        let exact_offset = session
            .exact_offsets
            .get(&event.asset_id)
            .copied()
            .unwrap_or_default();
        let next_offset =
            exact_offset + tuning.offset_rate * (tuning.signal - LEGACY_L2_OFFSET * exact_offset);

        let mut embedding_head = session.embedding_head.take().or_else(|| {
            embedding.map(|vector| {
                SessionEmbeddingHead::zero(state.projection_model_name.clone(), vector.len())
            })
        });
        if let (Some(head), Some(vector)) = (&mut embedding_head, embedding) {
            head.unary_step(
                vector,
                tuning.signal,
                tuning.head_rate,
                LEGACY_L2_SESSION_HEAD,
            );
        }
        session.embedding_head = embedding_head;
        if let (Some(model), Some(vector)) = (&mut state.projection, embedding) {
            model.gradient_step(
                vector,
                &asset.coords,
                tuning.projection_rate,
                LEGACY_PROJECTION_WEIGHT_DECAY,
            );
        }

        session.session.nudges += 1;
        if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
            session.exact_offsets.remove(&event.asset_id);
        } else {
            session.exact_offsets.insert(event.asset_id, next_offset);
        }
        Ok(())
    }

    fn apply_legacy_replay_heart(
        &self,
        state: &mut LegacyReplayState,
        event: LegacyReplayHeartEvent,
    ) -> anyhow::Result<()> {
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing replay session {}", event.session_id.0))?;
        let asset = state
            .assets
            .get_mut(&event.asset_id)
            .with_context(|| format!("missing replay asset {}", event.asset_id.0))?;
        if !event.active || asset.is_hearted {
            return Ok(());
        }
        asset.is_hearted = true;
        asset.heart_count = asset.heart_count.saturating_add(1);
        session.session.hearts = session.session.hearts.saturating_add(1);
        session.hearted_assets.insert(event.asset_id);
        Ok(())
    }

    fn apply_hierarchical_replay_comparison(
        &self,
        state: &mut HierarchicalReplayState,
        event: LegacyReplayComparisonEvent,
    ) -> anyhow::Result<()> {
        let left_embedding = state
            .embeddings
            .get(&event.left_asset_id)
            .map(Vec::as_slice);
        let right_embedding = state
            .embeddings
            .get(&event.right_asset_id)
            .map(Vec::as_slice);
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing hierarchical session {}", event.session_id.0))?;
        if event.left_asset_id == event.right_asset_id {
            bail!(
                "hierarchical replay encountered self duel for {}",
                event.left_asset_id.0
            );
        }
        let mut left = state
            .assets
            .remove(&event.left_asset_id)
            .with_context(|| format!("missing hierarchical asset {}", event.left_asset_id.0))?;
        let mut right = state
            .assets
            .remove(&event.right_asset_id)
            .with_context(|| format!("missing hierarchical asset {}", event.right_asset_id.0))?;
        refresh_asset_face_anchor(&mut left, &state.subjects);
        refresh_asset_face_anchor(&mut right, &state.subjects);
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
        let left_mean = hierarchical_session_utility_mean(&left, session, &event.left_asset_id);
        let right_mean = hierarchical_session_utility_mean(&right, session, &event.right_asset_id);
        let delta_mean = left_mean - right_mean;
        let delta_variance = hierarchical_session_utility_variance(&left, session)
            + hierarchical_session_utility_variance(&right, session);
        let Some(moments) =
            gaussian_duel_moment_match(delta_mean, delta_variance, outcome, HIERARCHICAL_DUEL_BETA)
        else {
            state.assets.insert(left.asset.id.clone(), left);
            state.assets.insert(right.asset.id.clone(), right);
            return Ok(());
        };

        diagonal_adf_update(
            &mut left.baseline_mean,
            &mut left.baseline_variance,
            1.0,
            outcome,
            moments,
        );
        diagonal_adf_update(
            &mut right.baseline_mean,
            &mut right.baseline_variance,
            -1.0,
            outcome,
            moments,
        );
        for axis in 0..crate::model::LATENT_DIM {
            diagonal_adf_update(
                &mut left.mood_loading_mean[axis],
                &mut left.mood_loading_variance[axis],
                session.session.mood[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut right.mood_loading_mean[axis],
                &mut right.mood_loading_variance[axis],
                -session.session.mood[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut session.session.mood[axis],
                &mut session.semantic_mood_variance[axis],
                left.mood_loading_mean[axis] - right.mood_loading_mean[axis],
                outcome,
                moments,
            );
        }
        if let (Some(mean), Some(variance)) =
            (&mut left.technical_mean, &mut left.technical_variance)
        {
            diagonal_adf_update(mean, variance, HIERARCHICAL_TECH_WEIGHT, outcome, moments);
        }
        if let (Some(mean), Some(variance)) =
            (&mut right.technical_mean, &mut right.technical_variance)
        {
            diagonal_adf_update(mean, variance, -HIERARCHICAL_TECH_WEIGHT, outcome, moments);
        }
        match (
            left.face.map(|face| face.identity_id),
            right.face.map(|face| face.identity_id),
        ) {
            (Some(left_id), Some(right_id)) if left_id != right_id => {
                if let Some(subject) = state.subjects.get_mut(&left_id) {
                    let mut variance = subject.beauty.sigma.powi(2).max(HIERARCHICAL_MIN_VARIANCE);
                    diagonal_adf_update(
                        &mut subject.beauty.mean,
                        &mut variance,
                        hierarchical_face_backflow_coeff(),
                        outcome,
                        moments,
                    );
                    subject.beauty.sigma = variance.max(HIERARCHICAL_MIN_VARIANCE).sqrt();
                }
                if let Some(subject) = state.subjects.get_mut(&right_id) {
                    let mut variance = subject.beauty.sigma.powi(2).max(HIERARCHICAL_MIN_VARIANCE);
                    diagonal_adf_update(
                        &mut subject.beauty.mean,
                        &mut variance,
                        -hierarchical_face_backflow_coeff(),
                        outcome,
                        moments,
                    );
                    subject.beauty.sigma = variance.max(HIERARCHICAL_MIN_VARIANCE).sqrt();
                }
            }
            (Some(identity_id), None) | (None, Some(identity_id)) => {
                let coefficient = if left.face.map(|face| face.identity_id) == Some(identity_id) {
                    hierarchical_face_backflow_coeff()
                } else {
                    -hierarchical_face_backflow_coeff()
                };
                if let Some(subject) = state.subjects.get_mut(&identity_id) {
                    let mut variance = subject.beauty.sigma.powi(2).max(HIERARCHICAL_MIN_VARIANCE);
                    diagonal_adf_update(
                        &mut subject.beauty.mean,
                        &mut variance,
                        coefficient,
                        outcome,
                        moments,
                    );
                    subject.beauty.sigma = variance.max(HIERARCHICAL_MIN_VARIANCE).sqrt();
                }
            }
            _ => {}
        }
        for axis in 0..VIBE_DESCRIPTOR_DIM {
            diagonal_adf_update(
                &mut left.vibe_mean[axis],
                &mut left.vibe_variance[axis],
                session.vibe_mean[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut right.vibe_mean[axis],
                &mut right.vibe_variance[axis],
                -session.vibe_mean[axis],
                outcome,
                moments,
            );
            diagonal_adf_update(
                &mut session.vibe_mean[axis],
                &mut session.vibe_variance[axis],
                left.vibe_mean[axis] - right.vibe_mean[axis],
                outcome,
                moments,
            );
        }

        left.asset.compare_count += 1;
        right.asset.compare_count += 1;
        if left_won {
            left.asset.win_count += 1;
        } else {
            right.asset.win_count += 1;
        }
        session.session.comparisons += 1;
        refresh_asset_face_anchor(&mut left, &state.subjects);
        refresh_asset_face_anchor(&mut right, &state.subjects);
        let mut embedding_head = session.embedding_head.take().or_else(|| {
            left_embedding.or(right_embedding).map(|vector| {
                SessionEmbeddingHead::zero(state.projection_model_name.clone(), vector.len())
            })
        });
        let logistic_err = if left_won { 1.0 } else { 0.0 } - sigmoid(delta_mean);
        if let (Some(head), Some(lhs), Some(rhs)) =
            (&mut embedding_head, left_embedding, right_embedding)
        {
            head.contrast_step(
                lhs,
                rhs,
                logistic_err,
                LEGACY_LR_DUEL_HEAD,
                LEGACY_L2_SESSION_HEAD,
            );
        }
        session.embedding_head = embedding_head;

        sync_hierarchical_asset_record(&mut left);
        sync_hierarchical_asset_record(&mut right);
        state.assets.insert(left.asset.id.clone(), left);
        state.assets.insert(right.asset.id.clone(), right);
        Ok(())
    }

    fn apply_hierarchical_replay_nudge(
        &self,
        state: &mut HierarchicalReplayState,
        event: LegacyReplayNudgeEvent,
    ) -> anyhow::Result<()> {
        let embedding = state.embeddings.get(&event.asset_id).map(Vec::as_slice);
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing hierarchical session {}", event.session_id.0))?;
        let asset = state
            .assets
            .get(&event.asset_id)
            .with_context(|| format!("missing hierarchical asset {}", event.asset_id.0))?;
        let feedback = LegacyUnaryFeedback::from_direction(event.direction)?;
        let utility_before = hierarchical_session_utility_mean(asset, session, &event.asset_id);
        let frontier_before = session.session.frontier;
        let tuning = feedback.tuning(utility_before, frontier_before);
        let exact_offset = session
            .exact_offsets
            .get(&event.asset_id)
            .copied()
            .unwrap_or_default();
        let next_offset =
            exact_offset + tuning.offset_rate * (tuning.signal - LEGACY_L2_OFFSET * exact_offset);
        session.session.frontier +=
            tuning.frontier_rate * (-tuning.signal - LEGACY_L2_FRONTIER * session.session.frontier);
        if next_offset.abs() < LEGACY_EXACT_OFFSET_EPSILON {
            session.exact_offsets.remove(&event.asset_id);
        } else {
            session.exact_offsets.insert(event.asset_id, next_offset);
        }
        let mut embedding_head = session.embedding_head.take().or_else(|| {
            embedding.map(|vector| {
                SessionEmbeddingHead::zero(state.projection_model_name.clone(), vector.len())
            })
        });
        if let (Some(head), Some(vector)) = (&mut embedding_head, embedding) {
            head.unary_step(
                vector,
                tuning.signal,
                tuning.head_rate,
                LEGACY_L2_SESSION_HEAD,
            );
        }
        session.embedding_head = embedding_head;
        session.session.nudges += 1;
        Ok(())
    }

    fn apply_hierarchical_replay_heart(
        &self,
        state: &mut HierarchicalReplayState,
        event: LegacyReplayHeartEvent,
    ) -> anyhow::Result<()> {
        let session = state
            .sessions
            .get_mut(&event.session_id)
            .with_context(|| format!("missing hierarchical session {}", event.session_id.0))?;
        let asset = state
            .assets
            .get_mut(&event.asset_id)
            .with_context(|| format!("missing hierarchical asset {}", event.asset_id.0))?;
        if !event.active || asset.asset.is_hearted {
            return Ok(());
        }
        asset.asset.is_hearted = true;
        asset.asset.heart_count = asset.asset.heart_count.saturating_add(1);
        session.session.hearts = session.session.hearts.saturating_add(1);
        session.hearted_assets.insert(event.asset_id);
        sync_hierarchical_asset_record(asset);
        Ok(())
    }
}

fn insert_quality_replay_cursor(
    tx: &rusqlite::Transaction<'_>,
    formal_version: QualityFormalVersion,
    cursor_key: &str,
    cursor_value: i64,
    updated_at: i64,
) -> anyhow::Result<()> {
    tx.execute(
        r"
        INSERT INTO quality_replay_cursors (
            formal_version,
            cursor_key,
            cursor_value,
            updated_at
        ) VALUES (?1, ?2, ?3, ?4)
        ",
        params![
            formal_version.as_str(),
            cursor_key,
            cursor_value,
            updated_at
        ],
    )?;
    Ok(())
}

fn decode_ts(raw: i64) -> anyhow::Result<OffsetDateTime> {
    OffsetDateTime::from_unix_timestamp(raw).context("decoding unix timestamp")
}

fn into_rusqlite(error: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(error.to_string())),
    )
}
