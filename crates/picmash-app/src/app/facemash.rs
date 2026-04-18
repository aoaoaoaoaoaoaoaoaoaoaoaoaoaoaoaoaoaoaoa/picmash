use super::*;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FacemashStatus {
    pub total_identities: u64,
    pub comparisons: u64,
    pub oracle_trained: bool,
    pub min_face_side: u16,
}

#[derive(Debug, Clone)]
pub struct FacemashLocalAssetView {
    pub id: AssetId,
    pub hearted: bool,
    pub rotation_quarters: i32,
    pub domain: AssetDomainView,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceFrameOverlay {
    pub left_percent: f32,
    pub top_percent: f32,
    pub width_percent: f32,
    pub height_percent: f32,
}

#[derive(Debug, Clone)]
pub struct FacemashFaceView {
    pub face: crate::store::FaceRecord,
    pub asset: FacemashLocalAssetView,
    pub frame: FaceFrameOverlay,
    pub predicted: Option<FacePrediction>,
}

#[derive(Debug, Clone)]
pub struct FacemashPairView {
    pub left: FacemashFaceView,
    pub right: FacemashFaceView,
}

#[derive(Debug, Clone, Copy)]
struct FacemashBelief {
    posterior_mean: f32,
    posterior_sigma: f32,
    model_mean: Option<f32>,
    model_sigma: Option<f32>,
}

impl FacemashBelief {
    fn forge(identity: &FaceIdentityRecord, prediction: Option<FacePrediction>) -> Self {
        Self {
            posterior_mean: identity.beauty.mean,
            posterior_sigma: identity.beauty.sigma,
            model_mean: prediction.map(|prediction| prediction.mean),
            model_sigma: prediction.map(|prediction| prediction.sigma),
        }
    }

    fn prettiness_signal(self) -> f32 {
        self.model_mean.unwrap_or(self.posterior_mean)
    }

    fn model_uncertainty(self) -> f32 {
        self.model_sigma.unwrap_or(0.0)
    }

    fn scarcity(self, identity: &FaceIdentityRecord) -> f32 {
        1.0 / (1.0 + identity.duel_count as f32).sqrt()
    }

    fn topness_bias(topness: f32) -> f32 {
        0.35 + 0.65 * topness
    }
}

#[derive(Debug, Clone)]
struct FacemashIdentityArena {
    identity: FaceIdentityRecord,
    faces: Vec<crate::store::FaceRecord>,
}

impl FacemashIdentityArena {
    fn representative_face(&self, recent_faces: &[FaceId]) -> Option<crate::store::FaceRecord> {
        self.faces.iter().cloned().max_by(|left, right| {
            let left_rank = AppState::facemash_recent_face_rank(recent_faces, left.id);
            let right_rank = AppState::facemash_recent_face_rank(recent_faces, right.id);
            let left_score = match left_rank {
                Some(rank) => -(rank as i32),
                None => i32::MAX / 4,
            };
            let right_score = match right_rank {
                Some(rank) => -(rank as i32),
                None => i32::MAX / 4,
            };
            left_score
                .cmp(&right_score)
                .then_with(|| left.confidence.total_cmp(&right.confidence))
                .then_with(|| left.id.0.cmp(&right.id.0))
        })
    }

    fn pooled_embedding(&self) -> Option<Vec<f32>> {
        pool_embeddings(
            self.faces
                .iter()
                .filter_map(|face| face.recognition_embedding.as_deref()),
        )
    }
}

impl AppState {
    pub(super) fn face_crop_path(&self, face_id: FaceId) -> PathBuf {
        PathBuf::from(&self.cache_root)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("faces")
            .join(format!("{}-{FACE_CROP_CACHE_VERSION}.png", face_id.0))
    }

    /// Get the current facemash pair for the arena.
    pub fn facemash_pair(
        &self,
    ) -> anyhow::Result<Option<(crate::store::FaceRecord, crate::store::FaceRecord)>> {
        let detector_model = self.embedder.face_detection_model_name();
        let min_face_side = self.facemash_min_face_side();
        let store = self.read_store()?;
        let candidates = store.facemash_identity_candidates(
            self.active.corpus_id,
            detector_model,
            min_face_side,
            FACEMASH_CANDIDATE_LIMIT,
        )?;
        let recent_faces = self
            .recent_facemash_faces
            .lock()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut viable = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let mut faces = Vec::with_capacity(candidate.faces.len());
            for face in candidate.faces {
                if face.is_local() {
                    faces.push(face);
                }
            }
            if !faces.is_empty() {
                viable.push(FacemashIdentityArena {
                    identity: candidate.identity,
                    faces,
                });
            }
        }
        if viable.len() < 2 {
            return Ok(None);
        }

        let asset_ids = viable
            .iter()
            .flat_map(|identity| identity.faces.iter())
            .filter_map(|face| face.asset_id.clone())
            .collect::<Vec<_>>();
        let asset_embeddings =
            store.embeddings_for_assets(&asset_ids, self.embedder.model_name())?;
        let identity_domains = self
            .pairing_asset_domain_labels(&store, &asset_ids, &asset_embeddings)?
            .map(|domains| {
                viable
                    .iter()
                    .filter_map(|identity| {
                        identity
                            .faces
                            .iter()
                            .find_map(|face| {
                                face.asset_id
                                    .as_ref()
                                    .and_then(|asset_id| domains.get(asset_id).copied())
                            })
                            .map(|label| (identity.identity.id, label))
                    })
                    .collect::<HashMap<_, _>>()
            });

        let oracle = self.ensure_face_oracle(&store)?;
        let predictions = viable
            .iter()
            .filter_map(|identity| {
                oracle
                    .as_ref()
                    .and_then(|oracle| {
                        identity
                            .pooled_embedding()
                            .and_then(|emb| oracle.predict(&emb))
                    })
                    .map(|prediction| (identity.identity.id, prediction))
            })
            .collect::<HashMap<_, _>>();
        let beliefs = viable
            .iter()
            .map(|identity| {
                (
                    identity.identity.id,
                    FacemashBelief::forge(
                        &identity.identity,
                        predictions.get(&identity.identity.id).copied(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let model_guided = beliefs
            .values()
            .filter(|belief| belief.model_mean.is_some())
            .count()
            * 2
            >= beliefs.len().max(1);
        let topness = Self::facemash_topness_percentiles(&viable, &beliefs);
        drop(store);

        let recent_pairs = self
            .recent_facemash_pairs
            .lock()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let recent_identities = self
            .recent_facemash_identities
            .lock()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut shortlist = BTreeSet::new();
        let mut frontier = viable
            .iter()
            .enumerate()
            .map(|(index, identity)| {
                let belief = beliefs
                    .get(&identity.identity.id)
                    .copied()
                    .unwrap_or_else(|| FacemashBelief::forge(&identity.identity, None));
                (
                    index,
                    self.facemash_frontier_score(
                        &identity.identity,
                        belief,
                        topness.get(&identity.identity.id).copied().unwrap_or(0.5),
                        Self::facemash_recent_identity_rank(
                            &recent_identities,
                            identity.identity.id,
                        ),
                    ),
                )
            })
            .collect::<Vec<_>>();
        frontier.sort_by(|lhs, rhs| {
            rhs.1.total_cmp(&lhs.1).then_with(|| {
                viable[lhs.0]
                    .identity
                    .id
                    .0
                    .cmp(&viable[rhs.0].identity.id.0)
            })
        });
        shortlist.extend(
            frontier
                .iter()
                .take(FACEMASH_FRONTIER_SHORTLIST)
                .map(|(index, _)| *index),
        );
        let mut coverage = viable
            .iter()
            .enumerate()
            .filter(|(_, identity)| {
                !model_guided
                    || topness.get(&identity.identity.id).copied().unwrap_or(0.5)
                        >= FACEMASH_MODEL_TOPNESS_FLOOR
            })
            .map(|(index, identity)| {
                let belief = beliefs
                    .get(&identity.identity.id)
                    .copied()
                    .unwrap_or_else(|| FacemashBelief::forge(&identity.identity, None));
                (
                    index,
                    self.facemash_coverage_score(
                        &identity.identity,
                        belief,
                        topness.get(&identity.identity.id).copied().unwrap_or(0.5),
                        Self::facemash_recent_identity_rank(
                            &recent_identities,
                            identity.identity.id,
                        ),
                    ),
                )
            })
            .collect::<Vec<_>>();
        coverage.sort_by(|lhs, rhs| {
            rhs.1.total_cmp(&lhs.1).then_with(|| {
                viable[lhs.0]
                    .identity
                    .id
                    .0
                    .cmp(&viable[rhs.0].identity.id.0)
            })
        });
        shortlist.extend(
            coverage
                .iter()
                .take(FACEMASH_COVERAGE_SHORTLIST)
                .map(|(index, _)| *index),
        );
        let shortlist = shortlist.into_iter().collect::<Vec<_>>();
        let mut best_pair = None::<((usize, usize), f32)>;
        for (offset, &left_index) in shortlist.iter().enumerate() {
            let left = &viable[left_index];
            let left_belief = beliefs
                .get(&left.identity.id)
                .copied()
                .unwrap_or_else(|| FacemashBelief::forge(&left.identity, None));
            for &right_index in shortlist.iter().skip(offset + 1) {
                let right = &viable[right_index];
                if identity_domains.as_ref().is_some_and(|domains| {
                    matches!(
                        (
                            domains.get(&left.identity.id),
                            domains.get(&right.identity.id)
                        ),
                        (Some(left_label), Some(right_label)) if left_label != right_label
                    )
                }) {
                    continue;
                }
                let right_belief = beliefs
                    .get(&right.identity.id)
                    .copied()
                    .unwrap_or_else(|| FacemashBelief::forge(&right.identity, None));
                let score = self.facemash_pair_score(
                    &left.identity,
                    &right.identity,
                    left_belief,
                    right_belief,
                    topness.get(&left.identity.id).copied().unwrap_or(0.5),
                    topness.get(&right.identity.id).copied().unwrap_or(0.5),
                    Self::facemash_recent_pair_rank(
                        &recent_pairs,
                        left.identity.id,
                        right.identity.id,
                    ),
                    Self::facemash_recent_identity_rank(&recent_identities, left.identity.id),
                    Self::facemash_recent_identity_rank(&recent_identities, right.identity.id),
                );
                match best_pair {
                    Some((_, best_score)) if best_score >= score => {}
                    _ => best_pair = Some(((left_index, right_index), score)),
                }
            }
        }
        Ok(best_pair.map(|((left_index, right_index), _)| {
            let left_identity = &viable[left_index];
            let right_identity = &viable[right_index];
            let left = left_identity
                .representative_face(&recent_faces)
                .unwrap_or_else(|| left_identity.faces[0].clone());
            let right = right_identity
                .representative_face(&recent_faces)
                .unwrap_or_else(|| right_identity.faces[0].clone());
            self.remember_facemash_pair(&left, &right);
            (left, right)
        }))
    }

    pub fn facemash_pair_by_ids(
        &self,
        left_face_id: FaceId,
        right_face_id: FaceId,
    ) -> anyhow::Result<Option<(crate::store::FaceRecord, crate::store::FaceRecord)>> {
        if left_face_id == right_face_id {
            return Ok(None);
        }
        let store = self.read_store()?;
        let Some(left) = store.face_by_id(left_face_id)? else {
            return Ok(None);
        };
        let Some(right) = store.face_by_id(right_face_id)? else {
            return Ok(None);
        };
        let domain_ok = if let (Some(left_asset_id), Some(right_asset_id)) =
            (left.asset_id.clone(), right.asset_id.clone())
        {
            let asset_ids = [left_asset_id.clone(), right_asset_id.clone()];
            let embeddings = store.embeddings_for_assets(&asset_ids, self.embedder.model_name())?;
            match self.pairing_asset_domain_labels(&store, &asset_ids, &embeddings)? {
                None => true,
                Some(labels) => !matches!(
                    (labels.get(&left_asset_id), labels.get(&right_asset_id)),
                    (Some(left_label), Some(right_label)) if left_label != right_label
                ),
            }
        } else {
            true
        };
        drop(store);
        let detector_model = self.embedder.face_detection_model_name();
        if left.detector_model != detector_model
            || right.detector_model != detector_model
            || !left.is_local()
            || !right.is_local()
            || left.shares_identity_with(&right)
            || !domain_ok
            || !left.usable_for_facemash(self.facemash_min_face_side())
            || !right.usable_for_facemash(self.facemash_min_face_side())
            || !self.face_preview_source_is_live(&left)?
            || !self.face_preview_source_is_live(&right)?
        {
            return Ok(None);
        }
        Ok(Some((left, right)))
    }

    pub fn facemash_target(&self) -> anyhow::Result<RedirectTarget> {
        let Some((left, right)) = self.facemash_pair()? else {
            return Ok(RedirectTarget::FacemashRoot);
        };
        Ok(RedirectTarget::FacemashPair {
            left_face_id: left.id,
            right_face_id: right.id,
        })
    }

    pub fn facemash_pair_view(&self) -> anyhow::Result<Option<FacemashPairView>> {
        let Some((left, right)) = self.facemash_pair()? else {
            return Ok(None);
        };
        self.facemash_pair_view_from_faces(left, right)
    }

    pub fn facemash_pair_view_by_ids(
        &self,
        left_face_id: FaceId,
        right_face_id: FaceId,
    ) -> anyhow::Result<Option<FacemashPairView>> {
        let Some((left, right)) = self.facemash_pair_by_ids(left_face_id, right_face_id)? else {
            return Ok(None);
        };
        self.facemash_pair_view_from_faces(left, right)
    }

    pub fn face_record(&self, face_id: FaceId) -> anyhow::Result<Option<crate::store::FaceRecord>> {
        self.read_store()?.face_by_id(face_id)
    }

    fn face_source_path_for_record(
        &self,
        face: &crate::store::FaceRecord,
    ) -> anyhow::Result<Option<PathBuf>> {
        if let Some(asset_id) = face.asset_id.as_ref() {
            return Ok(self
                .maybe_image_asset(asset_id)?
                .filter(|asset| !asset.hidden)
                .map(|asset| asset.path)
                .filter(|path| path.exists()));
        }
        if let Some(remote_item_id) = face.remote_item_id {
            return Ok(self
                .read_store()?
                .remote_item(remote_item_id)?
                .map(|item| item.path)
                .filter(|path| path.exists()));
        }
        Ok(None)
    }

    pub fn face_source_path(&self, face_id: FaceId) -> anyhow::Result<Option<PathBuf>> {
        let Some(face) = self.face_record(face_id)? else {
            return Ok(None);
        };
        self.face_source_path_for_record(&face)
    }

    fn face_preview_source_is_live(&self, face: &crate::store::FaceRecord) -> anyhow::Result<bool> {
        Ok(self.face_source_path_for_record(face)?.is_some())
    }

    fn facemash_recent_pair_rank(
        recent_pairs: &[FacemashPairKey],
        left: FaceIdentityId,
        right: FaceIdentityId,
    ) -> Option<usize> {
        let key = FacemashPairKey::forge(left, right);
        recent_pairs.iter().rev().position(|probe| *probe == key)
    }

    fn facemash_recent_identity_rank(
        recent_identities: &[FaceIdentityId],
        identity_id: FaceIdentityId,
    ) -> Option<usize> {
        recent_identities
            .iter()
            .rev()
            .position(|probe| *probe == identity_id)
    }

    pub(super) fn facemash_recent_face_rank(
        recent_faces: &[FaceId],
        face_id: FaceId,
    ) -> Option<usize> {
        recent_faces
            .iter()
            .rev()
            .position(|probe| *probe == face_id)
    }

    fn remember_facemash_pair(
        &self,
        left: &crate::store::FaceRecord,
        right: &crate::store::FaceRecord,
    ) {
        let key = FacemashPairKey::forge(left.identity.id, right.identity.id);
        let mut recent_pairs = self.recent_facemash_pairs.lock();
        if let Some(index) = recent_pairs.iter().position(|probe| *probe == key) {
            recent_pairs.remove(index);
        }
        recent_pairs.push_back(key);
        while recent_pairs.len() > FACEMASH_RECENT_PAIR_EXCLUDE {
            recent_pairs.pop_front();
        }
        drop(recent_pairs);

        let mut recent_identities = self.recent_facemash_identities.lock();
        for identity_id in [left.identity.id, right.identity.id] {
            if let Some(index) = recent_identities
                .iter()
                .position(|probe| *probe == identity_id)
            {
                recent_identities.remove(index);
            }
            recent_identities.push_back(identity_id);
        }
        while recent_identities.len() > FACEMASH_RECENT_FACE_EXCLUDE {
            recent_identities.pop_front();
        }
        drop(recent_identities);

        let mut recent_faces = self.recent_facemash_faces.lock();
        for face_id in [left.id, right.id] {
            if let Some(index) = recent_faces.iter().position(|probe| *probe == face_id) {
                recent_faces.remove(index);
            }
            recent_faces.push_back(face_id);
        }
        while recent_faces.len() > FACEMASH_RECENT_FACE_EXCLUDE {
            recent_faces.pop_front();
        }
    }

    fn facemash_frontier_score(
        &self,
        identity: &FaceIdentityRecord,
        belief: FacemashBelief,
        topness: f32,
        recent_identity_rank: Option<usize>,
    ) -> f32 {
        let scarcity = belief.scarcity(identity);
        let recency_penalty = recent_identity_rank
            .map(|rank| 220.0 / (rank as f32 + 1.0))
            .unwrap_or(0.0);
        FACEMASH_FRONTIER_TOPNESS_WEIGHT * topness
            + FACEMASH_FRONTIER_POSTERIOR_SIGMA_WEIGHT * belief.posterior_sigma
            + FACEMASH_FRONTIER_MODEL_SIGMA_WEIGHT * belief.model_uncertainty()
            + FACEMASH_FRONTIER_SCARCITY_WEIGHT * scarcity
            - recency_penalty
    }

    fn facemash_coverage_score(
        &self,
        identity: &FaceIdentityRecord,
        belief: FacemashBelief,
        topness: f32,
        recent_identity_rank: Option<usize>,
    ) -> f32 {
        let scarcity = belief.scarcity(identity);
        let recency_penalty = recent_identity_rank
            .map(|rank| 160.0 / (rank as f32 + 1.0))
            .unwrap_or(0.0);
        FACEMASH_COVERAGE_TOPNESS_WEIGHT * topness
            + FACEMASH_COVERAGE_POSTERIOR_SIGMA_WEIGHT * belief.posterior_sigma
            + FACEMASH_COVERAGE_MODEL_SIGMA_WEIGHT * belief.model_uncertainty()
            + FACEMASH_COVERAGE_SCARCITY_WEIGHT * scarcity
            - recency_penalty
    }

    fn facemash_pair_score(
        &self,
        left: &FaceIdentityRecord,
        right: &FaceIdentityRecord,
        left_belief: FacemashBelief,
        right_belief: FacemashBelief,
        left_topness: f32,
        right_topness: f32,
        recent_pair_rank: Option<usize>,
        recent_left_rank: Option<usize>,
        recent_right_rank: Option<usize>,
    ) -> f32 {
        let scarcity = left_belief.scarcity(left) + right_belief.scarcity(right);
        let pair_penalty = recent_pair_rank
            .map(|rank| 10_000.0 / (rank as f32 + 1.0))
            .unwrap_or(0.0);
        let face_penalty = [recent_left_rank, recent_right_rank]
            .into_iter()
            .flatten()
            .map(|rank| 300.0 / (rank as f32 + 1.0))
            .sum::<f32>();
        let margin = (left_belief.posterior_mean - right_belief.posterior_mean)
            / FACEMASH_PAIR_ENTROPY_TEMPERATURE;
        let win_probability = sigmoid(margin).clamp(1e-4, 1.0 - 1e-4);
        let entropy = -(win_probability * win_probability.ln()
            + (1.0 - win_probability) * (1.0 - win_probability).ln());
        let posterior_uncertainty = left_belief
            .posterior_sigma
            .hypot(right_belief.posterior_sigma);
        let model_uncertainty = left_belief
            .model_uncertainty()
            .hypot(right_belief.model_uncertainty());
        let uncertainty = FACEMASH_PAIR_POSTERIOR_SIGMA_WEIGHT * posterior_uncertainty
            + FACEMASH_PAIR_MODEL_SIGMA_WEIGHT * model_uncertainty;
        let topness = FacemashBelief::topness_bias(left_topness.max(right_topness));
        topness * entropy * uncertainty + FACEMASH_PAIR_SCARCITY_WEIGHT * scarcity
            - pair_penalty
            - face_penalty
    }

    fn facemash_topness_percentiles(
        viable: &[FacemashIdentityArena],
        beliefs: &HashMap<FaceIdentityId, FacemashBelief>,
    ) -> HashMap<FaceIdentityId, f32> {
        let mut ranking = viable
            .iter()
            .map(|identity| {
                (
                    identity.identity.id,
                    beliefs
                        .get(&identity.identity.id)
                        .copied()
                        .unwrap_or_else(|| FacemashBelief::forge(&identity.identity, None))
                        .prettiness_signal(),
                )
            })
            .collect::<Vec<_>>();
        ranking.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.0.cmp(&right.0.0))
        });
        let last_rank = ranking.len().saturating_sub(1) as f32;
        ranking
            .into_iter()
            .enumerate()
            .map(|(rank, (identity_id, _))| {
                let percentile = if last_rank <= 0.0 {
                    1.0
                } else {
                    rank as f32 / last_rank
                };
                (identity_id, percentile)
            })
            .collect()
    }

    pub fn facemash_pair_is_live(
        &self,
        left_face_id: FaceId,
        right_face_id: FaceId,
    ) -> anyhow::Result<bool> {
        Ok(self
            .facemash_pair_by_ids(left_face_id, right_face_id)?
            .is_some())
    }

    pub fn face_crop_bytes(&self, face_id: FaceId) -> anyhow::Result<Option<Vec<u8>>> {
        let Some(face) = self.face_record(face_id)? else {
            return Ok(None);
        };

        let crop_path = self.face_crop_path(face_id);
        if let Ok(bytes) = fs::read(&crop_path) {
            return Ok(Some(bytes));
        }

        let Some(source_path) = self.face_source_path_for_record(&face)? else {
            return Ok(None);
        };

        let image = crate::identity::canonical_embedding_image(
            &fs::read(&source_path)
                .with_context(|| format!("reading face source {}", source_path.display()))?,
        )
        .with_context(|| format!("canonicalizing face source {}", source_path.display()))?;
        let aligned = align_face_for_display(
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
        if let Some(parent) = crop_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        aligned
            .crop
            .save(&crop_path)
            .with_context(|| format!("writing face crop {}", crop_path.display()))?;
        let persisted_crop_path = crop_path.clone();
        self.with_write_store("set_face_aligned_path", move |store| {
            store.set_face_aligned_path(face_id, &persisted_crop_path.to_string_lossy())
        })?;
        Ok(Some(fs::read(&crop_path).with_context(|| {
            format!("reading face crop {}", crop_path.display())
        })?))
    }

    /// Record a facemash vote and retrain the oracle periodically.
    pub fn facemash_vote(&self, winner_id: FaceId, loser_id: FaceId) -> anyhow::Result<bool> {
        if !self.facemash_pair_is_live(winner_id, loser_id)? {
            return Ok(false);
        }

        let detector_model = self.embedder.face_detection_model_name().to_owned();
        let active_session_id = self.active.session_id;
        let retrain = self.with_write_store("record_face_comparison", move |store| {
            let winner = store
                .face_by_id(winner_id)?
                .with_context(|| format!("missing face {}", winner_id.0))?;
            let loser = store
                .face_by_id(loser_id)?
                .with_context(|| format!("missing face {}", loser_id.0))?;
            if winner.detector_model != detector_model || loser.detector_model != detector_model {
                bail!("facemash vote targets stale detector geometry");
            }

            let (new_winner_beauty, new_loser_beauty) =
                rate_face_win(winner.identity.beauty, loser.identity.beauty);
            store.record_face_comparison(
                active_session_id,
                winner_id,
                loser_id,
                new_winner_beauty,
                new_loser_beauty,
            )?;

            let total = store.face_comparison_count()?;
            store.touch_session(active_session_id)?;
            Ok(total % 10 == 0)
        })?;
        if retrain {
            let store = self.read_store()?;
            self.retrain_face_oracle(&store)?;
        }
        Ok(true)
    }

    pub fn facemash_hide_face(&self, face_id: FaceId) -> anyhow::Result<bool> {
        let active_session_id = self.active.session_id;
        self.with_write_store("tombstone_face", move |store| {
            let Some(face) = store.face_by_id(face_id)? else {
                return Ok(false);
            };
            if !face.is_local() {
                return Ok(false);
            }
            let changed = store.tombstone_face(face_id)?;
            if changed {
                store.touch_session(active_session_id)?;
            }
            Ok(changed)
        })
    }

    /// Facemash status for the UI.
    pub fn facemash_status(&self) -> anyhow::Result<FacemashStatus> {
        let store = self.read_store()?;
        let min_face_side = self.facemash_min_face_side();
        let total_identities = store.facemash_identity_count(
            self.active.corpus_id,
            self.embedder.face_detection_model_name(),
            min_face_side,
        )?;
        let comparisons = store.face_comparison_count()?;
        let oracle_trained = self.ensure_face_oracle(&store)?.is_some();
        Ok(FacemashStatus {
            total_identities,
            comparisons,
            oracle_trained,
            min_face_side: min_face_side.round() as u16,
        })
    }

    fn facemash_min_face_side(&self) -> f32 {
        self.config.read().facemash_min_face_side()
    }

    pub fn set_facemash_min_face_side(&self, min_face_side: u16) -> anyhow::Result<()> {
        let snapshot = {
            let mut config = self.config.write();
            config.shove_facemash_min_face_side(f32::from(min_face_side));
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(min_face_side, "updated facemash minimum face side");
        Ok(())
    }

    pub(super) fn ensure_face_oracle(&self, store: &Store) -> anyhow::Result<Option<FaceOracle>> {
        let cached = self.face_oracle.read().clone();
        if let Some(oracle) = cached {
            return Ok(Some(oracle));
        }
        let oracle = FaceOracle::train(
            &store.face_oracle_training_data(self.embedder.recognition_model_name(), 1)?,
        );
        if oracle.is_some() {
            info!("face oracle restored");
        }
        self.face_oracle.write().clone_from(&oracle);
        Ok(oracle)
    }

    pub(super) fn retrain_face_oracle(&self, store: &Store) -> anyhow::Result<()> {
        let training =
            store.face_oracle_training_data(self.embedder.recognition_model_name(), 1)?;
        let duel_count = training.duels.len();
        let oracle = FaceOracle::train(&training);
        if oracle.is_some() {
            info!(duels = duel_count, "face oracle trained");
        }
        *self.face_oracle.write() = oracle;
        Ok(())
    }

    fn facemash_pair_view_from_faces(
        &self,
        left: crate::store::FaceRecord,
        right: crate::store::FaceRecord,
    ) -> anyhow::Result<Option<FacemashPairView>> {
        let store = self.read_store()?;
        let oracle = self.ensure_face_oracle(&store)?;
        let field = self.session_field(&store)?;
        let Some(left) =
            self.facemash_face_view(&store, self.active.corpus_id, &field, oracle.as_ref(), left)?
        else {
            return Ok(None);
        };
        let Some(right) = self.facemash_face_view(
            &store,
            self.active.corpus_id,
            &field,
            oracle.as_ref(),
            right,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(FacemashPairView { left, right }))
    }

    fn facemash_face_view(
        &self,
        store: &Store,
        corpus_id: CorpusId,
        field: &SessionField,
        oracle: Option<&FaceOracle>,
        face: crate::store::FaceRecord,
    ) -> anyhow::Result<Option<FacemashFaceView>> {
        let Some(asset_id) = face.asset_id.clone() else {
            return Ok(None);
        };
        let Some(asset) = store.corpus_asset(corpus_id, &asset_id)? else {
            return Ok(None);
        };
        if asset.hidden || !asset.path.exists() {
            return Ok(None);
        }
        let identity_embedding = store.face_identity_recognition_embedding(
            face.identity.id,
            self.embedder.recognition_model_name(),
        )?;
        let predicted = oracle.and_then(|oracle| {
            identity_embedding
                .as_deref()
                .and_then(|embedding| oracle.predict(embedding))
        });
        let domain = self.asset_domain_view(store, &asset_id, field.embedding(&asset_id))?;
        let frame = face_frame_overlay(&asset, &face);
        Ok(Some(FacemashFaceView {
            face,
            asset: FacemashLocalAssetView {
                id: asset.id,
                hearted: field.hearted(&asset_id),
                rotation_quarters: asset.rotation_quarters,
                domain,
            },
            frame,
            predicted,
        }))
    }
}

fn face_frame_overlay(asset: &AssetRecord, face: &crate::store::FaceRecord) -> FaceFrameOverlay {
    let width = asset.width.max(1) as f32;
    let height = asset.height.max(1) as f32;
    let (x, y, w, h, frame_width, frame_height) = match asset.rotation_quarters.rem_euclid(4) {
        1 => (
            height - (face.bbox_y + face.bbox_h),
            face.bbox_x,
            face.bbox_h,
            face.bbox_w,
            height,
            width,
        ),
        2 => (
            width - (face.bbox_x + face.bbox_w),
            height - (face.bbox_y + face.bbox_h),
            face.bbox_w,
            face.bbox_h,
            width,
            height,
        ),
        3 => (
            face.bbox_y,
            width - (face.bbox_x + face.bbox_w),
            face.bbox_h,
            face.bbox_w,
            height,
            width,
        ),
        _ => (
            face.bbox_x,
            face.bbox_y,
            face.bbox_w,
            face.bbox_h,
            width,
            height,
        ),
    };
    FaceFrameOverlay {
        left_percent: (100.0 * x / frame_width).clamp(0.0, 100.0),
        top_percent: (100.0 * y / frame_height).clamp(0.0, 100.0),
        width_percent: (100.0 * w / frame_width).clamp(0.0, 100.0),
        height_percent: (100.0 * h / frame_height).clamp(0.0, 100.0),
    }
}
