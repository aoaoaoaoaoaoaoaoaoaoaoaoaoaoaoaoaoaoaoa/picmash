use super::*;

impl AppState {
    pub fn explore_empty(&self, map_mode: ExploreMapMode) -> anyhow::Result<ExploreView> {
        let store = self.read_store()?;
        Ok(match self.similarity_field(&store, Some(map_mode))? {
            Some(field) => ExploreView {
                map_mode,
                points: field.points,
                triad: None,
                selection: None,
            },
            None => ExploreView {
                map_mode,
                points: Vec::new(),
                triad: None,
                selection: None,
            },
        })
    }

    pub fn explore_target(
        &self,
        focus_id: Option<&AssetId>,
        map_mode: ExploreMapMode,
    ) -> anyhow::Result<RedirectTarget> {
        let store = self.read_store()?;
        let Some(vectors) = self.explore_vector_cache(&store)? else {
            return Ok(RedirectTarget::ExploreRoot { map_mode });
        };
        Ok(
            match choose_similarity_triad(&vectors, &store, self.active.corpus_id) {
                Some([asset_a, asset_b, asset_c]) => RedirectTarget::ExploreTriad {
                    asset_a,
                    asset_b,
                    asset_c,
                    focus_id: focus_id.cloned(),
                    map_mode,
                },
                None => RedirectTarget::ExploreRoot { map_mode },
            },
        )
    }

    pub fn explore_view(
        &self,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        focus_id: Option<&AssetId>,
        map_mode: ExploreMapMode,
    ) -> anyhow::Result<Option<ExploreView>> {
        let store = self.read_store()?;
        let Some(field) = self.similarity_field(&store, Some(map_mode))? else {
            return Ok(None);
        };
        let Some(panels) = explore_panels(&field, asset_a, asset_b, asset_c, focus_id, map_mode)
        else {
            return Ok(None);
        };
        Ok(Some(ExploreView {
            map_mode,
            selection: panels.selection,
            points: field.points,
            triad: panels.triad,
        }))
    }

    pub fn explore_panels(
        &self,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        focus_id: Option<&AssetId>,
        map_mode: ExploreMapMode,
    ) -> anyhow::Result<Option<ExplorePanels>> {
        let store = self.read_store()?;
        let Some(field) = self.similarity_field(&store, None)? else {
            return Ok(None);
        };
        Ok(explore_panels(
            &field, asset_a, asset_b, asset_c, focus_id, map_mode,
        ))
    }

    pub fn train_similarity(
        &self,
        asset_a: &AssetId,
        asset_b: &AssetId,
        asset_c: &AssetId,
        choice: SimilarityChoice,
        focus_id: Option<&AssetId>,
        map_mode: ExploreMapMode,
    ) -> anyhow::Result<RedirectTarget> {
        let store = self.read_store()?;
        let Some(mut vectors) = self.explore_vector_cache(&store)? else {
            return Ok(RedirectTarget::ExploreRoot { map_mode });
        };
        let [Some(_embedding_a), Some(_embedding_b), Some(_embedding_c)] = [
            vectors.embedding(asset_a),
            vectors.embedding(asset_b),
            vectors.embedding(asset_c),
        ] else {
            return Ok(RedirectTarget::ExploreRoot { map_mode });
        };
        let history = store.similarity_history(self.active.corpus_id)?;
        if history.len() + 1 >= crate::model::ORDINAL_BOOTSTRAP_TRIADS
            && !matches!(vectors.model, SimilarityModel::Ordinal(_))
            && let Some(bootstrapped) = vectors.model.bootstrap_ordinal(
                &vectors.embeddings,
                &history,
                LR_SIMILARITY,
                SIMILARITY_BETA,
                SIMILARITY_WEIGHT_DECAY,
            )
        {
            vectors.model = bootstrapped;
        }
        vectors.model.triad_step(
            &vectors.embeddings,
            asset_a,
            asset_b,
            asset_c,
            choice,
            LR_SIMILARITY,
            SIMILARITY_BETA,
            SIMILARITY_WEIGHT_DECAY,
        );
        vectors.refresh_learned_geometry();
        let corpus_id = self.active.corpus_id;
        let session_id = self.active.session_id;
        let asset_a = asset_a.clone();
        let asset_b = asset_b.clone();
        let asset_c = asset_c.clone();
        let persisted_model = vectors.model.clone();
        let next = choose_similarity_triad(&vectors, &store, corpus_id);
        self.with_write_store("persist_similarity_step", move |store| {
            store.persist_similarity_step(
                corpus_id,
                &persisted_model,
                &asset_a,
                &asset_b,
                &asset_c,
                choice,
            )?;
            store.touch_session(session_id)
        })?;
        self.purge_explore_vectors();
        self.purge_explore_layout(ExploreMapMode::Learned);

        Ok(match next {
            Some([next_a, next_b, next_c]) => RedirectTarget::ExploreTriad {
                asset_a: next_a,
                asset_b: next_b,
                asset_c: next_c,
                focus_id: focus_id.cloned(),
                map_mode,
            },
            None => RedirectTarget::ExploreRoot { map_mode },
        })
    }

    fn similarity_field(
        &self,
        store: &Store,
        layout_mode: Option<ExploreMapMode>,
    ) -> anyhow::Result<Option<SimilarityField>> {
        let Some(vector_cache) = self.explore_vector_cache(store)? else {
            return Ok(None);
        };
        let mut asset_map = visible_assets(store, self.active.corpus_id)?
            .into_iter()
            .map(|asset| (asset.id.clone(), asset))
            .collect::<HashMap<_, _>>();
        let assets = vector_cache
            .asset_ids
            .iter()
            .filter_map(|asset_id| asset_map.remove(asset_id))
            .collect::<Vec<_>>();
        if assets.is_empty() {
            return Ok(None);
        }
        let field = self.session_field(store)?;

        let plots = layout_mode.map_or_else(
            || vec![[0.5; MAP_DIM]; assets.len()],
            |map_mode| {
                let layout_corpus = match map_mode {
                    ExploreMapMode::Raw => vector_cache.raw_layout_corpus(),
                    ExploreMapMode::Learned => vector_cache.learned_corpus.clone(),
                };
                self.cached_explore_layout(map_mode, &vector_cache.asset_ids, &layout_corpus)
            },
        );
        let asset_ids = assets
            .iter()
            .map(|asset| asset.id.clone())
            .collect::<Vec<_>>();
        let (_, domains) = self.asset_domain_overlay(store, &asset_ids, &field.embeddings)?;
        let points = assets
            .into_iter()
            .zip(plots)
            .map(|(asset, plot)| ExploreEntry {
                latent: vector_cache
                    .latents
                    .get(&asset.id)
                    .copied()
                    .unwrap_or([0.0; SIMILARITY_DIM]),
                plot,
                domain: domains.get(&asset.id).copied().unwrap_or_default(),
                quality: field.quality_summary(&asset).unwrap(),
                asset,
            })
            .collect::<Vec<_>>();
        let point_index = points
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.asset.id.clone(), index))
            .collect::<HashMap<_, _>>();

        Ok(Some(SimilarityField {
            points,
            raw_vectors: vector_cache.raw_vectors,
            latents: vector_cache.latents,
            point_index,
        }))
    }

    fn explore_vector_cache(&self, store: &Store) -> anyhow::Result<Option<ExploreVectorCache>> {
        let mut assets = visible_assets(store, self.active.corpus_id)?;
        assets.sort_by(|lhs, rhs| lhs.id.0.cmp(&rhs.id.0));

        let cached = self.explore_vectors.read().clone();
        if let Some(cached) = cached {
            let cached_assets = assets
                .iter()
                .filter(|asset| cached.embeddings.contains_key(&asset.id))
                .map(|asset| asset.id.clone())
                .collect::<Vec<_>>();
            if cached.asset_ids == cached_assets {
                return Ok(Some(cached));
            }
        }

        let embeddings =
            store.corpus_embeddings(self.active.corpus_id, self.embedder.model_name())?;
        let assets = assets
            .into_iter()
            .filter(|asset| embeddings.contains_key(&asset.id))
            .collect::<Vec<_>>();
        if assets.is_empty() {
            return Ok(None);
        }

        let model_name = self.embedder.model_name().to_owned();
        let exemplar_dim = assets
            .iter()
            .find_map(|asset| embeddings.get(&asset.id).map(Vec::len))
            .unwrap_or_default();
        if exemplar_dim == 0 {
            return Ok(None);
        }

        let corpus = assets
            .iter()
            .filter_map(|asset| embeddings.get(&asset.id).cloned())
            .collect::<Vec<_>>();
        let seeded = || SimilarityModel::from_pca(model_name.clone(), &corpus);
        let asset_ids = assets
            .iter()
            .map(|asset| asset.id.clone())
            .collect::<Vec<_>>();
        let history = store.similarity_history(self.active.corpus_id)?;
        let mut model = match store.similarity_model(self.active.corpus_id, &model_name)? {
            Some(model)
                if model.dim() == exemplar_dim && model.is_viable_for(&asset_ids, &embeddings) =>
            {
                model
            }
            Some(_) => {
                warn!(
                    corpus_id = self.active.corpus_id.0,
                    "discarding degenerate learned similarity model and reseeding from PCA"
                );
                let Some(model) = seeded() else {
                    return Ok(None);
                };
                model
            }
            None => {
                let Some(model) = seeded() else {
                    return Ok(None);
                };
                model
            }
        };
        if history.len() >= crate::model::ORDINAL_BOOTSTRAP_TRIADS
            && !matches!(model, SimilarityModel::Ordinal(_))
            && let Some(bootstrapped) = model.bootstrap_ordinal(
                &embeddings,
                &history,
                LR_SIMILARITY,
                SIMILARITY_BETA,
                SIMILARITY_WEIGHT_DECAY,
            )
        {
            model = bootstrapped;
        }

        let raw_vectors = assets
            .iter()
            .filter_map(|asset| {
                embeddings
                    .get(&asset.id)
                    .map(|embedding| (asset.id.clone(), normalize_embedding(embedding)))
            })
            .collect::<HashMap<_, _>>();
        let mut cache = ExploreVectorCache {
            asset_ids,
            embeddings,
            raw_vectors,
            learned_corpus: Vec::new(),
            latents: HashMap::new(),
            model,
        };
        cache.refresh_learned_geometry();
        self.explore_vectors.write().replace(cache.clone());
        Ok(Some(cache))
    }

    fn cached_explore_layout(
        &self,
        map_mode: ExploreMapMode,
        asset_ids: &[AssetId],
        corpus: &[Vec<f32>],
    ) -> Vec<[f32; MAP_DIM]> {
        let cached = self
            .explore_layouts
            .read()
            .get(&map_mode)
            .filter(|cached| cached.asset_ids == asset_ids)
            .cloned();
        if let Some(cached) = cached {
            return cached.plots;
        }

        let plots = match map_mode {
            ExploreMapMode::Raw => crate::model::umap_reduce_points(corpus),
            ExploreMapMode::Learned => learned_reduce_points(corpus),
        };
        self.explore_layouts.write().insert(
            map_mode,
            ExploreLayoutCache {
                asset_ids: asset_ids.to_vec(),
                plots: plots.clone(),
            },
        );
        plots
    }

    pub(super) fn purge_all_explore_layouts(&self) {
        self.explore_layouts.write().clear();
    }

    pub(super) fn purge_explore_layout(&self, map_mode: ExploreMapMode) {
        self.explore_layouts.write().remove(&map_mode);
    }

    pub(super) fn purge_explore_vectors(&self) {
        self.explore_vectors.write().take();
    }
}

fn explore_triad(
    field: &SimilarityField,
    asset_a: &AssetId,
    asset_b: &AssetId,
    asset_c: &AssetId,
) -> Option<ExploreTriad> {
    Some(ExploreTriad {
        a: field.entry(asset_a)?,
        b: field.entry(asset_b)?,
        c: field.entry(asset_c)?,
    })
}

fn explore_panels(
    field: &SimilarityField,
    asset_a: &AssetId,
    asset_b: &AssetId,
    asset_c: &AssetId,
    focus_id: Option<&AssetId>,
    map_mode: ExploreMapMode,
) -> Option<ExplorePanels> {
    let triad = explore_triad(field, asset_a, asset_b, asset_c)?;
    Some(ExplorePanels {
        selection: field.selection(map_mode, focus_id, Some(&triad)),
        triad: Some(triad),
    })
}

fn choose_similarity_triad(
    vectors: &ExploreVectorCache,
    store: &Store,
    corpus_id: CorpusId,
) -> Option<[AssetId; 3]> {
    if vectors.asset_ids.len() < 3 {
        return None;
    }

    let recent_triads = store
        .recent_similarity_triads(corpus_id, EXPLORE_RECENT_EXCLUDE)
        .ok()?;
    let mut recent_counts = HashMap::<AssetId, usize>::new();
    let mut recent_exact = HashSet::<[AssetId; 3]>::new();
    for triad in recent_triads {
        let canonical = canonical_triad(&triad);
        for asset_id in &canonical {
            *recent_counts.entry(asset_id.clone()).or_insert(0) += 1;
        }
        recent_exact.insert(canonical);
    }

    let mut rng = rng();
    let mut anchors = vectors.asset_ids.clone();
    anchors.shuffle(&mut rng);

    let recent_pruned = anchors
        .iter()
        .filter(|asset_id| recent_count(&recent_counts, asset_id) == 0)
        .cloned()
        .collect::<Vec<_>>();
    let anchor_pool = if recent_pruned.len() >= 3 {
        recent_pruned
    } else {
        anchors
    };

    let mut contenders = Vec::<(f32, [AssetId; 3])>::new();
    for anchor_id in anchor_pool.iter().take(EXPLORE_TRIAD_ANCHORS) {
        let neighbors = vectors.nearest_neighbor_ids(
            ExploreMapMode::Learned,
            anchor_id,
            EXPLORE_TRIAD_NEIGHBORS + 3,
            false,
        );
        let mut candidate_ids = neighbors
            .into_iter()
            .filter(|(asset_id, _)| recent_count(&recent_counts, asset_id) == 0)
            .map(|(asset_id, _)| asset_id)
            .collect::<Vec<_>>();
        if candidate_ids.len() < 2 {
            candidate_ids = vectors
                .nearest_neighbor_ids(
                    ExploreMapMode::Learned,
                    anchor_id,
                    EXPLORE_TRIAD_NEIGHBORS + 3,
                    false,
                )
                .into_iter()
                .map(|(asset_id, _)| asset_id)
                .collect();
        }
        if candidate_ids.len() < 2 {
            let mut broad_fallback = vectors
                .asset_ids
                .iter()
                .filter(|candidate_id| {
                    *candidate_id != anchor_id && recent_count(&recent_counts, candidate_id) == 0
                })
                .cloned()
                .collect::<Vec<_>>();
            broad_fallback.shuffle(&mut rng);
            candidate_ids.extend(broad_fallback);
            candidate_ids.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
            candidate_ids.dedup();
        }
        if candidate_ids.len() < 2 {
            let mut broad_fallback = vectors
                .asset_ids
                .iter()
                .filter(|candidate_id| *candidate_id != anchor_id)
                .cloned()
                .collect::<Vec<_>>();
            broad_fallback.shuffle(&mut rng);
            candidate_ids.extend(broad_fallback);
            candidate_ids.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
            candidate_ids.dedup();
        }
        if candidate_ids.len() < 2 {
            continue;
        }

        let embedding_a = vectors.embedding(anchor_id)?;
        let pool_len = candidate_ids.len().min(EXPLORE_TRIAD_NEIGHBORS);
        for left in 0..pool_len {
            for right in (left + 1)..pool_len {
                let asset_b = &candidate_ids[left];
                let asset_c = &candidate_ids[right];
                let (Some(embedding_b), Some(embedding_c)) =
                    (vectors.embedding(asset_b), vectors.embedding(asset_c))
                else {
                    continue;
                };
                let probabilities = vectors.model.triad_probabilities(
                    anchor_id,
                    embedding_a,
                    asset_b,
                    embedding_b,
                    asset_c,
                    embedding_c,
                    SIMILARITY_BETA,
                );
                let entropy = triad_entropy(probabilities);
                let mean_distance = [
                    vectors.learned_distance_sq(anchor_id, asset_b)?,
                    vectors.learned_distance_sq(anchor_id, asset_c)?,
                    vectors.learned_distance_sq(asset_b, asset_c)?,
                ]
                .into_iter()
                .sum::<f32>()
                    / 3.0;
                let triad = [anchor_id.clone(), asset_b.clone(), asset_c.clone()];
                let canonical = canonical_triad(&triad);
                let exact_repeat_penalty = if recent_exact.contains(&canonical) {
                    1.8
                } else {
                    0.0
                };
                let repeat_penalty = canonical
                    .iter()
                    .map(|asset_id| recent_count(&recent_counts, asset_id) as f32)
                    .sum::<f32>()
                    * 0.28;
                let triad_score = entropy + (0.16 / (0.18 + mean_distance))
                    - exact_repeat_penalty
                    - repeat_penalty;
                contenders.push((triad_score, triad));
            }
        }
    }

    contenders.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
    let shortlist = contenders
        .into_iter()
        .take(EXPLORE_TRIAD_TOP_K)
        .collect::<Vec<_>>();
    if !shortlist.is_empty() {
        let floor = shortlist
            .iter()
            .map(|(score, _)| *score)
            .fold(f32::INFINITY, f32::min);
        let weights = shortlist
            .iter()
            .map(|(score, _)| (*score - floor + 0.05).max(0.0))
            .collect::<Vec<_>>();
        if let Some(index) = weighted_choice_index(&mut rng, &weights) {
            return Some(shortlist[index].1.clone());
        }
    }

    let mut fallback = vectors
        .asset_ids
        .iter()
        .filter(|asset_id| recent_count(&recent_counts, asset_id) == 0)
        .cloned()
        .collect::<Vec<_>>();
    if fallback.len() < 3 {
        fallback.clone_from(&vectors.asset_ids);
    }
    fallback.shuffle(&mut rng);
    (fallback.len() >= 3).then(|| {
        [
            fallback[0].clone(),
            fallback[1].clone(),
            fallback[2].clone(),
        ]
    })
}

fn recent_count(recent_counts: &HashMap<AssetId, usize>, asset_id: &AssetId) -> usize {
    recent_counts.get(asset_id).copied().unwrap_or_default()
}

fn canonical_triad(triad: &[AssetId; 3]) -> [AssetId; 3] {
    let mut canonical = triad.clone();
    canonical.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    canonical
}

fn triad_entropy(probabilities: [f32; 3]) -> f32 {
    probabilities
        .into_iter()
        .filter(|probability| *probability > f32::EPSILON)
        .map(|probability| -probability * probability.ln())
        .sum()
}
