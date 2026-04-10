use super::*;
use crate::identity::VisualKey;

impl Store {
    /// Ensure the corpus exists in the DB without scanning images.
    /// Returns the corpus ID immediately so the app can start serving.
    pub fn ensure_corpus_id(&self, root_path: &Path) -> anyhow::Result<CorpusId> {
        let root_path = root_path
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root_path.display()))?;
        self.ensure_corpus(&root_path)
    }

    fn ensure_corpus(&self, root_path: &Path) -> anyhow::Result<CorpusId> {
        let root = root_path.to_string_lossy();
        self.conn.execute(
            r"
            INSERT INTO corpora (root_path, created_at)
            VALUES (?1, ?2)
            ON CONFLICT(root_path) DO NOTHING
            ",
            params![root.as_ref(), now_ts()],
        )?;
        self.conn
            .query_row(
                "SELECT id FROM corpora WHERE root_path = ?1",
                params![root.as_ref()],
                |row| row.get::<_, i64>(0).map(CorpusId),
            )
            .map_err(Into::into)
    }

    /// Full corpus ingest: walk disk, hash, decode new images, embed.
    /// Safe to call on an already-populated corpus — unchanged images are
    /// detected by BLAKE3 and skipped without decoding.
    pub fn ingest_corpus(
        &mut self,
        root_path: &Path,
        corpus_id: CorpusId,
        embedder: &OnnxEngine,
    ) -> anyhow::Result<()> {
        if !root_path.exists() {
            bail!("corpus root does not exist: {}", root_path.display());
        }

        let root_path = root_path
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root_path.display()))?;
        let model_name = embedder.model_name().to_owned();
        let mut dino_available = embedder.enabled();
        let mut skipped_identity = BTreeMap::<String, IngestSkipSummary>::new();

        let image_paths: Vec<PathBuf> = WalkDir::new(&root_path)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .filter(|path| is_supported_image(path))
            .collect();
        let total = image_paths.len();
        info!(total, "corpus walk complete, ingesting images");

        let mut ingested = 0usize;
        let mut unchanged = 0usize;
        let mut embed_count = 0usize;
        let log_interval = (total / 20).max(50);

        for path in image_paths {
            let bytes = fs::read(&path)
                .with_context(|| format!("reading image bytes from {}", path.display()))?;
            let blob_id = BlobId(blake3::hash(&bytes).to_hex().to_string());
            let path_string = path.to_string_lossy().into_owned();

            if let Some(asset_id) = self.corpus_blob_match(corpus_id, &path_string, &blob_id)? {
                if dino_available && !self.embedding_exists(&asset_id, &model_name)? {
                    match embedder.embed(&path) {
                        Ok(Some(embedding)) => {
                            embed_count += 1;
                            self.save_embedding(&asset_id, &embedding)?;
                            let mut projection = self
                                .projection_model(&embedding.model_name)?
                                .unwrap_or_else(|| {
                                    ProjectionModel::zero(
                                        embedding.model_name.clone(),
                                        embedding.vector.len(),
                                    )
                                });
                            if projection.dim != embedding.vector.len() {
                                projection = ProjectionModel::zero(
                                    embedding.model_name.clone(),
                                    embedding.vector.len(),
                                );
                            }
                            self.save_projection(&projection)?;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            dino_available = false;
                            warn!(
                                "disabling ONNX embedding after runtime failure on {}: {error:#}",
                                path.display(),
                            );
                        }
                    }
                }
                unchanged += 1;
                ingested += 1;
                if ingested.is_multiple_of(log_interval) {
                    info!(
                        ingested,
                        total,
                        unchanged,
                        embedded = embed_count,
                        "corpus ingest progress"
                    );
                }
                continue;
            }

            let identity = match inspect_image_bytes(&bytes) {
                Ok(identity) => identity,
                Err(error) => {
                    note_ingest_skip(&mut skipped_identity, compact_ingest_error(&error), &path);
                    continue;
                }
            };

            let tx = self
                .conn
                .transaction()
                .context("opening ingest transaction")?;
            let asset_id =
                resolve_asset_id_for_identity(&tx, &identity)?.unwrap_or_else(mint_asset_id);
            let hidden = preserved_hidden_state_tx(&tx, corpus_id, &path_string, &asset_id)?;
            upsert_asset_identity_tx(&tx, &asset_id, &identity, 0)?;
            resolve_external_aliases_for_asset_identity_tx(&tx, &asset_id, &identity)?;
            upsert_corpus_variant_tx(&tx, corpus_id, &path_string, &asset_id, &identity, hidden)?;
            let needs_embedding = !embedding_exists_tx(&tx, &asset_id, &model_name)?;
            tx.commit().context("committing ingest transaction")?;

            if dino_available && needs_embedding {
                match embedder.embed(&path) {
                    Ok(Some(embedding)) => {
                        embed_count += 1;
                        self.save_embedding(&asset_id, &embedding)?;
                        let mut projection = self
                            .projection_model(&embedding.model_name)?
                            .unwrap_or_else(|| {
                                ProjectionModel::zero(
                                    embedding.model_name.clone(),
                                    embedding.vector.len(),
                                )
                            });
                        if projection.dim != embedding.vector.len() {
                            projection = ProjectionModel::zero(
                                embedding.model_name.clone(),
                                embedding.vector.len(),
                            );
                        }
                        self.save_projection(&projection)?;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        dino_available = false;
                        warn!(
                            "disabling ONNX embedding after runtime failure on {}: {error:#}",
                            path.display(),
                        );
                    }
                }
            }

            ingested += 1;
            if ingested.is_multiple_of(log_interval) {
                info!(
                    ingested,
                    total,
                    unchanged,
                    embedded = embed_count,
                    "corpus ingest progress"
                );
            }
        }

        for (cause, summary) in skipped_identity {
            warn!(
                skipped = summary.count,
                sample = %summary.sample_path.display(),
                error = %cause,
                "skipped images during ingest"
            );
        }

        info!(
            ingested,
            unchanged,
            embedded = embed_count,
            "corpus ingest complete"
        );
        Ok(())
    }

    fn corpus_blob_match(
        &self,
        corpus_id: CorpusId,
        path: &str,
        blob_id: &BlobId,
    ) -> anyhow::Result<Option<AssetId>> {
        let matched = self
            .conn
            .query_row(
                r"
                SELECT asset_id
                FROM corpus_assets
                WHERE corpus_id = ?1 AND path = ?2 AND blob_id = ?3
                LIMIT 1
                ",
                params![corpus_id.0, path, blob_id.0],
                |row| row.get::<_, String>(0).map(AssetId),
            )
            .optional()?;
        Ok(matched)
    }

    fn embedding_exists(&self, asset_id: &AssetId, model_name: &str) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                r"
                SELECT 1
                FROM embeddings
                WHERE asset_id = ?1 AND model_name = ?2
                LIMIT 1
                ",
                params![asset_id.0, model_name],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn corpus_assets(&self, corpus_id: CorpusId) -> anyhow::Result<Vec<AssetRecord>> {
        let mut assets = choose_corpus_asset_rows(self.load_corpus_asset_rows(corpus_id, None)?);
        assets.sort_by(|left, right| {
            right
                .alpha
                .total_cmp(&left.alpha)
                .then_with(|| right.win_count.cmp(&left.win_count))
                .then_with(|| right.compare_count.cmp(&left.compare_count))
                .then_with(|| left.id.0.cmp(&right.id.0))
        });
        Ok(assets)
    }

    pub fn corpus_asset(
        &self,
        corpus_id: CorpusId,
        asset_id: &AssetId,
    ) -> anyhow::Result<Option<AssetRecord>> {
        Ok(
            choose_corpus_asset_rows(self.load_corpus_asset_rows(corpus_id, Some(asset_id))?)
                .into_iter()
                .next(),
        )
    }

    fn load_corpus_asset_rows(
        &self,
        corpus_id: CorpusId,
        asset_id: Option<&AssetId>,
    ) -> anyhow::Result<Vec<CorpusAssetRow>> {
        let sql = if asset_id.is_some() {
            r"
            SELECT a.id,
                   ca.path,
                   a.alpha,
                   a.c0,
                   a.c1,
                   a.c2,
                   a.rotation_quarters,
                   a.compare_count,
                   a.win_count,
                   a.heart_count,
                   a.hearted,
                   CASE
                       WHEN ca.hidden != 0 OR tombstone.visual_key IS NOT NULL THEN 1
                       ELSE 0
                   END,
                   a.preferred_blob_id,
                   ca.blob_id,
                   ca.blob_width,
                   ca.blob_height,
                   a.visual_key
            FROM corpus_assets ca
            JOIN assets a ON a.id = ca.asset_id
            LEFT JOIN external_item_tombstones tombstone
              ON tombstone.visual_key = a.visual_key
            WHERE ca.corpus_id = ?1
              AND ca.asset_id = ?2
            "
        } else {
            r"
            SELECT a.id,
                   ca.path,
                   a.alpha,
                   a.c0,
                   a.c1,
                   a.c2,
                   a.rotation_quarters,
                   a.compare_count,
                   a.win_count,
                   a.heart_count,
                   a.hearted,
                   CASE
                       WHEN ca.hidden != 0 OR tombstone.visual_key IS NOT NULL THEN 1
                       ELSE 0
                   END,
                   a.preferred_blob_id,
                   ca.blob_id,
                   ca.blob_width,
                   ca.blob_height,
                   a.visual_key
            FROM corpus_assets ca
            JOIN assets a ON a.id = ca.asset_id
            LEFT JOIN external_item_tombstones tombstone
              ON tombstone.visual_key = a.visual_key
            WHERE ca.corpus_id = ?1
            "
        };
        let mut stmt = self.conn.prepare(sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            Ok(CorpusAssetRow {
                asset: AssetRecord {
                    id: AssetId(row.get(0)?),
                    path: PathBuf::from(row.get::<_, String>(1)?),
                    visual_key: row.get::<_, Option<String>>(16)?.map(VisualKey),
                    width: u32::try_from(row.get::<_, i64>(14)?).unwrap_or_default(),
                    height: u32::try_from(row.get::<_, i64>(15)?).unwrap_or_default(),
                    alpha: row.get(2)?,
                    coords: [row.get(3)?, row.get(4)?, row.get(5)?],
                    rotation_quarters: row.get(6)?,
                    compare_count: u32::try_from(row.get::<_, i64>(7)?).unwrap_or_default(),
                    win_count: u32::try_from(row.get::<_, i64>(8)?).unwrap_or_default(),
                    heart_count: u32::try_from(row.get::<_, i64>(9)?).unwrap_or_default(),
                    is_hearted: row.get::<_, i64>(10)? != 0,
                    hidden: row.get::<_, i64>(11)? != 0,
                },
                preferred_blob_id: row.get::<_, Option<String>>(12)?.map(BlobId),
                variant_blob_id: row.get::<_, Option<String>>(13)?.map(BlobId),
                blob_width: u32::try_from(row.get::<_, i64>(14)?).unwrap_or_default(),
                blob_height: u32::try_from(row.get::<_, i64>(15)?).unwrap_or_default(),
            })
        };
        let rows = match asset_id {
            Some(asset_id) => stmt.query_map(params![corpus_id.0, asset_id.0], map_row)?,
            None => stmt.query_map(params![corpus_id.0], map_row)?,
        };
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn asset_domain_labels(
        &self,
        asset_ids: &[AssetId],
    ) -> anyhow::Result<HashMap<AssetId, AssetDomainLabel>> {
        if asset_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = (0..asset_ids.len())
            .map(|_| "?".to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r"
            SELECT asset_id, label
            FROM asset_domain_labels
            WHERE asset_id IN ({placeholders})
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params =
            rusqlite::params_from_iter(asset_ids.iter().map(|asset_id| asset_id.0.as_str()));
        let rows = stmt.query_map(params, |row| {
            let raw: String = row.get(1)?;
            let label = raw.parse::<AssetDomainLabel>().map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
                )
            })?;
            Ok((AssetId(row.get(0)?), label))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn set_asset_domain_label(
        &self,
        asset_id: &AssetId,
        label: AssetDomainLabel,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT INTO asset_domain_labels (asset_id, label, updated_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(asset_id) DO UPDATE
            SET label = excluded.label,
                updated_at = excluded.updated_at
            ",
            params![asset_id.0, label.as_str(), now_ts()],
        )?;
        Ok(())
    }

    pub fn asset_domain_label_counts(&self) -> anyhow::Result<(usize, usize)> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT label, COUNT(*)
            FROM asset_domain_labels
            GROUP BY label
            ",
        )?;
        let mut real = 0usize;
        let mut anime = 0usize;
        for row in stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })? {
            let (label, count) = row?;
            let count = usize::try_from(count).unwrap_or(0);
            match label.parse::<AssetDomainLabel>() {
                Ok(AssetDomainLabel::Real) => real = count,
                Ok(AssetDomainLabel::Anime) => anime = count,
                Err(_) => {}
            }
        }
        Ok((real, anime))
    }

    pub fn asset_domain_training_rows(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<Vec<AssetDomainTrainingRow>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT ad.asset_id, ad.label, e.vector
            FROM asset_domain_labels ad
            JOIN corpus_assets ca
              ON ca.asset_id = ad.asset_id
             AND ca.corpus_id = ?1
            JOIN embeddings e
              ON e.asset_id = ad.asset_id
             AND e.model_name = ?2
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0, model_name], |row| {
            let raw_label: String = row.get(1)?;
            let label = raw_label.parse::<AssetDomainLabel>().map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
                )
            })?;
            Ok(AssetDomainTrainingRow {
                asset_id: AssetId(row.get(0)?),
                label,
                embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(2)?),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn ingest_external_import(
        &mut self,
        corpus_id: CorpusId,
        import_path: &Path,
        bytes: &[u8],
        rotation_quarters: i32,
        embedding: Option<&EmbeddingRecord>,
    ) -> anyhow::Result<AssetId> {
        let identity = inspect_image_bytes(bytes).context("inspecting external import identity")?;
        self.ingest_external_import_precomputed(
            corpus_id,
            import_path,
            &identity,
            rotation_quarters,
            embedding,
        )
    }

    pub fn ingest_external_import_precomputed(
        &mut self,
        corpus_id: CorpusId,
        import_path: &Path,
        identity: &ImageIdentity,
        rotation_quarters: i32,
        embedding: Option<&EmbeddingRecord>,
    ) -> anyhow::Result<AssetId> {
        let path_string = import_path.to_string_lossy().into_owned();
        let tx = self
            .conn
            .transaction()
            .context("opening external import transaction")?;
        let asset_id = resolve_asset_id_for_identity(&tx, identity)?.unwrap_or_else(mint_asset_id);
        let hidden = preserved_hidden_state_tx(&tx, corpus_id, &path_string, &asset_id)?;
        upsert_asset_identity_tx(&tx, &asset_id, identity, rotation_quarters.rem_euclid(4))?;
        resolve_external_aliases_for_asset_identity_tx(&tx, &asset_id, identity)?;
        upsert_corpus_variant_tx(&tx, corpus_id, &path_string, &asset_id, identity, hidden)?;
        if let Some(embedding) = embedding {
            upsert_embedding_tx(&tx, &asset_id, embedding)?;
        }
        tx.commit()
            .context("committing external import transaction")?;
        Ok(asset_id)
    }
}
