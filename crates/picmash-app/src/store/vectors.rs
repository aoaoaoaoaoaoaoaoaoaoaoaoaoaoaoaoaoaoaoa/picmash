use super::*;

impl Store {
    pub fn corpus_embeddings(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<HashMap<AssetId, Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT DISTINCT e.asset_id, e.vector
            FROM embeddings e
            JOIN corpus_assets ca ON ca.asset_id = e.asset_id
            WHERE ca.corpus_id = ?1 AND e.model_name = ?2
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0, model_name], |row| {
            Ok((
                AssetId(row.get(0)?),
                decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn embeddings_for_assets(
        &self,
        asset_ids: &[AssetId],
        model_name: &str,
    ) -> anyhow::Result<HashMap<AssetId, Vec<f32>>> {
        if asset_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = (0..asset_ids.len())
            .map(|_| "?".to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r"
            SELECT asset_id, vector
            FROM embeddings
            WHERE model_name = ?1
              AND asset_id IN ({placeholders})
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params = rusqlite::params_from_iter(
            std::iter::once(model_name).chain(asset_ids.iter().map(|asset_id| asset_id.0.as_str())),
        );
        let rows = stmt.query_map(params, |row| {
            Ok((
                AssetId(row.get(0)?),
                decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        rows.collect::<Result<HashMap<_, _>, _>>()
            .map_err(Into::into)
    }

    pub fn session_embedding_head(
        &self,
        session_id: SessionId,
        model_name: &str,
    ) -> anyhow::Result<Option<SessionEmbeddingHead>> {
        self.conn
            .query_row(
                r"
                SELECT dim, weights
                FROM session_embedding_heads
                WHERE session_id = ?1 AND model_name = ?2
                ",
                params![session_id.0, model_name],
                |row| {
                    Ok(SessionEmbeddingHead {
                        model_name: model_name.to_owned(),
                        dim: usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        weights: decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn embedding(
        &self,
        asset_id: &AssetId,
        model_name: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        self.conn
            .query_row(
                r"
                SELECT vector
                FROM embeddings
                WHERE asset_id = ?1 AND model_name = ?2
                ",
                params![asset_id.0, model_name],
                |row| {
                    let bytes: Vec<u8> = row.get(0)?;
                    Ok(decode_vec_f32(&bytes))
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_embedding(
        &self,
        asset_id: &AssetId,
        embedding: &EmbeddingRecord,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO embeddings (asset_id, model_name, dim, vector)
            VALUES (?1, ?2, ?3, ?4)
            ",
            params![
                asset_id.0,
                embedding.model_name,
                i64::try_from(embedding.vector.len())?,
                encode_vec_f32(&embedding.vector),
            ],
        )?;
        Ok(())
    }

    pub fn projection_model(&self, model_name: &str) -> anyhow::Result<Option<ProjectionModel>> {
        self.conn
            .query_row(
                r"
                SELECT dim, bias0, bias1, bias2, weights
                FROM projection_models
                WHERE model_name = ?1
                ",
                params![model_name],
                |row| {
                    Ok(ProjectionModel {
                        model_name: model_name.to_owned(),
                        dim: usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        bias: [row.get(1)?, row.get(2)?, row.get(3)?],
                        weights: decode_vec_f32(&row.get::<_, Vec<u8>>(4)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_projection(&self, model: &ProjectionModel) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO projection_models (
                model_name,
                dim,
                bias0,
                bias1,
                bias2,
                weights,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                model.model_name,
                i64::try_from(model.dim)?,
                model.bias[0],
                model.bias[1],
                model.bias[2],
                encode_vec_f32(&model.weights),
                now_ts(),
            ],
        )?;
        Ok(())
    }

    pub fn similarity_model(
        &self,
        corpus_id: CorpusId,
        model_name: &str,
    ) -> anyhow::Result<Option<SimilarityModel>> {
        let row = self
            .conn
            .query_row(
                r"
                SELECT dim, mean, weights, kind, payload
                FROM similarity_models
                WHERE corpus_id = ?1 AND model_name = ?2
                ",
                params![corpus_id.0, model_name],
                |row| {
                    Ok((
                        usize::try_from(row.get::<_, i64>(0)?).unwrap_or_default(),
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                    ))
                },
            )
            .optional()
            .map_err(anyhow::Error::from)?;
        let Some((dim, mean, weights, payload)) = row else {
            return Ok(None);
        };
        if let Some(payload) = payload {
            return serde_json::from_slice::<SimilarityModel>(&payload)
                .map(Some)
                .map_err(Into::into);
        }
        Ok(Some(SimilarityModel::Linear(
            crate::model::LinearSimilarityModel {
                model_name: model_name.to_owned(),
                dim,
                mean: decode_vec_f32(&mean),
                weights: decode_vec_f32(&weights),
            },
        )))
    }

    pub fn save_similarity_model(
        &self,
        corpus_id: CorpusId,
        model: &SimilarityModel,
    ) -> anyhow::Result<()> {
        let (dim, mean, weights) = match model {
            SimilarityModel::Linear(model) => (model.dim, &model.mean, &model.weights),
            SimilarityModel::Ordinal(model) => {
                (model.prior.dim, &model.prior.mean, &model.prior.weights)
            }
        };
        self.conn.execute(
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
        Ok(())
    }

    pub fn similarity_history(
        &self,
        corpus_id: CorpusId,
    ) -> anyhow::Result<Vec<SimilarityObservation>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id, chosen_pair
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id ASC
            ",
        )?;
        let rows = stmt.query_map(params![corpus_id.0], |row| {
            Ok((
                AssetId(row.get(0)?),
                AssetId(row.get(1)?),
                AssetId(row.get(2)?),
                row.get::<_, String>(3)?,
            ))
        })?;
        let raw = rows.collect::<Result<Vec<_>, _>>()?;
        raw.into_iter()
            .map(|(asset_a, asset_b, asset_c, choice)| {
                Ok(SimilarityObservation {
                    asset_a,
                    asset_b,
                    asset_c,
                    choice: choice.parse().map_err(anyhow::Error::msg)?,
                })
            })
            .collect()
    }

    pub fn recent_similarity_asset_ids(
        &self,
        corpus_id: CorpusId,
        limit: usize,
    ) -> anyhow::Result<Vec<AssetId>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id DESC
            LIMIT ?2
            ",
        )?;
        let mut rows = stmt.query(params![corpus_id.0, i64::try_from(limit)?])?;
        let mut ids = Vec::with_capacity(limit * 3);
        while let Some(row) = rows.next()? {
            ids.push(AssetId(row.get(0)?));
            ids.push(AssetId(row.get(1)?));
            ids.push(AssetId(row.get(2)?));
        }
        Ok(ids)
    }

    pub fn recent_similarity_triads(
        &self,
        corpus_id: CorpusId,
        limit: usize,
    ) -> anyhow::Result<Vec<[AssetId; 3]>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT asset_a_id, asset_b_id, asset_c_id
            FROM similarity_triads
            WHERE corpus_id = ?1
            ORDER BY id DESC
            LIMIT ?2
            ",
        )?;
        let mut rows = stmt.query(params![corpus_id.0, i64::try_from(limit)?])?;
        let mut triads = Vec::with_capacity(limit);
        while let Some(row) = rows.next()? {
            triads.push([
                AssetId(row.get(0)?),
                AssetId(row.get(1)?),
                AssetId(row.get(2)?),
            ]);
        }
        Ok(triads)
    }
}
