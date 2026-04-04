use super::*;

#[derive(Debug, Clone)]
pub struct FaceIdentityRecord {
    pub id: FaceIdentityId,
    pub name: Option<String>,
    pub beauty: FaceBeauty,
    pub duel_count: u32,
}

#[derive(Debug, Clone)]
pub struct FaceRecord {
    pub id: FaceId,
    pub asset_id: Option<AssetId>,
    pub remote_item_id: Option<RemoteItemId>,
    pub identity: FaceIdentityRecord,
    pub hidden: bool,
    geometry_key: String,
    pub detector_model: String,
    pub bbox_x: f32,
    pub bbox_y: f32,
    pub bbox_w: f32,
    pub bbox_h: f32,
    pub confidence: f32,
    pub landmarks: FaceLandmarks,
    pub aligned_path: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub recognition_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct FacemashIdentityCandidate {
    pub identity: FaceIdentityRecord,
    pub faces: Vec<FaceRecord>,
}

impl FaceRecord {
    #[must_use]
    pub const fn is_local(&self) -> bool {
        self.asset_id.is_some()
    }

    #[must_use]
    pub const fn is_hidden(&self) -> bool {
        self.hidden
    }

    #[must_use]
    pub fn shares_identity_with(&self, other: &Self) -> bool {
        self.identity.id == other.identity.id
    }

    #[must_use]
    pub fn geometry_key(&self) -> &str {
        &self.geometry_key
    }

    #[must_use]
    pub fn canonical_geometry_key(&self) -> String {
        if !self.geometry_key.is_empty() {
            return self.geometry_key.clone();
        }
        let face = DetectedFace {
            bbox: crate::face::FaceBbox {
                x: self.bbox_x,
                y: self.bbox_y,
                w: self.bbox_w,
                h: self.bbox_h,
            },
            landmarks: self.landmarks.clone(),
            confidence: self.confidence,
        };
        face_geometry_key(&face)
    }

    pub fn face_src(&self) -> String {
        format!("/faces/{}", self.id.0)
    }

    pub fn usable_for_facemash(&self, min_face_side: f32) -> bool {
        if self.hidden || self.aligned_path.is_none() || self.embedding.is_none() {
            return false;
        }
        if self.confidence < FACEMASH_MIN_CONFIDENCE || self.bbox_w.min(self.bbox_h) < min_face_side
        {
            return false;
        }

        let [left_eye, right_eye, nose, left_mouth, right_mouth] =
            self.landmarks.canonical_alignment_order();
        let eye_span = (right_eye.0 - left_eye.0).abs();
        let mouth_span = (right_mouth.0 - left_mouth.0).abs();
        let eye_mid_y = (left_eye.1 + right_eye.1) * 0.5;
        let mouth_mid_y = (left_mouth.1 + right_mouth.1) * 0.5;
        let feature_height = mouth_mid_y - eye_mid_y;
        let bbox_w = self.bbox_w.max(1.0);
        let bbox_h = self.bbox_h.max(1.0);

        eye_span >= bbox_w * FACEMASH_MIN_EYE_SPAN_FRACTION
            && mouth_span >= bbox_w * FACEMASH_MIN_MOUTH_SPAN_FRACTION
            && feature_height >= bbox_h * FACEMASH_MIN_FEATURE_HEIGHT_FRACTION
            && nose.1 > eye_mid_y
            && mouth_mid_y > nose.1
    }
}

impl Store {
    pub fn face_identity_by_id(
        &self,
        identity_id: FaceIdentityId,
    ) -> anyhow::Result<Option<FaceIdentityRecord>> {
        self.conn
            .query_row(
                r"
                SELECT id, name, rating_mu, rating_sigma, compare_count
                FROM face_identities
                WHERE id = ?1
                ",
                params![identity_id.0],
                |row| {
                    Ok(FaceIdentityRecord {
                        id: FaceIdentityId(row.get::<_, i64>(0)?),
                        name: row.get(1)?,
                        beauty: FaceBeauty::forge(row.get(2)?, row.get(3)?),
                        duel_count: row.get::<_, i64>(4)?.try_into().unwrap_or(0),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn faces_exist_for_asset(&self, asset_id: &AssetId) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT 1 FROM faces WHERE asset_id = ?1 LIMIT 1",
                params![asset_id.0],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn face_scan_exists_for_asset(
        &self,
        asset_id: &AssetId,
        detector_model: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT 1 FROM face_scans WHERE asset_id = ?1 AND detector_model = ?2 LIMIT 1",
                params![asset_id.0, detector_model],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn faces_exist_for_remote_item(&self, item_id: RemoteItemId) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT 1 FROM faces WHERE remote_item_id = ?1 LIMIT 1",
                params![item_id.0],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn face_scan_exists_for_remote_item(
        &self,
        item_id: RemoteItemId,
        detector_model: &str,
    ) -> anyhow::Result<bool> {
        self.conn
            .query_row(
                "SELECT 1 FROM face_scans WHERE remote_item_id = ?1 AND detector_model = ?2 LIMIT 1",
                params![item_id.0, detector_model],
                |_| Ok(()),
            )
            .optional()
            .map(|row| row.is_some())
            .map_err(Into::into)
    }

    pub fn insert_face(
        &mut self,
        asset_id: Option<&AssetId>,
        remote_item_id: Option<RemoteItemId>,
        detector_model: &str,
        face: &DetectedFace,
        aligned_path: Option<&str>,
        embedding: Option<(&str, &[f32])>,
        recognition: Option<(&str, &[f32])>,
    ) -> anyhow::Result<FaceId> {
        let landmarks_blob = face.landmarks.to_flat_bytes();
        let geometry_key = face_geometry_key(face);
        let (emb_model, emb_dim, emb_blob) = match embedding {
            Some((model, vec)) => (
                Some(model.to_owned()),
                Some(i64::try_from(vec.len()).unwrap_or_default()),
                Some(
                    vec.iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            ),
            None => (None, None, None),
        };
        let (recognition_model, recognition_dim, recognition_blob) = match recognition {
            Some((model, vec)) => (
                Some(model.to_owned()),
                Some(i64::try_from(vec.len()).unwrap_or_default()),
                Some(
                    vec.iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            ),
            None => (None, None, None),
        };

        let tx = self
            .conn
            .transaction()
            .context("opening face insert transaction")?;
        let identity_id =
            lookup_face_identity_binding_tx(&tx, asset_id, remote_item_id, &geometry_key)?
                .unwrap_or(create_face_identity_tx(&tx, None)?);
        upsert_face_identity_binding_tx(&tx, asset_id, remote_item_id, &geometry_key, identity_id)?;
        tx.execute(
            r"
            INSERT INTO faces (
                asset_id, remote_item_id,
                identity_id, hidden, geometry_key,
                detector_model,
                bbox_x, bbox_y, bbox_w, bbox_h,
                confidence, landmarks, aligned_path,
                embedding_model, embedding_dim, embedding,
                recognition_model, recognition_dim, recognition_embedding,
                created_at
            ) VALUES (
                ?1, ?2, ?3, 0, ?4,
                ?5,
                ?6, ?7, ?8, ?9,
                ?10, ?11, ?12,
                ?13, ?14, ?15,
                ?16, ?17, ?18,
                ?19
            )
            ",
            params![
                asset_id.map(|id| &id.0),
                remote_item_id.map(|id| id.0),
                identity_id.0,
                geometry_key,
                detector_model,
                face.bbox.x,
                face.bbox.y,
                face.bbox.w,
                face.bbox.h,
                face.confidence,
                landmarks_blob,
                aligned_path,
                emb_model,
                emb_dim,
                emb_blob,
                recognition_model,
                recognition_dim,
                recognition_blob,
                now_ts(),
            ],
        )?;
        let face_id = FaceId(tx.last_insert_rowid());
        tx.commit().context("committing face insert transaction")?;
        Ok(face_id)
    }

    pub fn face_by_id(&self, face_id: FaceId) -> anyhow::Result<Option<FaceRecord>> {
        self.conn
            .query_row(
                r"
                SELECT f.id, f.asset_id, f.remote_item_id,
                       fi.id, fi.name, f.hidden, f.geometry_key,
                       f.detector_model,
                       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                       f.confidence, f.landmarks, f.aligned_path,
                       f.embedding_model, f.embedding_dim, f.embedding,
                       f.recognition_model, f.recognition_dim, f.recognition_embedding,
                       fi.rating_mu, fi.rating_sigma, fi.compare_count
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![face_id.0],
                parse_face_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn face_is_tombstoned(
        &self,
        asset_id: Option<&AssetId>,
        remote_item_id: Option<RemoteItemId>,
        geometry_key: &str,
    ) -> anyhow::Result<bool> {
        match (asset_id, remote_item_id) {
            (Some(asset_id), None) => self
                .conn
                .query_row(
                    "SELECT 1 FROM face_tombstones WHERE asset_id = ?1 AND geometry_key = ?2 LIMIT 1",
                    params![asset_id.0, geometry_key],
                    |_| Ok(()),
                )
                .optional()
                .map(|row| row.is_some())
                .map_err(Into::into),
            (None, Some(item_id)) => self
                .conn
                .query_row(
                    "SELECT 1 FROM face_tombstones WHERE remote_item_id = ?1 AND geometry_key = ?2 LIMIT 1",
                    params![item_id.0, geometry_key],
                    |_| Ok(()),
                )
                .optional()
                .map(|row| row.is_some())
                .map_err(Into::into),
            _ => bail!("face tombstone lookup requires exactly one owner"),
        }
    }

    pub fn face_is_tombstoned_for_detection(
        &self,
        asset_id: Option<&AssetId>,
        remote_item_id: Option<RemoteItemId>,
        face: &DetectedFace,
    ) -> anyhow::Result<bool> {
        let geometry_key = face_geometry_key(face);
        self.face_is_tombstoned(asset_id, remote_item_id, &geometry_key)
    }

    pub fn tombstone_face(&mut self, face_id: FaceId) -> anyhow::Result<bool> {
        let Some(face) = self.face_by_id(face_id)? else {
            return Ok(false);
        };
        let geometry_key = face.canonical_geometry_key();
        if geometry_key.is_empty() {
            return Ok(false);
        }
        let tx = self
            .conn
            .transaction()
            .context("opening face tombstone transaction")?;
        match (&face.asset_id, face.remote_item_id) {
            (Some(asset_id), None) => {
                tx.execute(
                    "DELETE FROM face_identity_bindings WHERE asset_id = ?1 AND geometry_key = ?2",
                    params![asset_id.0, geometry_key],
                )?;
                tx.execute(
                    r"
                    INSERT OR IGNORE INTO face_tombstones (asset_id, remote_item_id, geometry_key, created_at)
                    VALUES (?1, NULL, ?2, ?3)
                    ",
                    params![asset_id.0, geometry_key, now_ts()],
                )?;
            }
            (None, Some(item_id)) => {
                tx.execute(
                    "DELETE FROM face_identity_bindings WHERE remote_item_id = ?1 AND geometry_key = ?2",
                    params![item_id.0, geometry_key],
                )?;
                tx.execute(
                    r"
                    INSERT OR IGNORE INTO face_tombstones (asset_id, remote_item_id, geometry_key, created_at)
                    VALUES (NULL, ?1, ?2, ?3)
                    ",
                    params![item_id.0, geometry_key, now_ts()],
                )?;
            }
            _ => bail!("face tombstone requires exactly one owner"),
        }
        tx.execute(
            "UPDATE faces SET hidden = 1 WHERE id = ?1",
            params![face_id.0],
        )?;
        tx.commit()
            .context("committing face tombstone transaction")?;
        Ok(true)
    }

    pub fn set_face_hidden(&self, face_id: FaceId, hidden: bool) -> anyhow::Result<bool> {
        Ok(self.conn.execute(
            "UPDATE faces SET hidden = ?2 WHERE id = ?1",
            params![face_id.0, i64::from(hidden)],
        )? > 0)
    }

    pub fn rename_face_identity(
        &mut self,
        face_id: FaceId,
        raw_name: &str,
    ) -> anyhow::Result<bool> {
        let Some((identity_id, _)) = self
            .conn
            .query_row(
                r"
                SELECT fi.id, fi.name
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![face_id.0],
                |row| {
                    Ok((
                        FaceIdentityId(row.get::<_, i64>(0)?),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(false);
        };
        self.rename_face_identity_by_id(identity_id, raw_name)
    }

    pub fn rename_face_identity_by_id(
        &mut self,
        identity_id: FaceIdentityId,
        raw_name: &str,
    ) -> anyhow::Result<bool> {
        let next_name = normalize_face_identity_name(raw_name);
        let tx = self
            .conn
            .transaction()
            .context("opening face naming transaction")?;
        let Some(current_name) = tx
            .query_row(
                "SELECT name FROM face_identities WHERE id = ?1 LIMIT 1",
                params![identity_id.0],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
        else {
            return Ok(false);
        };
        match next_name {
            Some(name) => {
                if let Some(target_identity) = lookup_face_identity_by_name_tx(&tx, &name)? {
                    if identity_id != target_identity {
                        if let Some(existing_name) = current_name.as_deref()
                            && existing_name != name
                        {
                            bail!(
                                "cannot rename distinct named face identity `{existing_name}` into existing name `{name}`"
                            );
                        }
                        merge_face_identity_ids_tx(&tx, target_identity, identity_id)?;
                    }
                } else {
                    tx.execute(
                        "UPDATE face_identities SET name = ?2 WHERE id = ?1",
                        params![identity_id.0, name],
                    )?;
                }
            }
            None => {
                tx.execute(
                    "UPDATE face_identities SET name = NULL WHERE id = ?1",
                    params![identity_id.0],
                )?;
            }
        }
        tx.commit().context("committing face naming transaction")?;
        Ok(true)
    }

    pub fn fuse_face_identities(
        &mut self,
        left_face_id: FaceId,
        right_face_id: FaceId,
    ) -> anyhow::Result<bool> {
        if left_face_id == right_face_id {
            return Ok(false);
        }

        let Some((left_identity, left_name)) = self
            .conn
            .query_row(
                r"
                SELECT fi.id, fi.name
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![left_face_id.0],
                |row| {
                    Ok((
                        FaceIdentityId(row.get::<_, i64>(0)?),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(false);
        };
        let Some((right_identity, right_name)) = self
            .conn
            .query_row(
                r"
                SELECT fi.id, fi.name
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![right_face_id.0],
                |row| {
                    Ok((
                        FaceIdentityId(row.get::<_, i64>(0)?),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(false);
        };
        let (winner, loser) = match (left_name.as_deref(), right_name.as_deref()) {
            (Some(lhs), Some(rhs)) if lhs != rhs => {
                bail!("cannot merge distinct named face identities `{lhs}` and `{rhs}`");
            }
            (Some(_), _) => (left_identity, right_identity),
            (_, Some(_)) => (right_identity, left_identity),
            _ if left_identity.0 <= right_identity.0 => (left_identity, right_identity),
            _ => (right_identity, left_identity),
        };
        self.merge_face_identities_by_id(winner, loser)
    }

    pub fn merge_face_identities_by_id(
        &mut self,
        winner: FaceIdentityId,
        loser: FaceIdentityId,
    ) -> anyhow::Result<bool> {
        if winner == loser {
            return Ok(false);
        }
        let tx = self
            .conn
            .transaction()
            .context("opening face identity merge transaction")?;
        let winner_name = tx
            .query_row(
                "SELECT name FROM face_identities WHERE id = ?1",
                params![winner.0],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten();
        let loser_name = tx
            .query_row(
                "SELECT name FROM face_identities WHERE id = ?1",
                params![loser.0],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten();
        if winner_name.is_none() && loser_name.is_none() {
            merge_face_identity_ids_tx(&tx, winner, loser)?;
        } else {
            match (winner_name.as_deref(), loser_name.as_deref()) {
                (Some(lhs), Some(rhs)) if lhs != rhs => {
                    bail!("cannot merge distinct named face identities `{lhs}` and `{rhs}`");
                }
                _ => merge_face_identity_ids_tx(&tx, winner, loser)?,
            }
        }
        tx.commit()
            .context("committing face identity merge transaction")?;
        Ok(true)
    }

    pub fn facemash_identity_count(
        &self,
        corpus_id: CorpusId,
        detector_model: &str,
        min_face_side: f32,
    ) -> anyhow::Result<u64> {
        self.conn
            .query_row(
                r"
                SELECT COUNT(DISTINCT f.identity_id)
                FROM faces f
                JOIN corpus_assets ca
                  ON ca.asset_id = f.asset_id
                 AND ca.corpus_id = ?1
                 AND ca.hidden = 0
                WHERE f.detector_model = ?2
                  AND f.asset_id IS NOT NULL
                  AND f.hidden = 0
                  AND f.embedding IS NOT NULL
                  AND f.aligned_path IS NOT NULL
                  AND min(f.bbox_w, f.bbox_h) >= ?3
                  AND f.confidence >= ?4
                ",
                params![
                    corpus_id.0,
                    detector_model,
                    min_face_side,
                    FACEMASH_MIN_CONFIDENCE,
                ],
                |row| row.get::<_, u64>(0),
            )
            .map_err(Into::into)
    }

    pub fn facemash_identity_candidates(
        &self,
        corpus_id: CorpusId,
        detector_model: &str,
        min_face_side: f32,
        limit: usize,
    ) -> anyhow::Result<Vec<FacemashIdentityCandidate>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT f.id, f.asset_id, f.remote_item_id,
                   fi.id, fi.name, f.hidden, f.geometry_key,
                   f.detector_model,
                   f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                   f.confidence, f.landmarks, f.aligned_path,
                   f.embedding_model, f.embedding_dim, f.embedding,
                   f.recognition_model, f.recognition_dim, f.recognition_embedding,
                   fi.rating_mu, fi.rating_sigma, fi.compare_count
            FROM faces f
            JOIN corpus_assets ca
              ON ca.asset_id = f.asset_id
             AND ca.corpus_id = ?1
             AND ca.hidden = 0
            JOIN face_identities fi ON fi.id = f.identity_id
            WHERE f.detector_model = ?2
              AND f.asset_id IS NOT NULL
              AND f.hidden = 0
              AND f.embedding IS NOT NULL
              AND f.aligned_path IS NOT NULL
              AND min(f.bbox_w, f.bbox_h) >= ?3
              AND f.confidence >= ?4
            ORDER BY fi.compare_count ASC, RANDOM()
            LIMIT ?5
            ",
        )?;
        let mut faces = stmt
            .query_map(
                params![
                    corpus_id.0,
                    detector_model,
                    min_face_side,
                    FACEMASH_MIN_CONFIDENCE,
                    i64::try_from(limit).unwrap_or(20)
                ],
                parse_face_row,
            )?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        faces.retain(|face| face.usable_for_facemash(min_face_side));

        let mut by_identity = BTreeMap::<FaceIdentityId, FacemashIdentityCandidate>::new();
        for face in faces {
            match by_identity.entry(face.identity.id) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(FacemashIdentityCandidate {
                        identity: face.identity.clone(),
                        faces: vec![face],
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut slot) => {
                    slot.get_mut().faces.push(face);
                }
            }
        }

        let mut candidates = by_identity.into_values().collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.identity
                .duel_count
                .cmp(&right.identity.duel_count)
                .then_with(|| left.identity.id.0.cmp(&right.identity.id.0))
        });
        candidates.truncate(limit);
        Ok(candidates)
    }

    pub fn record_face_comparison(
        &self,
        session_id: SessionId,
        winner: FaceId,
        loser: FaceId,
        new_winner_beauty: FaceBeauty,
        new_loser_beauty: FaceBeauty,
    ) -> anyhow::Result<()> {
        let (winner_identity, loser_identity, winner_duel_count, loser_duel_count): (
            FaceIdentityId,
            FaceIdentityId,
            u32,
            u32,
        ) = self.conn.query_row(
            r"
            SELECT winner.identity_id,
                   loser.identity_id,
                   winner_identity.compare_count,
                   loser_identity.compare_count
            FROM faces winner
            JOIN face_identities winner_identity ON winner_identity.id = winner.identity_id,
                 faces loser
            JOIN face_identities loser_identity ON loser_identity.id = loser.identity_id
            WHERE winner.id = ?1
              AND loser.id = ?2
            ",
            params![winner.0, loser.0],
            |row| {
                Ok((
                    FaceIdentityId(row.get::<_, i64>(0)?),
                    FaceIdentityId(row.get::<_, i64>(1)?),
                    row.get::<_, i64>(2)?.try_into().unwrap_or_default(),
                    row.get::<_, i64>(3)?.try_into().unwrap_or_default(),
                ))
            },
        )?;
        if winner_identity == loser_identity {
            bail!(
                "facemash comparison collapsed to same identity {}",
                winner_identity.0
            );
        }
        self.conn.execute(
            r"
            INSERT INTO face_comparisons (session_id, winner_face_id, loser_face_id, created_at)
            VALUES (?1, ?2, ?3, ?4)
            ",
            params![session_id.0, winner.0, loser.0, now_ts()],
        )?;
        self.conn.execute(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = compare_count + 1
            WHERE id = ?1
            ",
            params![
                winner_identity.0,
                new_winner_beauty.mean,
                new_winner_beauty.sigma
            ],
        )?;
        self.conn.execute(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = compare_count + 1
            WHERE id = ?1
            ",
            params![
                loser_identity.0,
                new_loser_beauty.mean,
                new_loser_beauty.sigma
            ],
        )?;
        let model = self.active_quality_model().ok();
        let updated_at = now();
        if let Some(model) = model {
            let write_cache = |store: &Store,
                               identity_id: FaceIdentityId,
                               beauty: FaceBeauty,
                               duel_count: u32|
             -> anyhow::Result<()> {
                let payload = match model.formal_version {
                    crate::quality::QualityFormalVersion::LegacyIndependentV1 => {
                        crate::quality::legacy_subject_quality_payload(beauty, duel_count)
                    }
                    crate::quality::QualityFormalVersion::HierarchicalGaussianV1 => {
                        crate::quality::SubjectQualityCachePayload::HierarchicalGaussianV1(
                            crate::quality::HierarchicalSubjectQualityCacheV1 {
                                beauty_mean: beauty.mean,
                                beauty_variance: beauty.sigma * beauty.sigma,
                                duel_count,
                            },
                        )
                    }
                    crate::quality::QualityFormalVersion::HierarchicalPerturbativeV2 => {
                        crate::quality::SubjectQualityCachePayload::HierarchicalPerturbativeV2(
                            crate::quality::HierarchicalSubjectQualityCacheV1 {
                                beauty_mean: beauty.mean,
                                beauty_variance: beauty.sigma * beauty.sigma,
                                duel_count,
                            },
                        )
                    }
                    crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3 => {
                        crate::quality::SubjectQualityCachePayload::HierarchicalPerturbativeV3(
                            crate::quality::HierarchicalSubjectQualityCacheV1 {
                                beauty_mean: beauty.mean,
                                beauty_variance: beauty.sigma * beauty.sigma,
                                duel_count,
                            },
                        )
                    }
                };
                store.save_subject_quality_cache(&crate::quality::StoredSubjectQualityCache {
                    identity_id,
                    payload,
                    updated_at,
                })
            };
            write_cache(
                self,
                winner_identity,
                new_winner_beauty,
                winner_duel_count.saturating_add(1),
            )?;
            write_cache(
                self,
                loser_identity,
                new_loser_beauty,
                loser_duel_count.saturating_add(1),
            )?;
        }
        Ok(())
    }

    pub fn face_comparison_count(&self) -> anyhow::Result<u64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM face_comparisons", [], |row| {
                row.get::<_, i64>(0)
            })
            .map(|count| u64::try_from(count).unwrap_or_default())
            .map_err(Into::into)
    }

    pub fn face_oracle_training_data(
        &self,
        recognition_model: &str,
        min_comparisons: u32,
    ) -> anyhow::Result<FaceOracleTrainingData> {
        let duels = {
            let mut stmt = self.conn.prepare(
                r"
                SELECT wf.recognition_embedding, lf.recognition_embedding
                FROM face_comparisons fc
                JOIN faces wf ON wf.id = fc.winner_face_id
                JOIN faces lf ON lf.id = fc.loser_face_id
                WHERE wf.asset_id IS NOT NULL
                  AND lf.asset_id IS NOT NULL
                  AND wf.recognition_model = ?1
                  AND lf.recognition_model = ?1
                  AND wf.recognition_embedding IS NOT NULL
                  AND lf.recognition_embedding IS NOT NULL
                ORDER BY fc.created_at ASC, fc.id ASC
                ",
            )?;
            stmt.query_map(params![recognition_model], |row| {
                Ok(FaceOracleDuelSample {
                    winner_embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(0)?),
                    loser_embedding: decode_vec_f32(&row.get::<_, Vec<u8>>(1)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
        };

        let calibration = {
            let mut stmt = self.conn.prepare(
                r"
                SELECT f.identity_id,
                       f.recognition_embedding,
                       fi.rating_mu,
                       fi.rating_sigma,
                       fi.compare_count
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.recognition_model = ?1
                  AND f.recognition_embedding IS NOT NULL
                  AND fi.compare_count >= ?2
                  AND f.asset_id IS NOT NULL
                ORDER BY f.identity_id ASC, f.id ASC
                ",
            )?;
            let rows = stmt.query_map(params![recognition_model, min_comparisons], |row| {
                let identity_id = FaceIdentityId(row.get::<_, i64>(0)?);
                let embedding = decode_vec_f32(&row.get::<_, Vec<u8>>(1)?);
                let mean: f32 = row.get(2)?;
                let sigma: f32 = row.get(3)?;
                let compare_count = row.get::<_, i64>(4)?.try_into().unwrap_or(0);
                Ok((
                    identity_id,
                    embedding,
                    FaceBeauty::forge(mean, sigma),
                    compare_count,
                ))
            })?;
            let mut grouped = BTreeMap::<FaceIdentityId, (Vec<Vec<f32>>, FaceBeauty, u32)>::new();
            for row in rows {
                let (identity_id, embedding, beauty, compare_count) = row?;
                let entry = grouped
                    .entry(identity_id)
                    .or_insert_with(|| (Vec::new(), beauty, compare_count));
                entry.0.push(embedding);
            }
            grouped
                .into_iter()
                .filter_map(|(_, (embeddings, beauty, compare_count))| {
                    pool_embeddings(embeddings.iter().map(Vec::as_slice)).map(|embedding| {
                        FaceOracleCalibrationSample {
                            embedding,
                            beauty,
                            compare_count,
                        }
                    })
                })
                .collect::<Vec<_>>()
        };

        Ok(FaceOracleTrainingData { duels, calibration })
    }

    pub fn rebuild_identity_beauty(&self) -> anyhow::Result<()> {
        let beauty = self.compute_identity_beauty_snapshot()?;
        let mut stmt = self.conn.prepare(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = ?4
            WHERE id = ?1
            ",
        )?;
        for &(identity_id, beauty, compare_count) in &beauty {
            stmt.execute(params![
                identity_id.0,
                beauty.mean,
                beauty.sigma,
                compare_count
            ])?;
        }
        Ok(())
    }

    pub fn compute_identity_beauty_snapshot(
        &self,
    ) -> anyhow::Result<Vec<(FaceIdentityId, FaceBeauty, u32)>> {
        let mut beauty = self
            .conn
            .prepare("SELECT id FROM face_identities")?
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|id| (FaceIdentityId(id), (FaceBeauty::newborn(), 0_u32)))
            .collect::<HashMap<_, _>>();

        let comparisons = self
            .conn
            .prepare(
                r"
                SELECT wf.identity_id, lf.identity_id
                FROM face_comparisons fc
                JOIN faces wf ON wf.id = fc.winner_face_id
                JOIN faces lf ON lf.id = fc.loser_face_id
                ORDER BY fc.created_at ASC, fc.id ASC
                ",
            )?
            .query_map([], |row| {
                Ok((
                    FaceIdentityId(row.get::<_, i64>(0)?),
                    FaceIdentityId(row.get::<_, i64>(1)?),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        for (winner_id, loser_id) in comparisons {
            if winner_id == loser_id {
                continue;
            }
            let Some((winner_beauty, winner_count)) = beauty.get(&winner_id).copied() else {
                continue;
            };
            let Some((loser_beauty, loser_count)) = beauty.get(&loser_id).copied() else {
                continue;
            };
            let (winner_beauty, loser_beauty) = rate_face_win(winner_beauty, loser_beauty);
            beauty.insert(winner_id, (winner_beauty, winner_count.saturating_add(1)));
            beauty.insert(loser_id, (loser_beauty, loser_count.saturating_add(1)));
        }

        Ok(beauty
            .into_iter()
            .map(|(identity_id, (beauty, compare_count))| (identity_id, beauty, compare_count))
            .collect())
    }

    pub fn save_identity_beauty_snapshot(
        &mut self,
        beauty: &[(FaceIdentityId, FaceBeauty, u32)],
    ) -> anyhow::Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("opening identity beauty save transaction")?;
        let mut stmt = tx.prepare(
            r"
            UPDATE face_identities
            SET rating_mu = ?2,
                rating_sigma = ?3,
                compare_count = ?4
            WHERE id = ?1
            ",
        )?;
        for &(identity_id, beauty, compare_count) in beauty {
            stmt.execute(params![
                identity_id.0,
                beauty.mean,
                beauty.sigma,
                compare_count
            ])?;
        }
        drop(stmt);
        tx.commit()
            .context("committing identity beauty save transaction")?;
        let Ok(model) = self.active_quality_model() else {
            return Ok(());
        };
        let updated_at = now();
        for &(identity_id, beauty, duel_count) in beauty {
            let payload = match model.formal_version {
                crate::quality::QualityFormalVersion::LegacyIndependentV1 => {
                    crate::quality::legacy_subject_quality_payload(beauty, duel_count)
                }
                crate::quality::QualityFormalVersion::HierarchicalGaussianV1 => {
                    crate::quality::SubjectQualityCachePayload::HierarchicalGaussianV1(
                        crate::quality::HierarchicalSubjectQualityCacheV1 {
                            beauty_mean: beauty.mean,
                            beauty_variance: beauty.sigma * beauty.sigma,
                            duel_count,
                        },
                    )
                }
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV2 => {
                    crate::quality::SubjectQualityCachePayload::HierarchicalPerturbativeV2(
                        crate::quality::HierarchicalSubjectQualityCacheV1 {
                            beauty_mean: beauty.mean,
                            beauty_variance: beauty.sigma * beauty.sigma,
                            duel_count,
                        },
                    )
                }
                crate::quality::QualityFormalVersion::HierarchicalPerturbativeV3 => {
                    crate::quality::SubjectQualityCachePayload::HierarchicalPerturbativeV3(
                        crate::quality::HierarchicalSubjectQualityCacheV1 {
                            beauty_mean: beauty.mean,
                            beauty_variance: beauty.sigma * beauty.sigma,
                            duel_count,
                        },
                    )
                }
            };
            if let Err(error) =
                self.save_subject_quality_cache(&crate::quality::StoredSubjectQualityCache {
                    identity_id,
                    payload,
                    updated_at,
                })
            {
                warn!(
                    error = %format!("{error:#}"),
                    identity_id = identity_id.0,
                    "failed to mirror legacy subject quality cache"
                );
            }
        }
        Ok(())
    }

    pub fn face_identity_recognition_embedding(
        &self,
        identity_id: FaceIdentityId,
        recognition_model: &str,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT recognition_embedding
            FROM faces
            WHERE identity_id = ?1
              AND recognition_model = ?2
              AND recognition_embedding IS NOT NULL
            ORDER BY id ASC
            ",
        )?;
        let embeddings = stmt
            .query_map(params![identity_id.0, recognition_model], |row| {
                row.get::<_, Vec<u8>>(0)
            })?
            .filter_map(Result::ok)
            .filter(|blob| blob.len() % 4 == 0)
            .map(|blob| decode_vec_f32(&blob))
            .collect::<Vec<_>>();
        Ok(pool_embeddings(embeddings.iter().map(Vec::as_slice)))
    }

    pub fn dominant_local_face_identities(
        &self,
        corpus_id: CorpusId,
    ) -> anyhow::Result<HashMap<AssetId, FaceIdentityRecord>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT f.asset_id,
                   fi.id,
                   fi.name,
                   fi.rating_mu,
                   fi.rating_sigma,
                   fi.compare_count,
                   (f.bbox_w * f.bbox_h) AS face_area
            FROM faces f
            JOIN corpus_assets ca
              ON ca.asset_id = f.asset_id
             AND ca.corpus_id = ?1
             AND ca.hidden = 0
            JOIN face_identities fi ON fi.id = f.identity_id
            WHERE f.asset_id IS NOT NULL
              AND f.hidden = 0
            ORDER BY f.asset_id ASC, face_area DESC, f.id ASC
            ",
        )?;
        let rows = stmt
            .query_map(params![corpus_id.0], |row| {
                Ok((
                    AssetId(row.get::<_, String>(0)?),
                    FaceIdentityRecord {
                        id: FaceIdentityId(row.get::<_, i64>(1)?),
                        name: row.get(2)?,
                        beauty: FaceBeauty::forge(row.get(3)?, row.get(4)?),
                        duel_count: row.get::<_, i64>(5)?.try_into().unwrap_or(0),
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = HashMap::new();
        for (asset_id, identity) in rows {
            out.entry(asset_id).or_insert(identity);
        }
        Ok(out)
    }

    pub fn face_count(&self) -> anyhow::Result<u64> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| u64::try_from(count).unwrap_or_default())
            .map_err(Into::into)
    }

    pub fn identity_review_faces(
        &self,
        corpus_id: CorpusId,
        detector_model: &str,
        recognition_model: &str,
    ) -> anyhow::Result<Vec<FaceRecord>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT f.id, f.asset_id, f.remote_item_id,
                   fi.id, fi.name, f.hidden, f.geometry_key,
                   f.detector_model,
                   f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                   f.confidence, f.landmarks, f.aligned_path,
                   f.embedding_model, f.embedding_dim, f.embedding,
                   f.recognition_model, f.recognition_dim, f.recognition_embedding,
                   fi.rating_mu, fi.rating_sigma, fi.compare_count
            FROM faces f
            JOIN corpus_assets ca
              ON ca.asset_id = f.asset_id
             AND ca.corpus_id = ?1
             AND ca.hidden = 0
            JOIN face_identities fi ON fi.id = f.identity_id
            WHERE f.asset_id IS NOT NULL
              AND f.hidden = 0
              AND f.detector_model = ?2
              AND f.recognition_model = ?3
              AND f.recognition_embedding IS NOT NULL
            ORDER BY fi.id ASC, f.id ASC
            ",
        )?;
        stmt.query_map(
            params![corpus_id.0, detector_model, recognition_model],
            parse_face_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
    }

    pub fn face_identity_vetoes(
        &self,
        recognition_model: &str,
    ) -> anyhow::Result<HashSet<(FaceIdentityId, FaceIdentityId)>> {
        self.conn
            .prepare(
                r"
                SELECT identity_lo, identity_hi
                FROM face_identity_vetoes
                WHERE recognition_model = ?1
                ",
            )?
            .query_map(params![recognition_model], |row| {
                Ok((
                    FaceIdentityId(row.get::<_, i64>(0)?),
                    FaceIdentityId(row.get::<_, i64>(1)?),
                ))
            })?
            .collect::<Result<HashSet<_>, _>>()
            .map_err(Into::into)
    }

    pub fn veto_face_identity_pair(
        &mut self,
        left_face_id: FaceId,
        right_face_id: FaceId,
        recognition_model: &str,
    ) -> anyhow::Result<bool> {
        let Some((left_identity, _)) = self
            .conn
            .query_row(
                r"
                SELECT fi.id, fi.name
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![left_face_id.0],
                |row| {
                    Ok((
                        FaceIdentityId(row.get::<_, i64>(0)?),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(false);
        };
        let Some((right_identity, _)) = self
            .conn
            .query_row(
                r"
                SELECT fi.id, fi.name
                FROM faces f
                JOIN face_identities fi ON fi.id = f.identity_id
                WHERE f.id = ?1
                ",
                params![right_face_id.0],
                |row| {
                    Ok((
                        FaceIdentityId(row.get::<_, i64>(0)?),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(false);
        };
        self.veto_face_identity_pair_by_id(left_identity, right_identity, recognition_model)
    }

    pub fn veto_face_identity_pair_by_id(
        &mut self,
        left_identity: FaceIdentityId,
        right_identity: FaceIdentityId,
        recognition_model: &str,
    ) -> anyhow::Result<bool> {
        if left_identity == right_identity {
            return Ok(false);
        }
        let tx = self
            .conn
            .transaction()
            .context("opening face identity veto transaction")?;
        let (identity_lo, identity_hi) =
            canonical_face_identity_pair(left_identity, right_identity);
        tx.execute(
            r"
            INSERT OR IGNORE INTO face_identity_vetoes (
                identity_lo,
                identity_hi,
                recognition_model,
                created_at
            ) VALUES (?1, ?2, ?3, ?4)
            ",
            params![identity_lo.0, identity_hi.0, recognition_model, now_ts()],
        )?;
        tx.commit()
            .context("committing face identity veto transaction")?;
        Ok(true)
    }

    pub fn save_face_recognition(
        &self,
        face_id: FaceId,
        model_name: &str,
        vector: &[f32],
    ) -> anyhow::Result<bool> {
        let blob = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        Ok(self.conn.execute(
            r"
            UPDATE faces
            SET recognition_model = ?2,
                recognition_dim = ?3,
                recognition_embedding = ?4
            WHERE id = ?1
            ",
            params![
                face_id.0,
                model_name,
                i64::try_from(vector.len()).unwrap_or_default(),
                blob,
            ],
        )? > 0)
    }

    pub fn faces_missing_recognition(
        &self,
        corpus_id: CorpusId,
        detector_model: &str,
        recognition_model: &str,
    ) -> anyhow::Result<Vec<FaceRecord>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT f.id, f.asset_id, f.remote_item_id,
                   fi.id, fi.name, f.hidden, f.geometry_key,
                   f.detector_model,
                   f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                   f.confidence, f.landmarks, f.aligned_path,
                   f.embedding_model, f.embedding_dim, f.embedding,
                   f.recognition_model, f.recognition_dim, f.recognition_embedding,
                   fi.rating_mu, fi.rating_sigma, fi.compare_count
            FROM faces f
            JOIN corpus_assets ca
              ON ca.asset_id = f.asset_id
             AND ca.corpus_id = ?1
             AND ca.hidden = 0
            JOIN face_identities fi ON fi.id = f.identity_id
            WHERE f.asset_id IS NOT NULL
              AND f.hidden = 0
              AND f.detector_model = ?2
              AND (
                f.recognition_embedding IS NULL
                OR f.recognition_model IS NULL
                OR f.recognition_model <> ?3
              )
            ORDER BY f.id ASC
            ",
        )?;
        stmt.query_map(
            params![corpus_id.0, detector_model, recognition_model],
            parse_face_row,
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
    }

    pub fn set_face_aligned_path(&self, face_id: FaceId, path: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE faces SET aligned_path = ?2 WHERE id = ?1",
            params![face_id.0, path],
        )?;
        Ok(())
    }

    pub fn set_face_aligned_paths(&mut self, paths: &[(FaceId, String)]) -> anyhow::Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("opening face aligned-path transaction")?;
        let mut stmt = tx.prepare("UPDATE faces SET aligned_path = ?2 WHERE id = ?1")?;
        for (face_id, path) in paths {
            stmt.execute(params![face_id.0, path])?;
        }
        drop(stmt);
        tx.commit()
            .context("committing face aligned-path transaction")?;
        Ok(())
    }

    pub fn note_face_scan_for_asset(
        &self,
        asset_id: &AssetId,
        detector_model: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR IGNORE INTO face_scans (asset_id, remote_item_id, detector_model, created_at)
            VALUES (?1, NULL, ?2, ?3)
            ",
            params![asset_id.0, detector_model, now_ts()],
        )?;
        Ok(())
    }

    pub fn note_face_scan_for_remote_item(
        &self,
        item_id: RemoteItemId,
        detector_model: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            r"
            INSERT OR IGNORE INTO face_scans (asset_id, remote_item_id, detector_model, created_at)
            VALUES (NULL, ?1, ?2, ?3)
            ",
            params![item_id.0, detector_model, now_ts()],
        )?;
        Ok(())
    }
}

fn normalize_face_identity_name(raw_name: &str) -> Option<String> {
    let trimmed = raw_name.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn face_geometry_key(face: &DetectedFace) -> String {
    fn q(value: f32) -> i32 {
        (value / 6.0).round() as i32
    }

    let mut chunks = vec![
        q(face.bbox.x).to_string(),
        q(face.bbox.y).to_string(),
        q(face.bbox.w).to_string(),
        q(face.bbox.h).to_string(),
    ];
    for (x, y) in face.landmarks.canonical_alignment_order() {
        chunks.push(q(x).to_string());
        chunks.push(q(y).to_string());
    }
    blake3::hash(chunks.join(":").as_bytes())
        .to_hex()
        .to_string()
}

fn canonical_face_identity_pair(
    left: FaceIdentityId,
    right: FaceIdentityId,
) -> (FaceIdentityId, FaceIdentityId) {
    if left.0 <= right.0 {
        (left, right)
    } else {
        (right, left)
    }
}

fn lookup_face_identity_binding_tx(
    tx: &Transaction<'_>,
    asset_id: Option<&AssetId>,
    remote_item_id: Option<RemoteItemId>,
    geometry_key: &str,
) -> anyhow::Result<Option<FaceIdentityId>> {
    match (asset_id, remote_item_id) {
        (Some(asset_id), None) => tx
            .query_row(
                r"
                SELECT identity_id
                FROM face_identity_bindings
                WHERE asset_id = ?1
                  AND geometry_key = ?2
                LIMIT 1
                ",
                params![asset_id.0, geometry_key],
                |row| row.get::<_, i64>(0).map(FaceIdentityId),
            )
            .optional()
            .map_err(Into::into),
        (None, Some(remote_item_id)) => tx
            .query_row(
                r"
                SELECT identity_id
                FROM face_identity_bindings
                WHERE remote_item_id = ?1
                  AND geometry_key = ?2
                LIMIT 1
                ",
                params![remote_item_id.0, geometry_key],
                |row| row.get::<_, i64>(0).map(FaceIdentityId),
            )
            .optional()
            .map_err(Into::into),
        _ => Ok(None),
    }
}

fn upsert_face_identity_binding_tx(
    tx: &Transaction<'_>,
    asset_id: Option<&AssetId>,
    remote_item_id: Option<RemoteItemId>,
    geometry_key: &str,
    identity_id: FaceIdentityId,
) -> anyhow::Result<()> {
    match (asset_id, remote_item_id) {
        (Some(asset_id), None) => {
            tx.execute(
                "DELETE FROM face_identity_bindings WHERE asset_id = ?1 AND geometry_key = ?2",
                params![asset_id.0, geometry_key],
            )?;
            tx.execute(
                r"
                INSERT INTO face_identity_bindings (
                    asset_id,
                    remote_item_id,
                    geometry_key,
                    identity_id,
                    created_at
                ) VALUES (?1, NULL, ?2, ?3, ?4)
                ",
                params![asset_id.0, geometry_key, identity_id.0, now_ts()],
            )?;
        }
        (None, Some(remote_item_id)) => {
            tx.execute(
                "DELETE FROM face_identity_bindings WHERE remote_item_id = ?1 AND geometry_key = ?2",
                params![remote_item_id.0, geometry_key],
            )?;
            tx.execute(
                r"
                INSERT INTO face_identity_bindings (
                    asset_id,
                    remote_item_id,
                    geometry_key,
                    identity_id,
                    created_at
                ) VALUES (NULL, ?1, ?2, ?3, ?4)
                ",
                params![remote_item_id.0, geometry_key, identity_id.0, now_ts()],
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn merge_face_identity_ids_tx(
    tx: &Transaction<'_>,
    winner: FaceIdentityId,
    loser: FaceIdentityId,
) -> anyhow::Result<()> {
    if winner == loser {
        return Ok(());
    }
    tx.execute(
        "UPDATE faces SET identity_id = ?2 WHERE identity_id = ?1",
        params![loser.0, winner.0],
    )?;
    tx.execute(
        "UPDATE face_identity_bindings SET identity_id = ?2 WHERE identity_id = ?1",
        params![loser.0, winner.0],
    )?;
    tx.execute(
        "DELETE FROM face_identities WHERE id = ?1",
        params![loser.0],
    )?;
    Ok(())
}

fn lookup_face_identity_by_name_tx(
    tx: &Transaction<'_>,
    name: &str,
) -> anyhow::Result<Option<FaceIdentityId>> {
    tx.query_row(
        "SELECT id FROM face_identities WHERE name = ?1 LIMIT 1",
        params![name],
        |row| row.get::<_, i64>(0).map(FaceIdentityId),
    )
    .optional()
    .map_err(Into::into)
}

pub(super) fn create_face_identity_tx(
    tx: &Transaction<'_>,
    name: Option<&str>,
) -> anyhow::Result<FaceIdentityId> {
    let beauty = FaceBeauty::newborn();
    tx.execute(
        r"
        INSERT INTO face_identities (name, rating_mu, rating_sigma, compare_count, created_at)
        VALUES (?1, ?2, ?3, 0, ?4)
        ",
        params![name, beauty.mean, beauty.sigma, now_ts()],
    )?;
    Ok(FaceIdentityId(tx.last_insert_rowid()))
}

pub(super) fn assign_face_identity_tx(
    tx: &Transaction<'_>,
    face_id: FaceId,
    identity_id: FaceIdentityId,
) -> anyhow::Result<()> {
    tx.execute(
        "UPDATE faces SET identity_id = ?2 WHERE id = ?1",
        params![face_id.0, identity_id.0],
    )?;
    Ok(())
}

fn parse_face_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<FaceRecord> {
    let landmarks_blob: Vec<u8> = row.get(13)?;
    let landmarks =
        FaceLandmarks::from_flat_bytes(&landmarks_blob).unwrap_or(FaceLandmarks([(0.0, 0.0); 5]));
    let emb_blob: Option<Vec<u8>> = row.get(17)?;
    let embedding = emb_blob.and_then(|blob| {
        if blob.len() % 4 != 0 {
            return None;
        }
        Some(
            blob.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or_default()))
                .collect(),
        )
    });
    let recognition_blob: Option<Vec<u8>> = row.get(20)?;
    let recognition_embedding = recognition_blob.and_then(|blob| {
        if blob.len() % 4 != 0 {
            return None;
        }
        Some(
            blob.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or_default()))
                .collect(),
        )
    });
    Ok(FaceRecord {
        id: FaceId(row.get(0)?),
        asset_id: row.get::<_, Option<String>>(1)?.map(AssetId),
        remote_item_id: row.get::<_, Option<i64>>(2)?.map(RemoteItemId),
        identity: FaceIdentityRecord {
            id: FaceIdentityId(row.get::<_, i64>(3)?),
            name: row.get(4)?,
            beauty: FaceBeauty::forge(row.get(21)?, row.get(22)?),
            duel_count: row.get::<_, i64>(23)?.try_into().unwrap_or(0),
        },
        hidden: row.get::<_, i64>(5)? != 0,
        geometry_key: row.get::<_, String>(6)?,
        detector_model: row.get(7)?,
        bbox_x: row.get(8)?,
        bbox_y: row.get(9)?,
        bbox_w: row.get(10)?,
        bbox_h: row.get(11)?,
        confidence: row.get(12)?,
        landmarks,
        aligned_path: row.get(14)?,
        embedding,
        recognition_embedding,
    })
}
