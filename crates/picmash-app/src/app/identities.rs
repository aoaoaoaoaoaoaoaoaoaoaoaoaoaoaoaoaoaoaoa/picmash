use super::*;
use std::collections::{BTreeMap, HashMap, HashSet};

const IDENTITY_HANDLE_TAG_ANCHOR: u8 = 0x41;
const IDENTITY_HANDLE_TAG_PAIR: u8 = 0x50;
const IDENTITY_REVIEW_MAX_ROWS: usize = 64;
const IDENTITY_REVIEW_MAX_CANDIDATES_PER_ROW: usize = 12;
const IDENTITY_REVIEW_MAX_FOCUS_CANDIDATES: usize = 64;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct IdentityReviewStatus {
    pub total_subjects: usize,
    pub named_subjects: usize,
    pub active_rows: usize,
    pub threshold_percent: u8,
}

#[derive(Debug, Clone)]
pub struct IdentityReviewView {
    pub status: IdentityReviewStatus,
    pub focus_name: Option<String>,
    pub recognition_ready: bool,
    pub rows: Vec<IdentityReviewRowView>,
}

#[derive(Debug, Clone)]
pub struct IdentityReviewRowView {
    pub name_slug: Option<String>,
    pub anchor_handle: String,
    pub anchor_face: crate::store::FaceRecord,
    pub anchor_name: Option<String>,
    pub member_count: usize,
    pub candidate_count: usize,
    pub top_similarity: Option<f32>,
    pub candidates: Vec<IdentityReviewCandidateView>,
}

#[derive(Debug, Clone)]
pub struct IdentityReviewCandidateView {
    pub pair_handle: String,
    pub face: crate::store::FaceRecord,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
struct IdentityArenaSubject {
    identity: FaceIdentityRecord,
    prototype: Vec<f32>,
    representative_face: crate::store::FaceRecord,
    member_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct IdentityReviewEdge {
    candidate_index: usize,
    similarity: f32,
}

impl IdentityArenaSubject {
    fn forge(identity: FaceIdentityRecord, faces: Vec<crate::store::FaceRecord>) -> Option<Self> {
        let prototype = pool_embeddings(
            faces
                .iter()
                .filter_map(|face| face.recognition_embedding.as_deref()),
        )?;
        let representative_face = faces
            .into_iter()
            .filter_map(|face| {
                let embedding = face.recognition_embedding.as_deref()?;
                let score = embedding_dot(&prototype, embedding);
                Some((score, face))
            })
            .max_by(|lhs, rhs| {
                lhs.0
                    .total_cmp(&rhs.0)
                    .then_with(|| lhs.1.id.0.cmp(&rhs.1.id.0))
            })?
            .1;
        Some(Self {
            identity,
            prototype,
            representative_face,
            member_count: 0,
        })
    }
}

impl AppState {
    pub fn identity_review_view(
        &self,
        focus_slug: Option<&str>,
        focus_anchor_handle: Option<&str>,
    ) -> anyhow::Result<IdentityReviewView> {
        let threshold = self.identity_match_threshold();
        if !self.embedder.recognition_enabled() {
            return Ok(IdentityReviewView {
                status: IdentityReviewStatus {
                    total_subjects: 0,
                    named_subjects: 0,
                    active_rows: 0,
                    threshold_percent: (threshold * 100.0).round() as u8,
                },
                focus_name: None,
                recognition_ready: false,
                rows: Vec::new(),
            });
        }

        let store = self.store.lock();
        let detector_model = self.embedder.face_detection_model_name();
        let recognition_model = self.embedder.recognition_model_name();
        let faces = store.identity_review_faces(
            self.active.corpus_id,
            detector_model,
            recognition_model,
        )?;
        let vetoes = store.face_identity_vetoes(recognition_model)?;
        drop(store);

        let mut grouped =
            BTreeMap::<FaceIdentityId, (FaceIdentityRecord, Vec<crate::store::FaceRecord>)>::new();
        for face in faces {
            match grouped.entry(face.identity.id) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert((face.identity.clone(), vec![face]));
                }
                std::collections::btree_map::Entry::Occupied(mut slot) => {
                    slot.get_mut().1.push(face);
                }
            }
        }

        let subjects = grouped
            .into_values()
            .filter_map(|(identity, faces)| {
                let member_count = faces.len();
                let mut subject = IdentityArenaSubject::forge(identity, faces)?;
                subject.member_count = member_count;
                Some(subject)
            })
            .collect::<Vec<_>>();

        let total_subjects = subjects.len();
        let named_subjects = subjects
            .iter()
            .filter(|subject| subject.identity.name.is_some())
            .count();
        if subjects.is_empty() {
            return Ok(IdentityReviewView {
                status: IdentityReviewStatus {
                    total_subjects,
                    named_subjects,
                    active_rows: 0,
                    threshold_percent: (threshold * 100.0).round() as u8,
                },
                focus_name: None,
                recognition_ready: true,
                rows: Vec::new(),
            });
        }

        let subject_index = subjects
            .iter()
            .enumerate()
            .map(|(index, subject)| (subject.identity.id, index))
            .collect::<HashMap<_, _>>();
        let tree = VpTree::forge(
            subjects
                .iter()
                .map(|subject| (subject.identity.id, subject.prototype.clone()))
                .collect::<Vec<_>>(),
        );
        let radius = identity_similarity_radius(threshold);
        let candidate_sets = subjects
            .iter()
            .map(|anchor| {
                let mut edges = tree
                    .ransack(&anchor.prototype, radius)
                    .into_iter()
                    .filter_map(|(candidate_id, _)| {
                        if candidate_id == anchor.identity.id {
                            return None;
                        }
                        let candidate_index = *subject_index.get(&candidate_id)?;
                        let candidate = subjects.get(candidate_index)?;
                        if candidate.identity.name.is_some() {
                            return None;
                        }
                        if vetoes.contains(&canonical_identity_pair(
                            anchor.identity.id,
                            candidate.identity.id,
                        )) {
                            return None;
                        }
                        let similarity = embedding_dot(&anchor.prototype, &candidate.prototype);
                        if similarity < threshold {
                            return None;
                        }
                        Some(IdentityReviewEdge {
                            candidate_index,
                            similarity,
                        })
                    })
                    .collect::<Vec<_>>();
                edges.sort_by(|lhs, rhs| {
                    rhs.similarity.total_cmp(&lhs.similarity).then_with(|| {
                        subjects[lhs.candidate_index]
                            .identity
                            .id
                            .0
                            .cmp(&subjects[rhs.candidate_index].identity.id.0)
                    })
                });
                edges
            })
            .collect::<Vec<_>>();

        let make_row = |anchor: &IdentityArenaSubject,
                        edges: &[IdentityReviewEdge],
                        candidate_limit: usize| {
            let candidate_count = edges.len();
            let top_similarity = edges.first().map(|edge| edge.similarity);
            let candidates = edges
                .iter()
                .take(candidate_limit)
                .map(|edge| {
                    let candidate = &subjects[edge.candidate_index];
                    IdentityReviewCandidateView {
                        pair_handle: self
                            .mint_identity_pair_handle(anchor.identity.id, candidate.identity.id),
                        face: candidate.representative_face.clone(),
                        similarity: edge.similarity,
                    }
                })
                .collect::<Vec<_>>();
            IdentityReviewRowView {
                name_slug: anchor.identity.name.as_deref().map(slugify_subject_name),
                anchor_handle: self.mint_identity_anchor_handle(anchor.identity.id),
                anchor_face: anchor.representative_face.clone(),
                anchor_name: anchor.identity.name.clone(),
                member_count: anchor.member_count,
                candidate_count,
                top_similarity,
                candidates,
            }
        };

        let focus_anchor_identity = focus_anchor_handle
            .and_then(|anchor_handle| self.resolve_identity_anchor_handle(anchor_handle));

        let mut rows = if let Some(slug) = focus_slug {
            subjects
                .iter()
                .enumerate()
                .find(|(_, subject)| {
                    subject
                        .identity
                        .name
                        .as_deref()
                        .map(slugify_subject_name)
                        .as_deref()
                        == Some(slug)
                })
                .map(|(anchor_index, anchor)| {
                    vec![make_row(
                        anchor,
                        &candidate_sets[anchor_index],
                        IDENTITY_REVIEW_MAX_FOCUS_CANDIDATES,
                    )]
                })
                .unwrap_or_default()
        } else if let Some(anchor_identity) = focus_anchor_identity {
            subjects
                .iter()
                .enumerate()
                .find(|(_, subject)| subject.identity.id == anchor_identity)
                .map(|(anchor_index, anchor)| {
                    vec![make_row(
                        anchor,
                        &candidate_sets[anchor_index],
                        IDENTITY_REVIEW_MAX_FOCUS_CANDIDATES,
                    )]
                })
                .unwrap_or_default()
        } else {
            let mut anchor_order = (0..subjects.len()).collect::<Vec<_>>();
            anchor_order.sort_by(|lhs, rhs| {
                let lhs_subject = &subjects[*lhs];
                let rhs_subject = &subjects[*rhs];
                rhs_subject
                    .identity
                    .name
                    .is_some()
                    .cmp(&lhs_subject.identity.name.is_some())
                    .then_with(|| {
                        candidate_sets[*rhs]
                            .first()
                            .map(|edge| edge.similarity)
                            .unwrap_or(-2.0)
                            .total_cmp(
                                &candidate_sets[*lhs]
                                    .first()
                                    .map(|edge| edge.similarity)
                                    .unwrap_or(-2.0),
                            )
                    })
                    .then_with(|| candidate_sets[*rhs].len().cmp(&candidate_sets[*lhs].len()))
                    .then_with(|| rhs_subject.member_count.cmp(&lhs_subject.member_count))
                    .then_with(|| lhs_subject.identity.id.0.cmp(&rhs_subject.identity.id.0))
            });

            let mut claimed_candidates = HashSet::new();
            let mut rows = Vec::new();
            for anchor_index in anchor_order {
                let anchor = &subjects[anchor_index];
                if claimed_candidates.contains(&anchor.identity.id) {
                    continue;
                }
                let edges = candidate_sets[anchor_index]
                    .iter()
                    .copied()
                    .filter(|edge| {
                        !claimed_candidates.contains(&subjects[edge.candidate_index].identity.id)
                    })
                    .collect::<Vec<_>>();
                if edges.is_empty() {
                    continue;
                }
                for edge in &edges {
                    claimed_candidates.insert(subjects[edge.candidate_index].identity.id);
                }
                rows.push(make_row(
                    anchor,
                    &edges,
                    IDENTITY_REVIEW_MAX_CANDIDATES_PER_ROW,
                ));
            }
            rows
        };

        rows.sort_by(|lhs, rhs| {
            rhs.top_similarity
                .unwrap_or(-2.0)
                .total_cmp(&lhs.top_similarity.unwrap_or(-2.0))
                .then_with(|| rhs.candidates.len().cmp(&lhs.candidates.len()))
                .then_with(|| rhs.anchor_name.is_some().cmp(&lhs.anchor_name.is_some()))
                .then_with(|| {
                    lhs.anchor_face
                        .identity
                        .id
                        .0
                        .cmp(&rhs.anchor_face.identity.id.0)
                })
        });

        let mut focus_name = None;
        if focus_slug.is_some() || focus_anchor_handle.is_some() {
            focus_name = rows
                .first()
                .and_then(|row| row.anchor_name.clone())
                .or_else(|| rows.first().map(|_| "unnamed subject".to_owned()));
        } else {
            rows.truncate(IDENTITY_REVIEW_MAX_ROWS);
        }

        Ok(IdentityReviewView {
            status: IdentityReviewStatus {
                total_subjects,
                named_subjects,
                active_rows: rows.len(),
                threshold_percent: (threshold * 100.0).round() as u8,
            },
            focus_name,
            recognition_ready: true,
            rows,
        })
    }

    pub fn set_identity_match_threshold_percent(
        &self,
        threshold_percent: u8,
    ) -> anyhow::Result<()> {
        let threshold = (f32::from(threshold_percent) / 100.0).clamp(0.0, 1.0);
        let snapshot = {
            let mut config = self.config.write();
            config.shove_identity_match_threshold(threshold);
            let snapshot = config.clone().normalized();
            *config = snapshot.clone();
            snapshot
        };
        self.persist_live_config(&snapshot)?;
        info!(
            threshold,
            threshold_percent, "updated identity match threshold"
        );
        Ok(())
    }

    pub fn identity_review_rename(&self, anchor_handle: &str, name: &str) -> anyhow::Result<bool> {
        let Some(identity_id) = self.resolve_identity_anchor_handle(anchor_handle) else {
            return Ok(false);
        };
        self.with_db_write_gate(|| {
            let mut store = self.store.lock();
            let changed = store.rename_face_identity_by_id(identity_id, name)?;
            if changed {
                store.touch_session(self.active.session_id)?;
            }
            Ok(changed)
        })
    }

    pub fn identity_review_confirm(&self, pair_handle: &str) -> anyhow::Result<bool> {
        let Some((anchor_identity, candidate_identity)) =
            self.resolve_identity_pair_handle(pair_handle)
        else {
            return Ok(false);
        };
        self.with_db_write_gate(|| {
            let mut store = self.store.lock();
            let Some(_anchor) = store.face_identity_by_id(anchor_identity)? else {
                return Ok(false);
            };
            let Some(candidate) = store.face_identity_by_id(candidate_identity)? else {
                return Ok(false);
            };
            if candidate.name.is_some() {
                return Ok(false);
            }
            let changed = store.merge_face_identities_by_id(anchor_identity, candidate_identity)?;
            if changed {
                store.touch_session(self.active.session_id)?;
            }
            Ok(changed)
        })
    }

    pub fn identity_review_veto(&self, pair_handle: &str) -> anyhow::Result<bool> {
        let Some((anchor_identity, candidate_identity)) =
            self.resolve_identity_pair_handle(pair_handle)
        else {
            return Ok(false);
        };
        self.with_db_write_gate(|| {
            let mut store = self.store.lock();
            let changed = store.veto_face_identity_pair_by_id(
                anchor_identity,
                candidate_identity,
                self.embedder.recognition_model_name(),
            )?;
            if changed {
                store.touch_session(self.active.session_id)?;
            }
            Ok(changed)
        })
    }

    pub fn devour_identity_review_refresh(&self) -> anyhow::Result<()> {
        let store = Store::open_hot(&self.db_path)?;
        let beauty = store.compute_identity_beauty_snapshot()?;
        self.with_fresh_store_write(|store| store.save_identity_beauty_snapshot(&beauty))?;
        let store = Store::open_hot(&self.db_path)?;
        self.retrain_face_oracle(&store)?;
        Ok(())
    }

    pub fn identity_anchor_focus_target(&self, anchor_handle: &str) -> Option<String> {
        self.resolve_identity_anchor_handle(anchor_handle)
            .map(|_| format!("/identities?anchor={anchor_handle}"))
    }

    pub fn identity_pair_focus_target(&self, pair_handle: &str) -> Option<String> {
        let (anchor_identity, _) = self.resolve_identity_pair_handle(pair_handle)?;
        Some(format!(
            "/identities?anchor={}",
            self.mint_identity_anchor_handle(anchor_identity)
        ))
    }

    fn identity_match_threshold(&self) -> f32 {
        self.config.read().identity_match_threshold()
    }

    fn mint_identity_anchor_handle(&self, identity_id: FaceIdentityId) -> String {
        self.mint_identity_handle(IDENTITY_HANDLE_TAG_ANCHOR, &[identity_id.0])
    }

    fn mint_identity_pair_handle(
        &self,
        anchor_identity: FaceIdentityId,
        candidate_identity: FaceIdentityId,
    ) -> String {
        self.mint_identity_handle(
            IDENTITY_HANDLE_TAG_PAIR,
            &[anchor_identity.0, candidate_identity.0],
        )
    }

    fn resolve_identity_anchor_handle(&self, raw: &str) -> Option<FaceIdentityId> {
        let values = self.resolve_identity_handle(IDENTITY_HANDLE_TAG_ANCHOR, raw)?;
        let [identity] = values.as_slice() else {
            return None;
        };
        Some(FaceIdentityId(*identity))
    }

    fn resolve_identity_pair_handle(&self, raw: &str) -> Option<(FaceIdentityId, FaceIdentityId)> {
        let values = self.resolve_identity_handle(IDENTITY_HANDLE_TAG_PAIR, raw)?;
        let [anchor, candidate] = values.as_slice() else {
            return None;
        };
        Some((FaceIdentityId(*anchor), FaceIdentityId(*candidate)))
    }

    fn mint_identity_handle(&self, tag: u8, values: &[i64]) -> String {
        let mut payload = Vec::with_capacity(1 + (values.len() * std::mem::size_of::<i64>()));
        payload.push(tag);
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let mac = blake3::keyed_hash(&self.identity_handle_key, &payload);
        payload.extend_from_slice(&mac.as_bytes()[..IDENTITY_HANDLE_MAC_LEN]);
        hex_encode(&payload)
    }

    fn resolve_identity_handle(&self, tag: u8, raw: &str) -> Option<Vec<i64>> {
        let bytes = hex_decode(raw)?;
        let (payload, mac) = bytes.split_at(bytes.len().checked_sub(IDENTITY_HANDLE_MAC_LEN)?);
        if payload.first().copied()? != tag {
            return None;
        }
        let expected = blake3::keyed_hash(&self.identity_handle_key, payload);
        if expected.as_bytes()[..IDENTITY_HANDLE_MAC_LEN] != *mac {
            return None;
        }
        let tail = &payload[1..];
        if tail.len() % std::mem::size_of::<i64>() != 0 {
            return None;
        }
        let mut values = Vec::with_capacity(tail.len() / std::mem::size_of::<i64>());
        for chunk in tail.chunks_exact(std::mem::size_of::<i64>()) {
            values.push(i64::from_le_bytes(chunk.try_into().ok()?));
        }
        Some(values)
    }
}

fn slugify_subject_name(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    let mut pending_dash = false;
    for ch in name.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(ch);
        } else if !slug.is_empty() {
            pending_dash = true;
        }
    }
    if slug.is_empty() {
        "subject".to_owned()
    } else {
        slug
    }
}

fn identity_similarity_radius(threshold: f32) -> f32 {
    (2.0 - (2.0 * threshold.clamp(-1.0, 1.0))).max(0.0).sqrt()
}

fn embedding_dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn canonical_identity_pair(
    left: FaceIdentityId,
    right: FaceIdentityId,
) -> (FaceIdentityId, FaceIdentityId) {
    if left.0 <= right.0 {
        (left, right)
    } else {
        (right, left)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[(byte >> 4) as usize]));
        encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    encoded
}

fn hex_decode(raw: &str) -> Option<Vec<u8>> {
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let raw = raw.as_bytes();
    if raw.len() % 2 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(raw.len() / 2);
    for pair in raw.chunks_exact(2) {
        let hi = nibble(pair[0])?;
        let lo = nibble(pair[1])?;
        bytes.push((hi << 4) | lo);
    }
    Some(bytes)
}
