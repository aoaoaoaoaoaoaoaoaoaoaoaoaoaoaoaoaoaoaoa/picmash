use super::*;

#[derive(Debug, Clone)]
pub enum RedirectTarget {
    ArenaRoot,
    ArenaPair {
        left: ArenaHandle,
        right: ArenaHandle,
    },
    FacemashRoot,
    FacemashPair {
        left_face_id: FaceId,
        right_face_id: FaceId,
    },
    ExploreRoot {
        map_mode: ExploreMapMode,
    },
    ExploreTriad {
        asset_a: AssetId,
        asset_b: AssetId,
        asset_c: AssetId,
        focus_id: Option<AssetId>,
        map_mode: ExploreMapMode,
    },
}

impl RedirectTarget {
    #[must_use]
    pub fn href(&self) -> String {
        match self {
            Self::ArenaRoot => "/arena".to_owned(),
            Self::ArenaPair { left, right } => format!("/arena/{}/{}", left.slug(), right.slug()),
            Self::FacemashRoot => "/facemash".to_owned(),
            Self::FacemashPair {
                left_face_id,
                right_face_id,
            } => format!("/facemash/{}/{}", left_face_id.0, right_face_id.0),
            Self::ExploreRoot { map_mode } => format!("/explore?mode={}", map_mode.as_str()),
            Self::ExploreTriad {
                asset_a,
                asset_b,
                asset_c,
                focus_id,
                map_mode,
            } => {
                let mut query = format!(
                    "mode={}&triad={},{},{}",
                    map_mode.as_str(),
                    asset_a.0,
                    asset_b.0,
                    asset_c.0
                );
                if let Some(focus_id) = focus_id {
                    query.push_str("&focus=");
                    query.push_str(&focus_id.0);
                }
                format!("/explore?{query}")
            }
        }
    }
}

pub(super) fn surviving_local_anchor<'a>(
    rejected: &ArenaHandle,
    pair_left: &'a ArenaHandle,
    pair_right: &'a ArenaHandle,
) -> Option<&'a AssetId> {
    match (rejected, pair_left, pair_right) {
        (ArenaHandle::Remote(_), ArenaHandle::Local(asset_id), ArenaHandle::Remote(_))
        | (ArenaHandle::Remote(_), ArenaHandle::Remote(_), ArenaHandle::Local(asset_id)) => {
            Some(asset_id)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimePhase {
    Loading,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSnapshot {
    pub phase: RuntimePhase,
    pub message: Option<String>,
}

enum RuntimeStateInner {
    Loading,
    Ready(SharedAppState),
    Failed(String),
}

pub struct RuntimeState {
    inner: RwLock<RuntimeStateInner>,
}

impl RuntimeState {
    #[must_use]
    pub fn loading() -> Self {
        Self {
            inner: RwLock::new(RuntimeStateInner::Loading),
        }
    }

    pub fn install_ready(&self, state: SharedAppState) {
        *self.inner.write() = RuntimeStateInner::Ready(state);
    }

    pub fn install_failed(&self, message: String) {
        *self.inner.write() = RuntimeStateInner::Failed(message);
    }

    #[must_use]
    pub fn ready_app(&self) -> Option<SharedAppState> {
        match &*self.inner.read() {
            RuntimeStateInner::Ready(state) => Some(state.clone()),
            RuntimeStateInner::Loading | RuntimeStateInner::Failed(_) => None,
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> RuntimeSnapshot {
        match &*self.inner.read() {
            RuntimeStateInner::Loading => RuntimeSnapshot {
                phase: RuntimePhase::Loading,
                message: None,
            },
            RuntimeStateInner::Ready(_) => RuntimeSnapshot {
                phase: RuntimePhase::Ready,
                message: None,
            },
            RuntimeStateInner::Failed(message) => RuntimeSnapshot {
                phase: RuntimePhase::Failed,
                message: Some(message.clone()),
            },
        }
    }

    pub fn close_ready(&self) -> anyhow::Result<()> {
        let ready = match &*self.inner.read() {
            RuntimeStateInner::Ready(state) => Some(state.clone()),
            RuntimeStateInner::Loading | RuntimeStateInner::Failed(_) => None,
        };
        if let Some(state) = ready {
            state.close()?;
        }
        Ok(())
    }
}

pub type SharedRuntimeState = Arc<RuntimeState>;
