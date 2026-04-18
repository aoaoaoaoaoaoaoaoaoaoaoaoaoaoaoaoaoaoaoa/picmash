use crate::{
    model::{RemoteItemRecord, SessionId, SessionSubsourceLock},
    store::Store,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ThreadKey {
    source_key: String,
    stream_id: i64,
}

impl ThreadKey {
    fn new(source_key: impl Into<String>, stream_id: i64) -> Self {
        Self {
            source_key: source_key.into(),
            stream_id,
        }
    }

    fn to_lock(&self) -> SessionSubsourceLock {
        SessionSubsourceLock {
            source_key: self.source_key.clone(),
            stream_id: self.stream_id,
        }
    }
}

impl From<&RemoteItemRecord> for ThreadKey {
    fn from(item: &RemoteItemRecord) -> Self {
        Self::new(item.source_key.clone(), item.stream_id)
    }
}

impl From<SessionSubsourceLock> for ThreadKey {
    fn from(lock: SessionSubsourceLock) -> Self {
        Self::new(lock.source_key, lock.stream_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ArenaScope {
    Global,
    LockedThread(ThreadKey),
}

impl ArenaScope {
    pub(super) fn reduce(&self, action: ArenaScopeAction) -> ArenaScopeTransition {
        use ArenaScopeAction::{Lock, Unlock, Veto};
        match (self, action) {
            (Self::LockedThread(current), Lock(thread)) if *current == thread => {
                ArenaScopeTransition::preserve(self.clone(), ScopeWrite::Preserve)
            }
            (Self::Global | Self::LockedThread(_), Lock(thread)) => ArenaScopeTransition::reset(
                Self::LockedThread(thread.clone()),
                ScopeWrite::Set(thread),
            ),
            (Self::Global, Unlock) => {
                ArenaScopeTransition::preserve(Self::Global, ScopeWrite::Preserve)
            }
            (Self::LockedThread(_), Unlock) => {
                ArenaScopeTransition::reset(Self::Global, ScopeWrite::Clear)
            }
            (Self::Global, Veto(_)) => {
                ArenaScopeTransition::reset(Self::Global, ScopeWrite::Preserve)
            }
            (Self::LockedThread(current), Veto(thread)) if *current == thread => {
                ArenaScopeTransition::reset(Self::Global, ScopeWrite::Clear)
            }
            (Self::LockedThread(current), Veto(_)) => ArenaScopeTransition::reset(
                Self::LockedThread(current.clone()),
                ScopeWrite::Preserve,
            ),
        }
    }
}

impl From<Option<SessionSubsourceLock>> for ArenaScope {
    fn from(lock: Option<SessionSubsourceLock>) -> Self {
        lock.map(ThreadKey::from)
            .map_or(Self::Global, Self::LockedThread)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ArenaScopeAction {
    Lock(ThreadKey),
    Unlock,
    Veto(ThreadKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PipelineDisposition {
    Preserve,
    Reset,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ScopeWrite {
    Preserve,
    Set(ThreadKey),
    Clear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ArenaScopeTransition {
    next_scope: ArenaScope,
    pipeline: PipelineDisposition,
    write: ScopeWrite,
}

impl ArenaScopeTransition {
    fn preserve(next_scope: ArenaScope, write: ScopeWrite) -> Self {
        Self {
            next_scope,
            pipeline: PipelineDisposition::Preserve,
            write,
        }
    }

    fn reset(next_scope: ArenaScope, write: ScopeWrite) -> Self {
        Self {
            next_scope,
            pipeline: PipelineDisposition::Reset,
            write,
        }
    }

    pub(super) const fn pipeline(&self) -> PipelineDisposition {
        self.pipeline
    }

    pub(super) fn next_lock(&self) -> Option<SessionSubsourceLock> {
        match &self.next_scope {
            ArenaScope::Global => None,
            ArenaScope::LockedThread(thread) => Some(thread.to_lock()),
        }
    }

    pub(super) fn clears_lock(&self) -> bool {
        matches!(self.write, ScopeWrite::Clear)
    }

    pub(super) fn apply(&self, store: &Store, session_id: SessionId) -> anyhow::Result<()> {
        match &self.write {
            ScopeWrite::Preserve => Ok(()),
            ScopeWrite::Set(thread) => {
                store.set_session_subsource_lock(session_id, &thread.source_key, thread.stream_id)
            }
            ScopeWrite::Clear => store.clear_session_subsource_lock(session_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread(source_key: &str, stream_id: i64) -> ThreadKey {
        ThreadKey::new(source_key, stream_id)
    }

    #[test]
    fn scope_reducer_exhausts_lock_unlock_veto() {
        let a = thread("4chan:s", 10);
        let b = thread("4chan:s", 11);

        assert_eq!(
            ArenaScope::Global.reduce(ArenaScopeAction::Lock(a.clone())),
            ArenaScopeTransition::reset(
                ArenaScope::LockedThread(a.clone()),
                ScopeWrite::Set(a.clone()),
            )
        );
        assert_eq!(
            ArenaScope::LockedThread(a.clone()).reduce(ArenaScopeAction::Unlock),
            ArenaScopeTransition::reset(ArenaScope::Global, ScopeWrite::Clear)
        );
        assert_eq!(
            ArenaScope::Global.reduce(ArenaScopeAction::Unlock),
            ArenaScopeTransition::preserve(ArenaScope::Global, ScopeWrite::Preserve)
        );
        assert_eq!(
            ArenaScope::Global.reduce(ArenaScopeAction::Veto(a.clone())),
            ArenaScopeTransition::reset(ArenaScope::Global, ScopeWrite::Preserve)
        );
        assert_eq!(
            ArenaScope::LockedThread(a.clone()).reduce(ArenaScopeAction::Veto(a.clone())),
            ArenaScopeTransition::reset(ArenaScope::Global, ScopeWrite::Clear)
        );
        assert_eq!(
            ArenaScope::LockedThread(a.clone()).reduce(ArenaScopeAction::Veto(b.clone())),
            ArenaScopeTransition::reset(ArenaScope::LockedThread(a.clone()), ScopeWrite::Preserve,)
        );
        assert_eq!(
            ArenaScope::LockedThread(a.clone()).reduce(ArenaScopeAction::Lock(a.clone())),
            ArenaScopeTransition::preserve(
                ArenaScope::LockedThread(a.clone()),
                ScopeWrite::Preserve,
            )
        );
        assert_eq!(
            ArenaScope::LockedThread(a).reduce(ArenaScopeAction::Lock(b.clone())),
            ArenaScopeTransition::reset(ArenaScope::LockedThread(b.clone()), ScopeWrite::Set(b),)
        );
    }
}
