use super::*;
use parking_lot::{Condvar, Mutex};
use std::{panic::Location, time::Duration as StdDuration};

const DB_WRITE_GATE_WAIT_WARN_MS: u128 = 200;
const DB_WRITE_GATE_WAIT_LOG_STEP_MS: u128 = 1000;
const DB_WRITE_GATE_HOLD_WARN_MS: u128 = 200;

#[derive(Debug, Clone)]
pub(super) struct DbWriteGateOwner {
    file: &'static str,
    line: u32,
    entered_at: Instant,
    thread_name: Option<String>,
}

impl DbWriteGateOwner {
    fn forge(caller: &'static Location<'static>) -> Self {
        Self {
            file: caller.file(),
            line: caller.line(),
            entered_at: Instant::now(),
            thread_name: thread::current().name().map(ToOwned::to_owned),
        }
    }

    fn held_description(&self) -> String {
        let held_ms = self.entered_at.elapsed().as_millis();
        match &self.thread_name {
            Some(name) => format!(
                "{}:{} on thread {name} for {held_ms} ms",
                self.file, self.line
            ),
            None => format!("{}:{} for {held_ms} ms", self.file, self.line),
        }
    }
}

#[derive(Debug, Default)]
struct DbWriteGateState {
    owner: Option<DbWriteGateOwner>,
}

#[derive(Debug, Default)]
pub(super) struct DbWriteGate {
    state: Mutex<DbWriteGateState>,
    changed: Condvar,
}

impl DbWriteGate {
    pub(super) fn new() -> Self {
        Self::default()
    }

    #[track_caller]
    fn acquire(&self) -> DbWriteLease<'_> {
        let caller = Location::caller();
        let wait_started = Instant::now();
        let mut next_wait_log_ms = DB_WRITE_GATE_WAIT_WARN_MS;
        let mut state = self.state.lock();
        while state.owner.is_some() {
            let waited_ms = wait_started.elapsed().as_millis();
            if waited_ms >= next_wait_log_ms {
                let holder = state.owner.as_ref().map_or_else(
                    || "unknown holder".to_owned(),
                    DbWriteGateOwner::held_description,
                );
                warn!(
                    wait_ms = waited_ms,
                    waiter_file = caller.file(),
                    waiter_line = caller.line(),
                    holder = %holder,
                    "waiting on db write gate"
                );
                next_wait_log_ms += DB_WRITE_GATE_WAIT_LOG_STEP_MS;
                continue;
            }
            let wait_for_ms = next_wait_log_ms.saturating_sub(waited_ms).max(1);
            self.changed.wait_for(
                &mut state,
                StdDuration::from_millis(wait_for_ms.min(u128::from(u64::MAX)) as u64),
            );
        }
        let waited = wait_started.elapsed();
        if waited.as_millis() >= DB_WRITE_GATE_WAIT_WARN_MS {
            warn!(
                wait_ms = waited.as_millis(),
                waiter_file = caller.file(),
                waiter_line = caller.line(),
                "acquired db write gate after wait"
            );
        }
        state.owner = Some(DbWriteGateOwner::forge(caller));
        drop(state);
        DbWriteLease {
            gate: self,
            caller_file: caller.file(),
            caller_line: caller.line(),
        }
    }
}

#[must_use]
struct DbWriteLease<'a> {
    gate: &'a DbWriteGate,
    caller_file: &'static str,
    caller_line: u32,
}

impl Drop for DbWriteLease<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.state.lock();
        let owner = state.owner.take();
        let held_ms = owner
            .as_ref()
            .map_or(0, |owner| owner.entered_at.elapsed().as_millis());
        if held_ms >= DB_WRITE_GATE_HOLD_WARN_MS {
            let holder = owner.as_ref().map_or_else(
                || format!("{}:{}", self.caller_file, self.caller_line),
                DbWriteGateOwner::held_description,
            );
            warn!(hold_ms = held_ms, holder = %holder, "held db write gate");
        }
        drop(state);
        self.gate.changed.notify_one();
    }
}

impl AppState {
    #[track_caller]
    pub(super) fn with_db_write_gate<T>(
        &self,
        f: impl FnOnce() -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let _lease = self.db_write_gate.acquire();
        f()
    }

    #[track_caller]
    pub(super) fn with_locked_store_write<T>(
        &self,
        f: impl FnOnce(&mut Store) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        self.with_db_write_gate(|| {
            let mut store = Store::open_hot(&self.db_path)?;
            f(&mut store)
        })
    }

    #[track_caller]
    pub(super) fn with_fresh_store_write<T>(
        &self,
        f: impl FnOnce(&mut Store) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        self.with_db_write_gate(|| {
            let mut store = Store::open_hot(&self.db_path)?;
            f(&mut store)
        })
    }
}
