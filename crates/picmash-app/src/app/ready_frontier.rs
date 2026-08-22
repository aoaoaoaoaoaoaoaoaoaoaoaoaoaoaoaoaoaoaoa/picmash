use std::collections::HashMap;

use time::Duration;

use crate::config::SourceConfig;

const LOCAL_DIRECTORY_READY_TARGET_PER_STREAM: usize = 8;
const LOCAL_DIRECTORY_READY_TARGET_CAP: usize = 64;
const REMOTE_READY_TARGET_PER_STREAM: usize = 3;
const REMOTE_READY_TARGET_CAP: usize = 96;
pub(super) const REMOTE_SOURCE_IDLE_SCAN_GRACE: Duration = Duration::minutes(10);
pub(super) const REMOTE_SOURCE_EMPTY_SCAN_BACKOFF: Duration = Duration::minutes(30);
pub(super) const REMOTE_SOURCE_RECENT_READY_CAP: usize = 12;

#[derive(Debug, Clone, Copy)]
pub(super) struct ReadyTargetProfile {
    per_stream: usize,
    cap: usize,
}

impl ReadyTargetProfile {
    pub(super) fn for_source(source: &SourceConfig) -> Self {
        if source.local_directory().is_some() {
            Self {
                per_stream: LOCAL_DIRECTORY_READY_TARGET_PER_STREAM,
                cap: LOCAL_DIRECTORY_READY_TARGET_CAP,
            }
        } else {
            Self {
                per_stream: REMOTE_READY_TARGET_PER_STREAM,
                cap: REMOTE_READY_TARGET_CAP,
            }
        }
    }

    #[cfg(test)]
    pub(super) const fn for_remote() -> Self {
        Self {
            per_stream: REMOTE_READY_TARGET_PER_STREAM,
            cap: REMOTE_READY_TARGET_CAP,
        }
    }

    pub(super) const fn idle_floor(self) -> usize {
        self.per_stream
    }

    pub(super) fn target_total(self, stream_count: usize) -> usize {
        stream_count
            .saturating_mul(self.per_stream)
            .min(self.cap)
            .max(self.per_stream)
    }

    pub(super) fn capped(self, cap: usize) -> Self {
        Self {
            per_stream: self.per_stream,
            cap: self.cap.min(cap).max(self.per_stream),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SourceReadyFrontier {
    total_ready: usize,
    ready_by_stream: HashMap<i64, usize>,
    target_total: usize,
    target_per_stream: usize,
}

impl SourceReadyFrontier {
    pub(super) fn new(
        total_ready: usize,
        ready_by_stream: HashMap<i64, usize>,
        stream_count: usize,
        profile: ReadyTargetProfile,
    ) -> Self {
        let target_total = profile.target_total(stream_count);
        Self {
            total_ready,
            ready_by_stream,
            target_total,
            target_per_stream: profile.per_stream,
        }
    }

    pub(super) fn source_saturated(&self) -> bool {
        self.total_ready >= self.target_total
    }

    pub(super) fn source_idle_warm(&self) -> bool {
        self.total_ready >= self.target_per_stream
    }

    pub(super) fn stream_saturated(&self, stream_id: i64) -> bool {
        self.ready_by_stream
            .get(&stream_id)
            .copied()
            .unwrap_or_default()
            >= self.target_per_stream
    }

    pub(super) fn ready_in_stream(&self, stream_id: i64) -> usize {
        self.ready_by_stream
            .get(&stream_id)
            .copied()
            .unwrap_or_default()
    }

    pub(super) fn note_ready(&mut self, stream_id: i64) {
        self.total_ready += 1;
        *self.ready_by_stream.entry(stream_id).or_default() += 1;
    }
}
