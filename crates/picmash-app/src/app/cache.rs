use std::{
    collections::{HashSet, VecDeque},
    env, fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use time::Duration;
use tracing::{info, warn};
use walkdir::WalkDir;

use super::{
    AppState, REMOTE_SOURCE_IDLE_SCAN_GRACE, REMOTE_SOURCE_RECENT_READY_CAP, ReadyTargetProfile,
};
use crate::store::{ExternalReadyCachePath, Store};

const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_RENDITION_CACHE_MAX_BYTES: u64 = GIB;
const DEFAULT_SOURCE_CACHE_MAX_BYTES: u64 = GIB;
const DEFAULT_CACHE_PRUNE_INTERVAL_SECONDS: u64 = 60;
const CACHE_PRUNE_TARGET_NUMERATOR: u64 = 9;
const CACHE_PRUNE_TARGET_DENOMINATOR: u64 = 10;
#[derive(Debug)]
struct CacheEntry {
    path: PathBuf,
    bytes: u64,
    last_used: SystemTime,
    protected: bool,
}

#[derive(Debug, Default)]
struct CachePruneSummary {
    max_bytes: u64,
    target_bytes: u64,
    bytes_before: u64,
    bytes_after: u64,
    files_seen: usize,
    files_removed: usize,
    bytes_removed: u64,
    removed_paths: Vec<PathBuf>,
}

impl AppState {
    pub fn cache_prune_interval(&self) -> Duration {
        Duration::seconds(
            i64::try_from(env_u64(
                "PICMASH_CACHE_PRUNE_INTERVAL_SECONDS",
                DEFAULT_CACHE_PRUNE_INTERVAL_SECONDS,
            ))
            .unwrap_or(300)
            .max(10),
        )
    }

    pub fn prune_disk_caches(&self) -> anyhow::Result<()> {
        let empty = HashSet::new();
        let renditions = prune_cache_tree(
            "renditions",
            &self.cache_root,
            env_u64(
                "PICMASH_RENDITION_CACHE_MAX_BYTES",
                DEFAULT_RENDITION_CACHE_MAX_BYTES,
            ),
            &empty,
        )?;
        let source_retention_paths = self.source_cache_retention_paths()?;
        let sources = prune_cache_tree(
            "sources",
            &self.source_cache_root,
            env_u64(
                "PICMASH_SOURCE_CACHE_MAX_BYTES",
                DEFAULT_SOURCE_CACHE_MAX_BYTES,
            ),
            &source_retention_paths,
        )?;
        if !sources.removed_paths.is_empty() {
            let removed_paths = sources.removed_paths.clone();
            let withdrawn = self.with_write_store("withdraw_pruned_source_cache_items", {
                move |store| store.withdraw_external_items_by_cached_paths(&removed_paths)
            })?;
            if withdrawn > 0 {
                self.purge_duplicate_frontier();
                info!(
                    withdrawn,
                    "withdrew pruned source-cache paths from external frontier"
                );
            }
        }
        log_prune_summary("renditions", &self.cache_root, &renditions);
        log_prune_summary("sources", &self.source_cache_root, &sources);
        Ok(())
    }

    fn source_cache_retention_paths(&self) -> anyhow::Result<HashSet<PathBuf>> {
        let store = self.read_store()?;
        let mut retained = HashSet::new();
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        let recent_since = now - REMOTE_SOURCE_IDLE_SCAN_GRACE.whole_seconds();
        let active_lock = store.session_subsource_lock(self.active.session_id)?;

        for source in self.configured_sources() {
            if source.local_directory().is_some() {
                continue;
            }
            let source_key = source.source_key();
            let ready_paths =
                store.external_source_ready_paths(&source_key, self.embedder.model_name())?;
            if ready_paths.is_empty() {
                continue;
            }

            let selected = if let Some(lock) = active_lock.as_ref() {
                if lock.source_key == source_key {
                    let limit = store
                        .external_stream_frontier_counts(
                            &lock.source_key,
                            lock.stream_id,
                            self.embedder.model_name(),
                        )?
                        .map_or(crate::app::EXTERNAL_LOCKED_STREAM_READY_TARGET, |counts| {
                            Self::locked_stream_ready_target(counts.live_items)
                        });
                    lock_retention_paths(ready_paths, lock.stream_id, limit)
                } else {
                    let limit = self.remote_source_retention_limit(
                        &store,
                        &source_key,
                        ReadyTargetProfile::for_source(&source),
                        recent_since,
                    )?;
                    round_robin_retention_paths(ready_paths, limit)
                }
            } else {
                let limit = self.remote_source_retention_limit(
                    &store,
                    &source_key,
                    ReadyTargetProfile::for_source(&source),
                    recent_since,
                )?;
                round_robin_retention_paths(ready_paths, limit)
            };
            retained.extend(selected);
        }
        Ok(retained)
    }

    fn remote_source_retention_limit(
        &self,
        store: &Store,
        source_key: &str,
        profile: ReadyTargetProfile,
        recent_since: i64,
    ) -> anyhow::Result<usize> {
        let (active_streams, _, _) = store.external_source_counts(source_key)?;
        Ok(
            if store.external_source_recently_selected(
                self.active.session_id,
                source_key,
                recent_since,
            )? {
                profile
                    .target_total(active_streams)
                    .min(REMOTE_SOURCE_RECENT_READY_CAP)
                    .max(profile.idle_floor())
            } else {
                profile.idle_floor()
            },
        )
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn prune_target_bytes(max_bytes: u64) -> u64 {
    max_bytes.saturating_mul(CACHE_PRUNE_TARGET_NUMERATOR) / CACHE_PRUNE_TARGET_DENOMINATOR
}

fn prune_cache_tree(
    label: &str,
    root: &Path,
    max_bytes: u64,
    protected_paths: &HashSet<PathBuf>,
) -> anyhow::Result<CachePruneSummary> {
    let target_bytes = prune_target_bytes(max_bytes);
    let mut summary = CachePruneSummary {
        max_bytes,
        target_bytes,
        ..CachePruneSummary::default()
    };
    if !root.exists() {
        return Ok(summary);
    }

    let mut entries = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                warn!(cache = label, root = %root.display(), error = %error, "failed to walk cache entry");
                continue;
            }
        };
        if !entry.file_type().is_file() || is_active_cache_temp(entry.path()) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) => {
                warn!(cache = label, path = %entry.path().display(), error = %error, "failed to stat cache file");
                continue;
            }
        };
        let bytes = metadata.len();
        let last_used = metadata
            .accessed()
            .or_else(|_| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        summary.bytes_before = summary.bytes_before.saturating_add(bytes);
        summary.files_seen += 1;
        entries.push(CacheEntry {
            path: entry.path().to_path_buf(),
            bytes,
            last_used,
            protected: protected_paths.contains(entry.path()),
        });
    }
    summary.bytes_after = summary.bytes_before;
    if summary.bytes_after <= max_bytes {
        return Ok(summary);
    }

    entries.sort_by(|lhs, rhs| {
        lhs.protected
            .cmp(&rhs.protected)
            .then_with(|| lhs.last_used.cmp(&rhs.last_used))
            .then_with(|| lhs.path.cmp(&rhs.path))
    });
    for entry in entries {
        if summary.bytes_after <= target_bytes {
            break;
        }
        match fs::remove_file(&entry.path) {
            Ok(()) => {
                summary.files_removed += 1;
                summary.bytes_removed = summary.bytes_removed.saturating_add(entry.bytes);
                summary.bytes_after = summary.bytes_after.saturating_sub(entry.bytes);
                summary.removed_paths.push(entry.path);
            }
            Err(error) => {
                warn!(cache = label, path = %entry.path.display(), error = %error, "failed to remove cache file");
            }
        }
    }
    remove_empty_cache_dirs(label, root)?;
    Ok(summary)
}

fn lock_retention_paths(
    ready_paths: Vec<ExternalReadyCachePath>,
    locked_stream_id: i64,
    limit: usize,
) -> HashSet<PathBuf> {
    ready_paths
        .into_iter()
        .filter(|entry| entry.stream_id == locked_stream_id)
        .map(|entry| entry.path)
        .take(limit)
        .collect()
}

fn round_robin_retention_paths(
    ready_paths: Vec<ExternalReadyCachePath>,
    limit: usize,
) -> HashSet<PathBuf> {
    if limit == 0 {
        return HashSet::new();
    }
    let mut streams = Vec::<(i64, VecDeque<PathBuf>)>::new();
    for entry in ready_paths {
        if let Some((_, queued)) = streams
            .iter_mut()
            .find(|(stream_id, _)| *stream_id == entry.stream_id)
        {
            queued.push_back(entry.path);
        } else {
            streams.push((entry.stream_id, VecDeque::from([entry.path])));
        }
    }
    let mut retained = HashSet::new();
    while retained.len() < limit {
        let mut advanced = false;
        for (_, queued) in &mut streams {
            if retained.len() >= limit {
                break;
            }
            if let Some(path) = queued.pop_front() {
                retained.insert(path);
                advanced = true;
            }
        }
        if !advanced {
            break;
        }
    }
    retained
}

fn is_active_cache_temp(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.starts_with(".tmp")
                || path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp"))
        })
}

fn remove_empty_cache_dirs(label: &str, root: &Path) -> anyhow::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(root)
        .contents_first(true)
        .min_depth(1)
        .follow_links(false)
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                warn!(cache = label, root = %root.display(), error = %error, "failed to walk cache directory for cleanup");
                continue;
            }
        };
        if entry.file_type().is_dir()
            && let Err(error) = fs::remove_dir(entry.path())
            && error.kind() != std::io::ErrorKind::DirectoryNotEmpty
            && error.kind() != std::io::ErrorKind::NotFound
        {
            warn!(cache = label, path = %entry.path().display(), error = %error, "failed to remove empty cache directory");
        }
    }
    Ok(())
}

fn log_prune_summary(label: &str, root: &Path, summary: &CachePruneSummary) {
    if summary.files_removed > 0 {
        info!(
            cache = label,
            root = %root.display(),
            max_bytes = summary.max_bytes,
            target_bytes = summary.target_bytes,
            bytes_before = summary.bytes_before,
            bytes_after = summary.bytes_after,
            files_seen = summary.files_seen,
            files_removed = summary.files_removed,
            bytes_removed = summary.bytes_removed,
            "cache pruned"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs, path::PathBuf};

    use tempfile::TempDir;

    use crate::store::ExternalReadyCachePath;

    use super::{
        REMOTE_SOURCE_RECENT_READY_CAP, ReadyTargetProfile, lock_retention_paths, prune_cache_tree,
        round_robin_retention_paths,
    };

    fn write_file(root: &TempDir, name: &str, len: usize) {
        fs::write(root.path().join(name), vec![b'x'; len]).expect("write cache fixture");
    }

    #[test]
    fn prune_cache_tree_reduces_stable_files_below_target() {
        let root = TempDir::new().expect("tempdir");
        write_file(&root, "a.cache", 10);
        write_file(&root, "b.cache", 10);
        write_file(&root, "c.cache", 10);

        let summary =
            prune_cache_tree("test", root.path(), 20, &HashSet::new()).expect("prune cache");

        assert_eq!(summary.bytes_before, 30);
        assert!(summary.bytes_after <= 18);
        assert!(summary.files_removed >= 2);
        assert_eq!(summary.removed_paths.len(), summary.files_removed);
    }

    #[test]
    fn prune_cache_tree_skips_active_temp_files() {
        let root = TempDir::new().expect("tempdir");
        write_file(&root, ".tmp-active", 100);
        write_file(&root, "stable.cache", 100);

        let summary =
            prune_cache_tree("test", root.path(), 10, &HashSet::new()).expect("prune cache");

        assert_eq!(summary.bytes_before, 100);
        assert_eq!(summary.files_removed, 1);
        assert!(root.path().join(".tmp-active").exists());
    }

    #[test]
    fn prune_cache_tree_spares_protected_paths_first() {
        let root = TempDir::new().expect("tempdir");
        let protected = root.path().join("live.cache");
        let dead_one = root.path().join("dead-a.cache");
        let dead_two = root.path().join("dead-b.cache");
        write_file(&root, "live.cache", 10);
        write_file(&root, "dead-a.cache", 10);
        write_file(&root, "dead-b.cache", 10);

        let summary =
            prune_cache_tree("test", root.path(), 20, &HashSet::from([protected.clone()]))
                .expect("prune cache");

        assert!(summary.bytes_after <= 18);
        assert!(protected.exists());
        assert!(!dead_one.exists());
        assert!(!dead_two.exists());
    }

    #[test]
    fn round_robin_retention_spreads_across_streams_before_doubling_up() {
        let retained = round_robin_retention_paths(
            vec![
                ExternalReadyCachePath {
                    stream_id: 10,
                    path: PathBuf::from("10-a"),
                },
                ExternalReadyCachePath {
                    stream_id: 10,
                    path: PathBuf::from("10-b"),
                },
                ExternalReadyCachePath {
                    stream_id: 20,
                    path: PathBuf::from("20-a"),
                },
                ExternalReadyCachePath {
                    stream_id: 20,
                    path: PathBuf::from("20-b"),
                },
                ExternalReadyCachePath {
                    stream_id: 30,
                    path: PathBuf::from("30-a"),
                },
            ],
            4,
        );

        assert_eq!(retained.len(), 4);
        assert!(retained.contains(&PathBuf::from("10-a")));
        assert!(retained.contains(&PathBuf::from("20-a")));
        assert!(retained.contains(&PathBuf::from("30-a")));
    }

    #[test]
    fn lock_retention_only_keeps_the_locked_stream() {
        let retained = lock_retention_paths(
            vec![
                ExternalReadyCachePath {
                    stream_id: 10,
                    path: PathBuf::from("10-a"),
                },
                ExternalReadyCachePath {
                    stream_id: 11,
                    path: PathBuf::from("11-a"),
                },
                ExternalReadyCachePath {
                    stream_id: 10,
                    path: PathBuf::from("10-b"),
                },
            ],
            10,
            8,
        );

        assert_eq!(
            retained,
            HashSet::from([PathBuf::from("10-a"), PathBuf::from("10-b")])
        );
    }

    #[test]
    fn recent_remote_retention_cap_stays_above_idle_floor() {
        let profile = ReadyTargetProfile::for_remote();
        assert_eq!(
            profile
                .target_total(128)
                .min(REMOTE_SOURCE_RECENT_READY_CAP)
                .max(profile.idle_floor()),
            REMOTE_SOURCE_RECENT_READY_CAP
        );
    }
}
