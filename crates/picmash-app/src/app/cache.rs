use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use time::Duration;
use tracing::{info, warn};
use walkdir::WalkDir;

use super::AppState;

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
        let source_retention_paths = self.read_store()?.active_external_source_cache_paths()?;
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
    use std::{collections::HashSet, fs};

    use tempfile::TempDir;

    use super::prune_cache_tree;

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
}
