use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use eternalist_apps::configuration::Configuration;
use serde::{Deserialize, Serialize};

const DEFAULT_CATALOG_THREADS: u16 = 24;
const DEFAULT_THREAD_FETCHES: u8 = 2;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub living_water: bool,
    pub images_per_row: u16,
    pub remote: RemoteConfig,
}

impl Config {
    pub fn legacy_fallback(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("read legacy configuration at {}", path.display()))?;
        let legacy = toml::from_str::<LegacyConfig>(&text)
            .with_context(|| format!("parse legacy configuration at {}", path.display()))?;
        let sources = legacy
            .sources
            .into_iter()
            .map(SourceConfig::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            remote: RemoteConfig {
                enabled: !sources.is_empty(),
                sample_probability: Probability::try_from(legacy.arena.external_probability)?,
                sources,
                ..RemoteConfig::default()
            },
            ..Self::default()
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            living_water: true,
            images_per_row: 5,
            remote: RemoteConfig::default(),
        }
    }
}

impl Configuration for Config {
    fn validate(&self) -> std::result::Result<(), String> {
        if !(1..=12).contains(&self.images_per_row) {
            return Err("images_per_row must lie between 1 and 12".to_owned());
        }
        self.remote.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RemoteConfig {
    pub enabled: bool,
    pub sample_probability: Probability,
    pub reservoir_capacity: ReservoirCapacity,
    pub metadata_capacity: MetadataCapacity,
    pub sources: Vec<SourceConfig>,
}

impl RemoteConfig {
    fn validate(&self) -> std::result::Result<(), String> {
        let enabled_sources = self
            .sources
            .iter()
            .filter(|source| source.enabled())
            .count();
        if self.enabled && enabled_sources == 0 {
            return Err(
                "remote acquisition is enabled but every source has zero weight".to_owned(),
            );
        }
        if self.sources.len() > 32 {
            return Err("at most 32 remote sources are admitted".to_owned());
        }
        if enabled_sources > usize::from(self.metadata_capacity.get()) {
            return Err(
                "remote metadata capacity must admit at least one candidate per enabled source"
                    .to_owned(),
            );
        }
        let mut identities = std::collections::HashSet::new();
        for source in &self.sources {
            source.validate()?;
            let identity = source.identity();
            if !identities.insert(identity.clone()) {
                return Err(format!("duplicate remote source `{identity}`"));
            }
        }
        Ok(())
    }
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_probability: Probability::new(100),
            reservoir_capacity: ReservoirCapacity::default(),
            metadata_capacity: MetadataCapacity::default(),
            sources: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    pub weight: Weight,
    pub import_policy: ImportPolicy,
    pub scan_interval_seconds: ScanInterval,
    pub upstream: Upstream,
}

impl SourceConfig {
    pub fn identity(&self) -> SourceIdentity {
        match &self.upstream {
            Upstream::FourChanBoard(source) => {
                SourceIdentity::new(format!("4chan:{}", source.board))
            }
            Upstream::LocalDirectory(source) => {
                let digest = blake3::hash(source.root.as_os_str().as_encoded_bytes());
                SourceIdentity::new(format!("directory:{}", &digest.to_hex()[..16]))
            }
        }
    }

    pub const fn enabled(&self) -> bool {
        self.weight.quantum() > 0
    }

    fn validate(&self) -> std::result::Result<(), String> {
        match &self.upstream {
            Upstream::FourChanBoard(source) => source.validate(),
            Upstream::LocalDirectory(source) => source.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", content = "settings", rename_all = "snake_case")]
pub enum Upstream {
    #[serde(rename = "4chan_board")]
    FourChanBoard(FourChanBoard),
    LocalDirectory(LocalDirectory),
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourChanBoard {
    pub board: String,
    #[serde(default)]
    pub content: FourChanContent,
    #[serde(default)]
    pub harvest: FourChanHarvest,
    #[serde(default)]
    pub filters: ImageFilter,
}

impl FourChanBoard {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.board.is_empty()
            || !self
                .board
                .chars()
                .all(|glyph| glyph.is_ascii_alphanumeric())
            || self.board.chars().any(|glyph| glyph.is_ascii_uppercase())
        {
            return Err(format!("invalid lowercase 4chan board `{}`", self.board));
        }
        if self.content.allow_video {
            return Err("remote video acquisition is not implemented".to_owned());
        }
        if !self.content.allow_nsfw {
            return Err("4chan does not expose item-level content ratings; allow_nsfw=false cannot be enforced".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalDirectory {
    pub root: PathBuf,
    #[serde(default = "yes")]
    pub recurse: bool,
    #[serde(default)]
    pub filters: ImageFilter,
}

impl LocalDirectory {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.root.as_os_str().is_empty() {
            Err("remote local-directory root cannot be empty".to_owned())
        } else if !self.root.is_absolute() {
            Err("remote local-directory root must be absolute".to_owned())
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FourChanContent {
    pub allow_nsfw: bool,
    pub allow_video: bool,
}

impl Default for FourChanContent {
    fn default() -> Self {
        Self {
            allow_nsfw: true,
            allow_video: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FourChanHarvest {
    pub catalog_threads: CatalogThreadLimit,
    pub thread_fetches_per_scan: ThreadFetchLimit,
}

impl Default for FourChanHarvest {
    fn default() -> Self {
        Self {
            catalog_threads: CatalogThreadLimit(DEFAULT_CATALOG_THREADS),
            thread_fetches_per_scan: ThreadFetchLimit(DEFAULT_THREAD_FETCHES),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageFilter {
    pub min_shortest_edge: ShortestEdge,
    pub max_download_bytes: DownloadLimit,
    pub max_pixels: PixelLimit,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportPolicy {
    #[default]
    NotX,
    Hearted,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceIdentity(String);

impl SourceIdentity {
    pub(crate) fn new(identity: String) -> Self {
        Self(identity)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SourceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Probability(u16);

impl Probability {
    const fn new(permille: u16) -> Self {
        Self(permille)
    }

    pub fn ratio(self) -> f64 {
        f64::from(self.0) / 1_000.0
    }

    pub const fn draw(self, sample: u16) -> bool {
        sample < self.0
    }
}

impl TryFrom<f64> for Probability {
    type Error = anyhow::Error;

    fn try_from(value: f64) -> Result<Self> {
        anyhow::ensure!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "remote sample probability must lie between 0 and 1"
        );
        Ok(Self((value * 1_000.0).round() as u16))
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> Self {
        value.ratio()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Weight(u32);

impl Weight {
    pub const fn quantum(self) -> u32 {
        self.0
    }
}

impl TryFrom<f64> for Weight {
    type Error = anyhow::Error;

    fn try_from(value: f64) -> Result<Self> {
        anyhow::ensure!(
            value.is_finite() && (0.0..=1_000.0).contains(&value),
            "source weight must lie between 0 and 1000"
        );
        Ok(Self((value * 1_000.0).round() as u32))
    }
}

impl From<Weight> for f64 {
    fn from(value: Weight) -> Self {
        Self::from(value.0) / 1_000.0
    }
}

macro_rules! ranged_integer {
    ($name:ident, $raw:ty, $serde_raw:literal, $min:expr, $max:expr, $default:expr) => {
        #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
        #[serde(try_from = $serde_raw, into = $serde_raw)]
        pub struct $name($raw);

        impl $name {
            pub const fn get(self) -> $raw {
                self.0
            }
        }

        impl TryFrom<$raw> for $name {
            type Error = anyhow::Error;

            fn try_from(value: $raw) -> Result<Self> {
                anyhow::ensure!(
                    ($min..=$max).contains(&value),
                    concat!(stringify!($name), " lies outside its admitted range")
                );
                Ok(Self(value))
            }
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self($default)
            }
        }
    };
}

ranged_integer!(ReservoirCapacity, u8, "u8", 1, 8, 4);
ranged_integer!(MetadataCapacity, u16, "u16", 8, 256, 64);
ranged_integer!(ShortestEdge, u32, "u32", 0, 16_384, 800);
ranged_integer!(
    DownloadLimit,
    u64,
    "u64",
    1_024,
    512 * 1024 * 1024,
    8 * 1024 * 1024
);
ranged_integer!(PixelLimit, u64, "u64", 1_000_000, 64_000_000, 40_000_000);
ranged_integer!(
    CatalogThreadLimit,
    u16,
    "u16",
    1,
    64,
    DEFAULT_CATALOG_THREADS
);
ranged_integer!(ThreadFetchLimit, u8, "u8", 1, 4, DEFAULT_THREAD_FETCHES);
ranged_integer!(ScanInterval, u32, "u32", 15, 86_400, 120);

fn yes() -> bool {
    true
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct LegacyConfig {
    arena: LegacyArena,
    sources: Vec<LegacySource>,
}

#[derive(Deserialize)]
#[serde(default)]
struct LegacyArena {
    external_probability: f64,
}

impl Default for LegacyArena {
    fn default() -> Self {
        Self {
            external_probability: 0.1,
        }
    }
}

#[derive(Deserialize)]
struct LegacySource {
    #[serde(default = "one")]
    weight: f64,
    #[serde(default)]
    import_policy: ImportPolicy,
    #[serde(default = "legacy_interval")]
    scan_interval_seconds: u32,
    upstream: LegacyUpstream,
}

#[derive(Deserialize)]
#[serde(tag = "type", content = "settings", rename_all = "snake_case")]
enum LegacyUpstream {
    #[serde(rename = "4chan_board")]
    FourChanBoard(LegacyFourChan),
    LocalDirectory(LegacyLocalDirectory),
}

#[derive(Deserialize)]
struct LegacyFourChan {
    board: String,
    #[serde(default)]
    content: FourChanContent,
    #[serde(default)]
    harvest: LegacyHarvest,
    #[serde(default)]
    filters: ImageFilter,
}

#[derive(Deserialize)]
struct LegacyLocalDirectory {
    root: PathBuf,
    #[serde(default = "yes")]
    recurse: bool,
    #[serde(default)]
    filters: ImageFilter,
}

#[derive(Default, Deserialize)]
struct LegacyHarvest {
    #[serde(default)]
    catalog_threads: u16,
    #[serde(default)]
    thread_fetches_per_scan: u8,
}

impl TryFrom<LegacySource> for SourceConfig {
    type Error = anyhow::Error;

    fn try_from(source: LegacySource) -> Result<Self> {
        let upstream = match source.upstream {
            LegacyUpstream::FourChanBoard(mut board) => {
                board.board.make_ascii_lowercase();
                Upstream::FourChanBoard(FourChanBoard {
                    board: board.board,
                    content: board.content,
                    harvest: FourChanHarvest {
                        catalog_threads: CatalogThreadLimit::try_from(
                            if board.harvest.catalog_threads == 0 {
                                DEFAULT_CATALOG_THREADS
                            } else {
                                board.harvest.catalog_threads
                            },
                        )?,
                        thread_fetches_per_scan: ThreadFetchLimit::try_from(
                            if board.harvest.thread_fetches_per_scan == 0 {
                                DEFAULT_THREAD_FETCHES
                            } else {
                                board.harvest.thread_fetches_per_scan
                            },
                        )?,
                    },
                    filters: board.filters,
                })
            }
            LegacyUpstream::LocalDirectory(directory) => Upstream::LocalDirectory(LocalDirectory {
                root: directory.root,
                recurse: directory.recurse,
                filters: directory.filters,
            }),
        };
        let source = Self {
            weight: Weight::try_from(source.weight)?,
            import_policy: source.import_policy,
            scan_interval_seconds: ScanInterval::try_from(if source.scan_interval_seconds == 0 {
                legacy_interval()
            } else {
                source.scan_interval_seconds
            })?,
            upstream,
        };
        source.validate().map_err(anyhow::Error::msg)?;
        Ok(source)
    }
}

fn one() -> f64 {
    1.0
}

fn legacy_interval() -> u32 {
    120
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    #[test]
    fn legacy_remote_configuration_migrates_without_resurrecting_rejected_models() -> Result<()> {
        let mut legacy = tempfile::NamedTempFile::new()?;
        legacy.write_all(
            br#"
[runtime]
corpus_root = "/discarded"

[arena]
external_probability = 1.0
explore = 0.55
dedup_radius = 0.4

[facemash]
min_face_side = 120.0

[[sources]]
weight = 0.7
import_policy = "not_x"
scan_interval_seconds = 120

[sources.upstream]
type = "4chan_board"

[sources.upstream.settings]
board = "HC"

[sources.upstream.settings.content]
allow_nsfw = true
allow_video = false

[sources.upstream.settings.harvest]
catalog_threads = 0
thread_fetches_per_scan = 0

[sources.upstream.settings.filters]
min_shortest_edge = 800
max_download_bytes = 88388608

[[sources]]
weight = 2.0
import_policy = "hearted"
scan_interval_seconds = 120

[sources.upstream]
type = "local_directory"

[sources.upstream.settings]
root = "/images/mdl"
recurse = true
"#,
        )?;

        let migrated = Config::legacy_fallback(legacy.path())?;

        assert!(migrated.remote.enabled);
        assert_eq!(migrated.remote.sample_probability, Probability::new(1_000));
        assert_eq!(migrated.remote.sources.len(), 2);
        let Upstream::FourChanBoard(board) = &migrated.remote.sources[0].upstream else {
            anyhow::bail!("4chan source changed kind during migration");
        };
        assert_eq!(board.board, "hc");
        assert_eq!(board.harvest.catalog_threads.get(), DEFAULT_CATALOG_THREADS);
        assert_eq!(
            board.harvest.thread_fetches_per_scan.get(),
            DEFAULT_THREAD_FETCHES
        );
        assert_eq!(board.filters.max_pixels, PixelLimit::default());
        assert!(matches!(
            migrated.remote.sources[1].upstream,
            Upstream::LocalDirectory(_)
        ));
        Ok(())
    }
}
