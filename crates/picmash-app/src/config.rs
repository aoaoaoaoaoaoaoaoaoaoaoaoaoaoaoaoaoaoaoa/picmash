use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

const DEFAULT_EXTERNAL_PROBABILITY: f32 = 0.10;
const MAX_EXTERNAL_PROBABILITY: f32 = 1.0;
const DEFAULT_ARENA_EXPLORE: f32 = 0.35;
const MAX_ARENA_EXPLORE: f32 = 1.0;
const DEFAULT_DEDUP_RADIUS: f32 = 0.35;
const MAX_DEDUP_RADIUS: f32 = 1.0;
const DEFAULT_FACEMASH_MIN_FACE_SIDE: f32 = 80.0;
const MIN_FACEMASH_MIN_FACE_SIDE: f32 = 24.0;
const MAX_FACEMASH_MIN_FACE_SIDE: f32 = 256.0;
const DEFAULT_IDENTITY_MATCH_THRESHOLD: f32 = 0.35;
const MIN_IDENTITY_MATCH_THRESHOLD: f32 = -1.0;
const MAX_IDENTITY_MATCH_THRESHOLD: f32 = 1.0;
const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8788";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub arena: ArenaConfig,
    #[serde(default)]
    pub facemash: FacemashConfig,
    #[serde(default)]
    pub identities: IdentitiesConfig,
    #[serde(default = "default_sources")]
    pub sources: Vec<SourceConfig>,
}

impl AppConfig {
    pub fn load_or_init(config_dir: &Path) -> anyhow::Result<(Self, PathBuf, String)> {
        fs::create_dir_all(config_dir)
            .with_context(|| format!("creating config directory {}", config_dir.display()))?;
        let config_path = config_dir.join("config.toml");
        if config_path.exists() {
            let (parsed, digest) = Self::load(&config_path)?;
            return Ok((parsed, config_path, digest));
        }

        let config = Self::default();
        let digest = config
            .write(&config_path)
            .with_context(|| format!("writing default {}", config_path.display()))?;
        Ok((config, config_path, digest))
    }

    pub fn load(path: &Path) -> anyhow::Result<(Self, String)> {
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&raw, path)
    }

    pub fn parse(raw: &str, path: &Path) -> anyhow::Result<(Self, String)> {
        let parsed =
            toml::from_str::<Self>(raw).with_context(|| format!("parsing {}", path.display()))?;
        let normalized = parsed.normalized();
        normalized
            .validate()
            .with_context(|| format!("validating {}", path.display()))?;
        Ok((normalized, digest_text(raw)))
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<String> {
        let encoded =
            toml::to_string_pretty(&self.clone().normalized()).context("encoding app config")?;
        write_atomically(path, encoded.as_bytes())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(digest_text(&encoded))
    }

    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.runtime.bind_addr = normalize_bind_addr(&self.runtime.bind_addr);
        self.arena.external_probability =
            clamp_external_probability(self.arena.external_probability);
        self.arena.explore = clamp_arena_explore(self.arena.explore);
        self.arena.dedup_radius = clamp_dedup_radius(self.arena.dedup_radius);
        self.facemash.min_face_side = clamp_facemash_min_face_side(self.facemash.min_face_side);
        self.identities.match_threshold =
            clamp_identity_match_threshold(self.identities.match_threshold);
        if self.sources.is_empty() {
            self.sources = default_sources();
        }
        for source in &mut self.sources {
            source.normalize();
        }
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.bind_addr()
            .parse::<SocketAddr>()
            .with_context(|| format!("parsing runtime.bind_addr `{}`", self.bind_addr()))?;
        if let Some(root) = self.corpus_root()
            && root.as_os_str().is_empty()
        {
            bail!("runtime.corpus_root is empty");
        }
        for source in &self.sources {
            source.validate()?;
        }
        Ok(())
    }

    #[must_use]
    pub fn external_probability(&self) -> f32 {
        clamp_external_probability(self.arena.external_probability)
    }

    #[must_use]
    pub fn dedup_radius(&self) -> f32 {
        clamp_dedup_radius(self.arena.dedup_radius)
    }

    #[must_use]
    pub fn arena_explore(&self) -> f32 {
        clamp_arena_explore(self.arena.explore)
    }

    #[must_use]
    pub fn bind_addr(&self) -> &str {
        &self.runtime.bind_addr
    }

    #[must_use]
    pub fn facemash_min_face_side(&self) -> f32 {
        clamp_facemash_min_face_side(self.facemash.min_face_side)
    }

    #[must_use]
    pub fn identity_match_threshold(&self) -> f32 {
        clamp_identity_match_threshold(self.identities.match_threshold)
    }

    #[must_use]
    pub fn corpus_root(&self) -> Option<&Path> {
        self.runtime.corpus_root.as_deref()
    }

    pub fn shove_external_probability(&mut self, probability: f32) {
        self.arena.external_probability = clamp_external_probability(probability);
    }

    pub fn shove_arena_explore(&mut self, explore: f32) {
        self.arena.explore = clamp_arena_explore(explore);
    }

    pub fn shove_dedup_radius(&mut self, radius: f32) {
        self.arena.dedup_radius = clamp_dedup_radius(radius);
    }

    pub fn shove_bind_addr(&mut self, bind_addr: impl Into<String>) {
        self.runtime.bind_addr = normalize_bind_addr(&bind_addr.into());
    }

    pub fn shove_facemash_min_face_side(&mut self, min_face_side: f32) {
        self.facemash.min_face_side = clamp_facemash_min_face_side(min_face_side);
    }

    pub fn shove_identity_match_threshold(&mut self, threshold: f32) {
        self.identities.match_threshold = clamp_identity_match_threshold(threshold);
    }

    pub fn shove_corpus_root(&mut self, root: PathBuf) {
        self.runtime.corpus_root = Some(root);
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            arena: ArenaConfig::default(),
            facemash: FacemashConfig::default(),
            identities: IdentitiesConfig::default(),
            sources: default_sources(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,
    #[serde(default)]
    pub corpus_root: Option<PathBuf>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_bind_addr(),
            corpus_root: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaConfig {
    #[serde(default = "default_external_probability")]
    pub external_probability: f32,
    #[serde(default = "default_arena_explore")]
    pub explore: f32,
    #[serde(default = "default_dedup_radius")]
    pub dedup_radius: f32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            external_probability: DEFAULT_EXTERNAL_PROBABILITY,
            explore: DEFAULT_ARENA_EXPLORE,
            dedup_radius: DEFAULT_DEDUP_RADIUS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacemashConfig {
    #[serde(default = "default_facemash_min_face_side")]
    pub min_face_side: f32,
}

impl Default for FacemashConfig {
    fn default() -> Self {
        Self {
            min_face_side: DEFAULT_FACEMASH_MIN_FACE_SIDE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentitiesConfig {
    #[serde(default = "default_identity_match_threshold")]
    pub match_threshold: f32,
}

impl Default for IdentitiesConfig {
    fn default() -> Self {
        Self {
            match_threshold: DEFAULT_IDENTITY_MATCH_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    #[serde(default = "default_source_weight")]
    pub weight: f32,
    #[serde(default)]
    pub import_policy: ImportPolicy,
    #[serde(default = "default_scan_interval_seconds")]
    pub scan_interval_seconds: u64,
    pub upstream: UpstreamSource,
}

impl SourceConfig {
    #[must_use]
    pub fn source_key(&self) -> String {
        self.upstream.source_key()
    }

    #[must_use]
    pub fn display_name(&self) -> String {
        self.upstream.display_name()
    }

    #[must_use]
    pub fn source_type_name(&self) -> &'static str {
        self.upstream.source_type_name()
    }

    #[must_use]
    pub fn source_locator(&self) -> String {
        self.upstream.source_locator()
    }

    #[must_use]
    pub fn four_chan_board(&self) -> Option<&FourChanBoardSource> {
        self.upstream.four_chan_board()
    }

    #[must_use]
    pub fn local_directory(&self) -> Option<&LocalDirectorySource> {
        self.upstream.local_directory()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.weight < 0.0 || !self.weight.is_finite() {
            bail!("invalid source weight: {}", self.weight);
        }
        self.upstream.validate()
    }

    fn normalize(&mut self) {
        self.weight = normalize_source_weight(self.weight);
        self.upstream.normalize();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "settings", rename_all = "snake_case")]
pub enum UpstreamSource {
    #[serde(rename = "4chan_board")]
    FourChanBoard(FourChanBoardSource),
    #[serde(rename = "local_directory")]
    LocalDirectory(LocalDirectorySource),
}

impl UpstreamSource {
    #[must_use]
    pub fn source_key(&self) -> String {
        match self {
            Self::FourChanBoard(source) => format!("4chan:{}", source.board),
            Self::LocalDirectory(source) => source.source_key(),
        }
    }

    #[must_use]
    pub fn display_name(&self) -> String {
        match self {
            Self::FourChanBoard(_) => self.source_key(),
            Self::LocalDirectory(source) => source.display_name(),
        }
    }

    #[must_use]
    pub const fn source_type_name(&self) -> &'static str {
        match self {
            Self::FourChanBoard(_) => "4chan_board",
            Self::LocalDirectory(_) => "local_directory",
        }
    }

    #[must_use]
    pub fn source_locator(&self) -> String {
        match self {
            Self::FourChanBoard(source) => source.board.clone(),
            Self::LocalDirectory(source) => source.root.to_string_lossy().into_owned(),
        }
    }

    #[must_use]
    pub fn four_chan_board(&self) -> Option<&FourChanBoardSource> {
        match self {
            Self::FourChanBoard(source) => Some(source),
            Self::LocalDirectory(_) => None,
        }
    }

    #[must_use]
    pub fn local_directory(&self) -> Option<&LocalDirectorySource> {
        match self {
            Self::LocalDirectory(source) => Some(source),
            Self::FourChanBoard(_) => None,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::FourChanBoard(source) => source.validate(),
            Self::LocalDirectory(source) => source.validate(),
        }
    }

    fn normalize(&mut self) {
        match self {
            Self::FourChanBoard(source) => source.normalize(),
            Self::LocalDirectory(source) => source.normalize(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourChanBoardSource {
    pub board: String,
    #[serde(default)]
    pub content: FourChanContentConfig,
    #[serde(default)]
    pub harvest: FourChanHarvestConfig,
    #[serde(default)]
    pub filters: RemoteImageFilterConfig,
}

impl FourChanBoardSource {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.board.is_empty()
            || !self
                .board
                .chars()
                .all(|glyph| glyph.is_ascii_alphanumeric())
        {
            bail!("invalid 4chan board name: {}", self.board);
        }
        if self.content.allow_video {
            bail!("video upstreams are not implemented yet");
        }
        Ok(())
    }

    fn normalize(&mut self) {
        if self.board.chars().any(|glyph| glyph.is_ascii_uppercase()) {
            self.board.make_ascii_lowercase();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDirectorySource {
    pub root: PathBuf,
    #[serde(default = "default_local_recurse")]
    pub recurse: bool,
    #[serde(default)]
    pub filters: RemoteImageFilterConfig,
}

impl LocalDirectorySource {
    #[must_use]
    pub fn source_key(&self) -> String {
        let digest = blake3::hash(self.root.to_string_lossy().as_bytes()).to_hex();
        format!("local:{}", &digest[..16])
    }

    #[must_use]
    pub fn display_name(&self) -> String {
        let label = self
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.root.to_string_lossy().into_owned());
        format!("dir:{label}")
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.root.as_os_str().is_empty() {
            bail!("local directory source root is empty");
        }
        if !self.root.exists() {
            bail!("local directory source missing: {}", self.root.display());
        }
        if !self.root.is_dir() {
            bail!(
                "local directory source is not a directory: {}",
                self.root.display()
            );
        }
        Ok(())
    }

    fn normalize(&mut self) {
        if let Ok(root) = self.root.canonicalize() {
            self.root = root;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourChanContentConfig {
    #[serde(default = "default_allow_nsfw")]
    pub allow_nsfw: bool,
    #[serde(default = "default_allow_video")]
    pub allow_video: bool,
}

impl Default for FourChanContentConfig {
    fn default() -> Self {
        Self {
            allow_nsfw: default_allow_nsfw(),
            allow_video: default_allow_video(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourChanHarvestConfig {
    #[serde(default = "default_catalog_threads")]
    pub catalog_threads: usize,
    #[serde(default = "default_thread_fetches_per_scan")]
    pub thread_fetches_per_scan: usize,
}

impl FourChanHarvestConfig {
    #[must_use]
    pub const fn catalog_thread_cap(&self) -> Option<usize> {
        nonzero_limit(self.catalog_threads)
    }

    #[must_use]
    pub const fn thread_fetch_cap(&self) -> Option<usize> {
        nonzero_limit(self.thread_fetches_per_scan)
    }
}

impl Default for FourChanHarvestConfig {
    fn default() -> Self {
        Self {
            catalog_threads: default_catalog_threads(),
            thread_fetches_per_scan: default_thread_fetches_per_scan(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteImageFilterConfig {
    #[serde(default = "default_min_shortest_edge")]
    pub min_shortest_edge: u32,
    #[serde(default = "default_max_download_bytes")]
    pub max_download_bytes: u64,
}

impl Default for RemoteImageFilterConfig {
    fn default() -> Self {
        Self {
            min_shortest_edge: default_min_shortest_edge(),
            max_download_bytes: default_max_download_bytes(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImportPolicy {
    #[default]
    NotX,
    Hearted,
}

fn clamp_external_probability(probability: f32) -> f32 {
    if probability.is_finite() {
        probability.clamp(0.0, MAX_EXTERNAL_PROBABILITY)
    } else {
        DEFAULT_EXTERNAL_PROBABILITY
    }
}

fn clamp_arena_explore(explore: f32) -> f32 {
    if explore.is_finite() {
        explore.clamp(0.0, MAX_ARENA_EXPLORE)
    } else {
        DEFAULT_ARENA_EXPLORE
    }
}

fn clamp_dedup_radius(radius: f32) -> f32 {
    if radius.is_finite() {
        radius.clamp(0.0, MAX_DEDUP_RADIUS)
    } else {
        DEFAULT_DEDUP_RADIUS
    }
}

fn clamp_facemash_min_face_side(min_face_side: f32) -> f32 {
    if min_face_side.is_finite() {
        min_face_side.clamp(MIN_FACEMASH_MIN_FACE_SIDE, MAX_FACEMASH_MIN_FACE_SIDE)
    } else {
        DEFAULT_FACEMASH_MIN_FACE_SIDE
    }
}

fn clamp_identity_match_threshold(threshold: f32) -> f32 {
    if threshold.is_finite() {
        threshold.clamp(MIN_IDENTITY_MATCH_THRESHOLD, MAX_IDENTITY_MATCH_THRESHOLD)
    } else {
        DEFAULT_IDENTITY_MATCH_THRESHOLD
    }
}

const fn default_dedup_radius() -> f32 {
    DEFAULT_DEDUP_RADIUS
}

const fn default_arena_explore() -> f32 {
    DEFAULT_ARENA_EXPLORE
}

const fn default_facemash_min_face_side() -> f32 {
    DEFAULT_FACEMASH_MIN_FACE_SIDE
}

const fn default_identity_match_threshold() -> f32 {
    DEFAULT_IDENTITY_MATCH_THRESHOLD
}

fn normalize_bind_addr(bind_addr: &str) -> String {
    let trimmed = bind_addr.trim();
    if trimmed.is_empty() {
        DEFAULT_BIND_ADDR.to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn default_sources() -> Vec<SourceConfig> {
    vec![SourceConfig {
        weight: default_source_weight(),
        import_policy: ImportPolicy::NotX,
        scan_interval_seconds: default_scan_interval_seconds(),
        upstream: UpstreamSource::FourChanBoard(FourChanBoardSource {
            board: "s".to_owned(),
            content: FourChanContentConfig::default(),
            harvest: FourChanHarvestConfig::default(),
            filters: RemoteImageFilterConfig::default(),
        }),
    }]
}

const fn default_external_probability() -> f32 {
    DEFAULT_EXTERNAL_PROBABILITY
}

fn default_bind_addr() -> String {
    DEFAULT_BIND_ADDR.to_owned()
}

fn digest_text(raw: &str) -> String {
    blake3::hash(raw.as_bytes()).to_hex().to_string()
}

fn write_atomically(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("resolving parent directory for {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating parent directory {}", parent.display()))?;
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.toml");
    let temp_path = parent.join(format!(".{stem}.{}.tmp", Ulid::new()));
    fs::write(&temp_path, bytes)
        .with_context(|| format!("writing temporary config {}", temp_path.display()))?;
    fs::rename(&temp_path, path)
        .or_else(|rename_error| {
            let copy_result = fs::copy(&temp_path, path)
                .map(|_| ())
                .and_then(|_| fs::remove_file(&temp_path));
            copy_result.map_err(|_| rename_error)
        })
        .with_context(|| {
            format!(
                "replacing config atomically from {} into {}",
                temp_path.display(),
                path.display()
            )
        })
}

const fn default_allow_nsfw() -> bool {
    true
}

const fn default_source_weight() -> f32 {
    1.0
}

const fn default_allow_video() -> bool {
    false
}

const fn default_local_recurse() -> bool {
    true
}

const fn default_scan_interval_seconds() -> u64 {
    120
}

const fn default_catalog_threads() -> usize {
    0
}

const fn default_thread_fetches_per_scan() -> usize {
    0
}

const fn default_min_shortest_edge() -> u32 {
    800
}

const fn default_max_download_bytes() -> u64 {
    8 * 1024 * 1024
}

fn normalize_source_weight(weight: f32) -> f32 {
    if weight.is_finite() && weight >= 0.0 {
        weight
    } else {
        default_source_weight()
    }
}

const fn nonzero_limit(limit: usize) -> Option<usize> {
    if limit == 0 { None } else { Some(limit) }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        AppConfig, ArenaConfig, FacemashConfig, FourChanBoardSource, FourChanContentConfig,
        FourChanHarvestConfig, IdentitiesConfig, ImportPolicy, RemoteImageFilterConfig,
        RuntimeConfig, SourceConfig, UpstreamSource,
    };
    #[test]
    fn explicit_scan_budget_round_trips() {
        let config = AppConfig {
            runtime: RuntimeConfig::default(),
            arena: ArenaConfig::default(),
            facemash: FacemashConfig::default(),
            identities: IdentitiesConfig::default(),
            sources: vec![SourceConfig {
                weight: 1.0,
                import_policy: ImportPolicy::NotX,
                scan_interval_seconds: 120,
                upstream: UpstreamSource::FourChanBoard(FourChanBoardSource {
                    board: "wg".to_owned(),
                    content: FourChanContentConfig::default(),
                    harvest: FourChanHarvestConfig {
                        catalog_threads: 20,
                        thread_fetches_per_scan: 7,
                    },
                    filters: RemoteImageFilterConfig::default(),
                }),
            }],
        }
        .normalized();
        let source = config.sources[0]
            .four_chan_board()
            .expect("4chan config present");
        assert_eq!(source.harvest.catalog_thread_cap(), Some(20));
        assert_eq!(source.harvest.thread_fetch_cap(), Some(7));
    }

    #[test]
    fn blank_bind_addr_snaps_back_to_default() {
        let config = AppConfig {
            runtime: RuntimeConfig {
                bind_addr: "   ".to_owned(),
                corpus_root: None,
            },
            arena: ArenaConfig::default(),
            facemash: FacemashConfig::default(),
            identities: IdentitiesConfig::default(),
            sources: AppConfig::default().sources,
        }
        .normalized();
        assert_eq!(config.bind_addr(), "127.0.0.1:8788");
    }

    #[test]
    fn nested_upstream_source_shape_parses() {
        let raw = r#"
[runtime]
bind_addr = "127.0.0.1:8788"
corpus_root = "/tmp/picmash"

[arena]
external_probability = 0.4

[[sources]]
weight = 1.0
import_policy = "not_x"
scan_interval_seconds = 120

[sources.upstream]
type = "4chan_board"

[sources.upstream.settings]
board = "s"

[sources.upstream.settings.content]
allow_nsfw = true
allow_video = false

[sources.upstream.settings.harvest]
catalog_threads = 0
thread_fetches_per_scan = 0

[sources.upstream.settings.filters]
min_shortest_edge = 800
max_download_bytes = 88388608
"#;
        let parsed = toml::from_str::<AppConfig>(raw).expect("parse nested config");
        let source = parsed.sources[0]
            .four_chan_board()
            .expect("4chan source present");
        assert_eq!(source.board, "s");
        assert!(source.content.allow_nsfw);
        assert_eq!(source.filters.max_download_bytes, 88_388_608);
    }

    #[test]
    fn multiple_nested_sources_round_trip() {
        let raw = r#"
[[sources]]
weight = 1.0
import_policy = "not_x"
scan_interval_seconds = 120

[sources.upstream]
type = "4chan_board"

[sources.upstream.settings]
board = "s"

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
weight = 0.7
import_policy = "not_x"
scan_interval_seconds = 120

[sources.upstream]
type = "4chan_board"

[sources.upstream.settings]
board = "hc"

[sources.upstream.settings.content]
allow_nsfw = true
allow_video = false

[sources.upstream.settings.harvest]
catalog_threads = 0
thread_fetches_per_scan = 0

[sources.upstream.settings.filters]
min_shortest_edge = 800
max_download_bytes = 88388608
"#;
        let parsed = toml::from_str::<AppConfig>(raw).expect("parse multiple nested sources");
        assert_eq!(parsed.sources.len(), 2);
        assert_eq!(
            parsed.sources[0]
                .four_chan_board()
                .expect("first 4chan source")
                .board,
            "s"
        );
        assert_eq!(
            parsed.sources[1]
                .four_chan_board()
                .expect("second 4chan source")
                .board,
            "hc"
        );
    }

    #[test]
    fn local_directory_source_round_trips() {
        let raw = r#"
[[sources]]
weight = 1.0
import_policy = "not_x"
scan_interval_seconds = 600

[sources.upstream]
type = "local_directory"

[sources.upstream.settings]
root = "/tmp/picmash-dump"
recurse = true

[sources.upstream.settings.filters]
min_shortest_edge = 640
max_download_bytes = 16777216
"#;
        let parsed = toml::from_str::<AppConfig>(raw).expect("parse local source");
        let source = parsed.sources[0]
            .local_directory()
            .expect("local source present");
        assert_eq!(source.root, PathBuf::from("/tmp/picmash-dump"));
        assert!(source.recurse);
        assert_eq!(source.filters.min_shortest_edge, 640);
    }
}
