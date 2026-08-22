use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs,
    fs::File,
    io::Write as _,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context as _, Result, ensure};
use atomic_write_file::AtomicWriteFile;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::ImageDecoder as _;
use jxl_oxide::integration::JxlDecoder;
use picmash_engine::canonical_image;
use serde::Deserialize;
use ureq::{
    Agent,
    tls::{TlsConfig, TlsProvider},
};
use walkdir::WalkDir;

use super::{
    Discovery, FileStamp, Harvest, Origin, Prepared, RemoteItemId, SourceIx, StreamId, slug,
    validate_payload_dimensions,
};
use crate::configuration::{FourChanBoard, LocalDirectory, SourceConfig, Upstream};

const API_ROOT: &str = "https://a.4cdn.org";
const IMAGE_ROOT: &str = "https://i.4cdn.org";
const JSON_LIMIT: u64 = 8 * 1024 * 1024;
const MAX_DISCOVERIES_PER_HARVEST: usize = 64;
const MAX_LOCAL_WALK_ENTRIES: usize = 100_000;

#[derive(Clone)]
pub struct Harvester {
    agent: Agent,
    cache_root: PathBuf,
}

impl Harvester {
    pub fn new(cache_root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_root)
            .with_context(|| format!("create remote cache at {}", cache_root.display()))?;
        let config = Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(4)))
            .timeout_global(Some(Duration::from_secs(12)))
            .max_redirects(2)
            .max_idle_connections_per_host(2)
            .user_agent("picmash/0.1 bounded-remote-acquisition")
            .tls_config(
                TlsConfig::builder()
                    .provider(TlsProvider::NativeTls)
                    .build(),
            )
            .build();
        Ok(Self {
            agent: config.into(),
            cache_root,
        })
    }

    pub fn catalog(
        &self,
        source: SourceIx,
        config: &SourceConfig,
        ordinal: u64,
    ) -> Result<Harvest> {
        match &config.upstream {
            Upstream::FourChanBoard(board) => self.catalog_4chan(source, config, board, ordinal),
            Upstream::LocalDirectory(directory) => {
                self.catalog_directory(source, config, directory, ordinal)
            }
        }
    }

    pub fn fetch(&self, discovery: &Discovery) -> Result<Prepared> {
        let bytes = match &discovery.origin {
            Origin::Network { url, expected_md5 } => {
                let mut response = self
                    .agent
                    .get(url)
                    .call()
                    .with_context(|| format!("fetch remote image {url}"))?;
                let bytes = response
                    .body_mut()
                    .with_config()
                    .limit(discovery.byte_len.saturating_add(1))
                    .read_to_vec()
                    .with_context(|| format!("read remote image {url}"))?;
                ensure!(
                    u64::try_from(bytes.len())? == discovery.byte_len,
                    "remote payload length changed for {}",
                    discovery.item_id
                );
                if let Some(expected) = expected_md5 {
                    let actual = BASE64.encode(md5::compute(&bytes).0);
                    ensure!(
                        actual == *expected,
                        "remote payload digest changed for {}",
                        discovery.item_id
                    );
                }
                bytes
            }
            Origin::Local { path, stamp } => {
                let metadata = fs::metadata(path)
                    .with_context(|| format!("inspect remote local file {}", path.display()))?;
                ensure!(
                    metadata.len() == discovery.byte_len
                        && FileStamp::from_metadata(&metadata) == *stamp,
                    "remote local file changed at {}",
                    path.display()
                );
                let bytes = fs::read(path)
                    .with_context(|| format!("read remote local file {}", path.display()))?;
                ensure!(
                    FileStamp::read(path)? == *stamp,
                    "remote local file changed while reading {}",
                    path.display()
                );
                bytes
            }
        };
        validate_payload_dimensions(
            &bytes,
            &discovery.extension,
            (discovery.width, discovery.height),
            discovery.max_pixels,
        )?;
        let image = canonical_image(&bytes)
            .with_context(|| format!("decode remote candidate {}", discovery.item_id))?;
        ensure!(
            image.width() > 0 && image.height() > 0,
            "remote candidate decoded empty"
        );
        let payload_digest = blake3::hash(&bytes).to_hex().to_string();
        let directory = self
            .cache_root
            .join(slug(discovery.source_identity.as_str()));
        fs::create_dir_all(&directory)
            .with_context(|| format!("create source cache at {}", directory.display()))?;
        let stem = blake3::hash(discovery.item_id.as_str().as_bytes()).to_hex();
        let cache_path = directory.join(format!("{stem}.{}", discovery.extension));
        persist(&cache_path, &bytes)?;
        Ok(Prepared {
            discovery: discovery.clone(),
            cache_path,
            payload_digest,
        })
    }

    fn catalog_4chan(
        &self,
        source: SourceIx,
        config: &SourceConfig,
        board: &FourChanBoard,
        ordinal: u64,
    ) -> Result<Harvest> {
        let url = format!("{API_ROOT}/{}/catalog.json", board.board);
        let mut response = self
            .agent
            .get(&url)
            .call()
            .with_context(|| format!("fetch 4chan catalog {url}"))?;
        let bytes = response
            .body_mut()
            .with_config()
            .limit(JSON_LIMIT)
            .read_to_vec()
            .with_context(|| format!("read 4chan catalog {url}"))?;
        let mut threads = serde_json::from_slice::<Vec<CatalogPage>>(&bytes)
            .with_context(|| format!("decode 4chan catalog {url}"))?
            .into_iter()
            .flat_map(|page| page.threads)
            .filter(|thread| {
                thread.no > 0
                    && thread.images > 0
                    && thread.replies > 0
                    && !thread.closed
                    && !thread.sticky
            })
            .collect::<Vec<_>>();
        threads.sort_by_key(|thread| Reverse((thread.last_modified, thread.images, thread.no)));
        threads.truncate(usize::from(board.harvest.catalog_threads.get()));
        let fetches = usize::from(board.harvest.thread_fetches_per_scan.get()).min(threads.len());
        let start = if threads.is_empty() {
            0
        } else {
            usize::try_from(ordinal % u64::try_from(threads.len())?)?
        };
        let mut discoveries = Vec::new();
        for thread in threads.iter().cycle().skip(start).take(fetches) {
            let url = format!("{API_ROOT}/{}/thread/{}.json", board.board, thread.no);
            let mut response = self
                .agent
                .get(&url)
                .call()
                .with_context(|| format!("fetch 4chan thread {url}"))?;
            let bytes = response
                .body_mut()
                .with_config()
                .limit(JSON_LIMIT)
                .read_to_vec()
                .with_context(|| format!("read 4chan thread {url}"))?;
            let thread = serde_json::from_slice::<Thread>(&bytes)
                .with_context(|| format!("decode 4chan thread {url}"))?;
            let stream_title = thread
                .posts
                .first()
                .map(post_title)
                .unwrap_or_else(|| "UNTITLED THREAD".to_owned());
            for post in thread.posts.into_iter().rev() {
                let Some(extension) = post.extension.as_deref().and_then(image_extension) else {
                    continue;
                };
                let (Some(timestamp), Some(md5)) = (post.timestamp, post.md5.clone()) else {
                    continue;
                };
                let (Ok(width), Ok(height), Ok(byte_len)) = (
                    u32::try_from(post.width),
                    u32::try_from(post.height),
                    u64::try_from(post.byte_len),
                ) else {
                    continue;
                };
                if width.min(height) < board.filters.min_shortest_edge.get()
                    || post.no <= 0
                    || timestamp <= 0
                    || byte_len == 0
                    || byte_len > board.filters.max_download_bytes.get()
                    || u64::from(width).saturating_mul(u64::from(height))
                        > board.filters.max_pixels.get()
                {
                    continue;
                }
                discoveries.push(Discovery {
                    source,
                    source_identity: config.identity(),
                    source_name: format!("4CHAN /{}/", board.board.to_ascii_uppercase()),
                    import_policy: config.import_policy,
                    stream_id: StreamId::new(format!("4chan:{}:{}", board.board, thread_no(&post))),
                    stream_title: stream_title.clone(),
                    item_id: RemoteItemId::new(format!("4chan:{}:{}", board.board, post.no)),
                    title: post_title(&post),
                    origin: Origin::Network {
                        url: format!("{IMAGE_ROOT}/{}/{}.{extension}", board.board, timestamp),
                        expected_md5: Some(md5),
                    },
                    extension: extension.to_owned(),
                    width,
                    height,
                    byte_len,
                    max_pixels: board.filters.max_pixels.get(),
                });
                if discoveries.len() == MAX_DISCOVERIES_PER_HARVEST {
                    break;
                }
            }
            if discoveries.len() == MAX_DISCOVERIES_PER_HARVEST {
                break;
            }
        }
        Ok(Harvest {
            source,
            source_identity: config.identity(),
            discoveries,
        })
    }

    fn catalog_directory(
        &self,
        source: SourceIx,
        config: &SourceConfig,
        directory: &LocalDirectory,
        ordinal: u64,
    ) -> Result<Harvest> {
        ensure!(
            directory.root.is_dir(),
            "remote directory is unavailable: {}",
            directory.root.display()
        );
        let depth = if directory.recurse { usize::MAX } else { 1 };
        let mut selection = BinaryHeap::<RankedPath>::new();
        for entry in WalkDir::new(&directory.root)
            .follow_links(false)
            .max_depth(depth)
            .into_iter()
            .take(MAX_LOCAL_WALK_ENTRIES)
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.into_path();
            let Some(extension) = path
                .extension()
                .and_then(|extension| extension.to_str())
                .and_then(image_extension)
            else {
                continue;
            };
            let Ok(metadata) = fs::metadata(&path) else {
                continue;
            };
            let stamp = FileStamp::from_metadata(&metadata);
            if metadata.len() == 0 || metadata.len() > directory.filters.max_download_bytes.get() {
                continue;
            }
            let relative = path.strip_prefix(&directory.root).unwrap_or(&path);
            let mut ranker = blake3::Hasher::new();
            ranker.update(&ordinal.to_le_bytes());
            ranker.update(relative.as_os_str().as_encoded_bytes());
            let mut rank_bytes = [0; 8];
            rank_bytes.copy_from_slice(&ranker.finalize().as_bytes()[..8]);
            let ranked = RankedPath {
                rank: u64::from_le_bytes(rank_bytes),
                path,
                extension: extension.to_owned(),
                byte_len: metadata.len(),
                stamp,
            };
            selection.push(ranked);
            if selection.len() > MAX_DISCOVERIES_PER_HARVEST {
                let _discarded = selection.pop();
            }
        }
        let mut ranked = selection.into_sorted_vec();
        ranked.reverse();
        let mut discoveries = Vec::with_capacity(ranked.len());
        for candidate in ranked {
            let Some((width, height)) = image_dimensions(&candidate.path) else {
                continue;
            };
            if FileStamp::read(&candidate.path).ok() != Some(candidate.stamp) {
                continue;
            }
            if width.min(height) < directory.filters.min_shortest_edge.get()
                || u64::from(width).saturating_mul(u64::from(height))
                    > directory.filters.max_pixels.get()
            {
                continue;
            }
            let relative = candidate
                .path
                .strip_prefix(&directory.root)
                .unwrap_or(&candidate.path);
            let parent = relative.parent().unwrap_or_else(|| Path::new("."));
            let stream_digest = blake3::hash(parent.as_os_str().as_encoded_bytes());
            let mut item_hasher = blake3::Hasher::new();
            item_hasher.update(config.identity().as_str().as_bytes());
            item_hasher.update(relative.as_os_str().as_encoded_bytes());
            item_hasher.update(candidate.stamp.as_bytes());
            let item_digest = item_hasher.finalize();
            discoveries.push(Discovery {
                source,
                source_identity: config.identity(),
                source_name: directory.root.file_name().map_or_else(
                    || directory.root.display().to_string(),
                    |name| name.to_string_lossy().into_owned(),
                ),
                import_policy: config.import_policy,
                stream_id: StreamId::new(format!("directory:{}", &stream_digest.to_hex()[..20])),
                stream_title: parent.display().to_string(),
                item_id: RemoteItemId::new(format!("directory:{}", &item_digest.to_hex()[..24])),
                title: candidate.path.file_name().map_or_else(
                    || candidate.path.display().to_string(),
                    |name| name.to_string_lossy().into_owned(),
                ),
                origin: Origin::Local {
                    path: candidate.path,
                    stamp: candidate.stamp,
                },
                extension: candidate.extension,
                width,
                height,
                byte_len: candidate.byte_len,
                max_pixels: directory.filters.max_pixels.get(),
            });
        }
        Ok(Harvest {
            source,
            source_identity: config.identity(),
            discoveries,
        })
    }
}

#[derive(Eq, PartialEq)]
struct RankedPath {
    rank: u64,
    path: PathBuf,
    extension: String,
    byte_len: u64,
    stamp: FileStamp,
}

impl Ord for RankedPath {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank
            .cmp(&other.rank)
            .then_with(|| self.path.cmp(&other.path))
    }
}

impl PartialOrd for RankedPath {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn persist(target: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = AtomicWriteFile::open(target)
        .with_context(|| format!("stage remote cache at {}", target.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write remote cache at {}", target.display()))?;
    file.commit()
        .with_context(|| format!("commit remote cache at {}", target.display()))
}

fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jxl"))
    {
        let decoder = JxlDecoder::new(File::open(path).ok()?).ok()?;
        return Some(decoder.dimensions());
    }
    image::image_dimensions(path).ok()
}

fn image_extension(extension: &str) -> Option<&'static str> {
    match extension
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "bmp" => Some("bmp"),
        "gif" => Some("gif"),
        "jpeg" | "jpg" => Some("jpg"),
        "jxl" => Some("jxl"),
        "png" => Some("png"),
        "webp" => Some("webp"),
        _ => None,
    }
}

#[derive(Deserialize)]
struct CatalogPage {
    threads: Vec<CatalogThread>,
}

#[derive(Deserialize)]
struct CatalogThread {
    no: i64,
    #[serde(default, deserialize_with = "boolish")]
    sticky: bool,
    #[serde(default, deserialize_with = "boolish")]
    closed: bool,
    last_modified: i64,
    #[serde(default)]
    replies: i64,
    #[serde(default)]
    images: i64,
}

#[derive(Deserialize)]
struct Thread {
    posts: Vec<Post>,
}

#[derive(Deserialize)]
struct Post {
    no: i64,
    #[serde(default)]
    resto: i64,
    #[serde(default)]
    sub: Option<String>,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default, rename = "tim")]
    timestamp: Option<i64>,
    #[serde(default, rename = "ext")]
    extension: Option<String>,
    #[serde(default, rename = "fsize")]
    byte_len: i64,
    #[serde(default, rename = "w")]
    width: i64,
    #[serde(default, rename = "h")]
    height: i64,
    #[serde(default)]
    md5: Option<String>,
}

fn boolish<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<i64>::deserialize(deserializer)?.unwrap_or(0) != 0)
}

fn thread_no(post: &Post) -> i64 {
    if post.resto == 0 { post.no } else { post.resto }
}

fn post_title(post: &Post) -> String {
    post.filename
        .as_deref()
        .or(post.sub.as_deref())
        .filter(|title| !title.is_empty())
        .map_or_else(
            || format!("POST {}", post.no),
            |title| title.chars().take(160).collect(),
        )
}
