use std::{
    cmp::Reverse,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Duration as StdDuration,
};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use html_escape::decode_html_entities;
use image::{ImageDecoder, image_dimensions};
use jxl_oxide::integration::JxlDecoder;
use md5::compute as md5_digest;
use reqwest::blocking::Client;
use serde::{Deserialize, Deserializer};
use tempfile::NamedTempFile;
use tracing::info;
use walkdir::WalkDir;

use crate::{
    config::{FourChanBoardSource, LocalDirectorySource, SourceConfig, UpstreamSource},
    identity::decode_image,
};

const API_ROOT: &str = "https://a.4cdn.org";
const IMAGE_ROOT: &str = "https://i.4cdn.org";
const USER_AGENT: &str = "picmash/0.1";

#[derive(Debug, Clone)]
pub struct SourceScanner {
    client: Client,
    cache_root: PathBuf,
}

impl SourceScanner {
    pub fn new(cache_root: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(cache_root)
            .with_context(|| format!("creating source cache root {}", cache_root.display()))?;
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(12))
            .connect_timeout(StdDuration::from_secs(4))
            .build()
            .context("building source scanner client")?;
        Ok(Self {
            client,
            cache_root: cache_root.to_path_buf(),
        })
    }

    pub fn harvest(&self, source: &SourceConfig) -> anyhow::Result<SourceHarvest> {
        source.validate()?;
        match &source.upstream {
            UpstreamSource::FourChanBoard(board_source) => {
                self.harvest_4chan_board(source, board_source)
            }
            UpstreamSource::LocalDirectory(dir_source) => {
                self.harvest_local_directory(source, dir_source)
            }
        }
    }

    fn harvest_4chan_board(
        &self,
        source: &SourceConfig,
        board_source: &FourChanBoardSource,
    ) -> anyhow::Result<SourceHarvest> {
        let board = board_source.board.as_str();
        let catalog_url = format!("{API_ROOT}/{board}/catalog.json");
        let catalog = self
            .client
            .get(&catalog_url)
            .send()
            .with_context(|| format!("fetching 4chan catalog {catalog_url}"))?
            .error_for_status()
            .with_context(|| format!("reading 4chan catalog {catalog_url}"))?
            .json::<Vec<FourChanCatalogPage>>()
            .with_context(|| format!("decoding 4chan catalog {catalog_url}"))?;

        let mut thread_index = catalog
            .into_iter()
            .flat_map(|page| page.threads)
            .filter(|thread| {
                thread.replies > 0
                    && thread.images > 0
                    && !thread.closed
                    && !thread.sticky
                    && thread.no > 0
            })
            .collect::<Vec<_>>();
        thread_index.sort_by(|lhs, rhs| {
            rhs.last_modified
                .cmp(&lhs.last_modified)
                .then_with(|| rhs.images.cmp(&lhs.images))
                .then_with(|| rhs.no.cmp(&lhs.no))
        });
        let discovered_threads = thread_index.len();
        if let Some(cap) = board_source.harvest.catalog_thread_cap() {
            thread_index.truncate(cap);
        }
        let selected_threads = thread_index.len();

        let fetch_budget = board_source
            .harvest
            .thread_fetch_cap()
            .unwrap_or(selected_threads);
        let fetch_count = selected_threads.min(fetch_budget);
        let mut streams = Vec::with_capacity(fetch_count);
        for thread_stub in thread_index.into_iter().take(fetch_count) {
            let thread_url = format!("{API_ROOT}/{board}/thread/{}.json", thread_stub.no);
            let thread = self
                .client
                .get(&thread_url)
                .send()
                .with_context(|| format!("fetching 4chan thread {thread_url}"))?
                .error_for_status()
                .with_context(|| format!("reading 4chan thread {thread_url}"))?
                .json::<FourChanThreadResponse>()
                .with_context(|| format!("decoding 4chan thread {thread_url}"))?;
            let title = thread_title(&thread);

            let mut items = thread
                .posts
                .into_iter()
                .filter_map(|post| RemoteItemSnapshot::from_post(board, thread_stub.no, post))
                .filter(|item| {
                    supported_ext(&item.ext, board_source.content.allow_video)
                        && item.file_size <= board_source.filters.max_download_bytes
                        && item.shortest_edge() >= board_source.filters.min_shortest_edge
                })
                .collect::<Vec<_>>();
            items.sort_by_key(|item| Reverse(item.post_no));
            if items.is_empty() {
                continue;
            }

            streams.push(RemoteStreamSnapshot {
                thread_no: thread_stub.no,
                title,
                semantic_slug: String::new(),
                last_modified: thread_stub.last_modified,
                reply_count: thread_stub.replies.max(0) as u32,
                image_count: thread_stub.images.max(0) as u32,
                items,
            });
        }

        info!(
            source = %source.source_key(),
            board,
            discovered_threads,
            selected_threads,
            fetched_threads = fetch_count,
            "4chan catalog harvest budget"
        );

        Ok(SourceHarvest {
            source_key: source.source_key(),
            display_name: source.display_name(),
            streams,
        })
    }

    fn harvest_local_directory(
        &self,
        source: &SourceConfig,
        dir_source: &LocalDirectorySource,
    ) -> anyhow::Result<SourceHarvest> {
        let root = dir_source.root.as_path();
        let max_depth = if dir_source.recurse { usize::MAX } else { 1 };
        let mut grouped = std::collections::BTreeMap::<String, Vec<RemoteItemSnapshot>>::new();
        let mut stream_meta = std::collections::HashMap::<String, (i64, u32)>::new();
        let mut discovered_files = 0usize;

        for entry in WalkDir::new(root)
            .follow_links(false)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.into_path();
            let ext = normalized_extension(&path);
            if !supported_ext(&format!(".{ext}"), false) {
                continue;
            }
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            let file_size = metadata.len();
            if file_size > dir_source.filters.max_download_bytes {
                continue;
            }
            let (width, height) = match sniff_image_dimensions(&path) {
                Some(dimensions) => dimensions,
                None => continue,
            };
            if width.min(height) < dir_source.filters.min_shortest_edge {
                continue;
            }

            let relative = path.strip_prefix(root).unwrap_or(&path);
            let stream_rel = relative
                .parent()
                .map(relative_component)
                .unwrap_or_else(|| ".".to_owned());
            let modified = metadata
                .modified()
                .ok()
                .and_then(|ts| ts.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|ts| i64::try_from(ts.as_secs()).unwrap_or_default())
                .unwrap_or_default();
            let stream_id = hashed_local_id(&stream_rel);
            let post_id = hashed_local_id(&relative_component(relative));
            let title = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|stem| !stem.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("file {post_id}"));
            let image_url = format!("file://{}", path.display());

            grouped
                .entry(stream_rel.clone())
                .or_default()
                .push(RemoteItemSnapshot {
                    thread_no: stream_id,
                    post_no: post_id,
                    title,
                    image_url: image_url.clone(),
                    thumb_url: image_url,
                    ext: format!(".{ext}"),
                    md5: None,
                    width,
                    height,
                    file_size,
                    materialized_path: Some(path),
                });
            stream_meta
                .entry(stream_rel)
                .and_modify(|(last_modified, count)| {
                    *last_modified = (*last_modified).max(modified);
                    *count += 1;
                })
                .or_insert((modified, 1));
            discovered_files += 1;
        }

        let streams = grouped
            .into_iter()
            .filter_map(|(stream_rel, mut items)| {
                items.sort_by_key(|item| Reverse(item.post_no));
                let (last_modified, image_count) = stream_meta.get(&stream_rel).copied()?;
                Some(RemoteStreamSnapshot {
                    thread_no: hashed_local_id(&stream_rel),
                    title: if stream_rel == "." {
                        dir_source.display_name()
                    } else {
                        stream_rel.clone()
                    },
                    semantic_slug: stream_rel,
                    last_modified,
                    reply_count: image_count,
                    image_count,
                    items,
                })
            })
            .collect::<Vec<_>>();

        info!(
            source = %source.source_key(),
            root = %dir_source.root.display(),
            discovered_files,
            streams = streams.len(),
            "local directory harvest complete"
        );

        Ok(SourceHarvest {
            source_key: source.source_key(),
            display_name: source.display_name(),
            streams,
        })
    }

    pub fn cache_remote_image(
        &self,
        source_key: &str,
        item: &RemoteItemSnapshot,
    ) -> anyhow::Result<PathBuf> {
        if let Some(path) = &item.materialized_path {
            return Ok(path.clone());
        }
        let ext = item.ext.trim_start_matches('.');
        let shard = format!(
            "{}-{:02}",
            sanitize_source_key(source_key),
            item.post_no % 97
        );
        let source_dir = self.cache_root.join(shard);
        fs::create_dir_all(&source_dir)
            .with_context(|| format!("creating remote cache directory {}", source_dir.display()))?;
        let cache_path = source_dir.join(format!(
            "{}-{}-{}.{}",
            sanitize_source_key(source_key),
            item.thread_no,
            item.post_no,
            ext
        ));
        if cached_remote_image_is_sound(&cache_path, item)? {
            return Ok(cache_path);
        }
        if cache_path.exists() {
            fs::remove_file(&cache_path)
                .with_context(|| format!("removing stale remote cache {}", cache_path.display()))?;
        }

        let bytes = self
            .client
            .get(&item.image_url)
            .send()
            .with_context(|| format!("fetching remote image {}", item.image_url))?
            .error_for_status()
            .with_context(|| format!("reading remote image {}", item.image_url))?
            .bytes()
            .with_context(|| format!("collecting remote image {}", item.image_url))?;
        ensure_remote_payload_matches_snapshot(bytes.as_ref(), item).with_context(|| {
            format!(
                "validating fetched remote image {} against source snapshot",
                item.image_url
            )
        })?;
        persist_remote_cache_atomically(&source_dir, &cache_path, bytes.as_ref())?;
        Ok(cache_path)
    }
}

fn cached_remote_image_is_sound(path: &Path, item: &RemoteItemSnapshot) -> anyhow::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if item.file_size > 0 {
        let cached_len = fs::metadata(path)
            .with_context(|| format!("reading remote cache metadata {}", path.display()))?
            .len();
        if cached_len != item.file_size {
            return Ok(false);
        }
    }
    if item.md5.is_none() {
        return Ok(true);
    }
    let bytes = fs::read(path)
        .with_context(|| format!("reading cached remote image {}", path.display()))?;
    Ok(remote_payload_matches_snapshot(&bytes, item))
}

fn ensure_remote_payload_matches_snapshot(
    bytes: &[u8],
    item: &RemoteItemSnapshot,
) -> anyhow::Result<()> {
    if remote_payload_matches_snapshot(bytes, item) {
        Ok(())
    } else {
        anyhow::bail!("remote payload does not match source snapshot");
    }
}

fn remote_payload_matches_snapshot(bytes: &[u8], item: &RemoteItemSnapshot) -> bool {
    if item.file_size > 0 && u64::try_from(bytes.len()).ok() != Some(item.file_size) {
        return false;
    }
    item.md5.as_ref().is_none_or(|expected| {
        let actual = BASE64_STANDARD.encode(md5_digest(bytes).0);
        actual == *expected
    })
}

fn persist_remote_cache_atomically(
    source_dir: &Path,
    cache_path: &Path,
    bytes: &[u8],
) -> anyhow::Result<()> {
    let mut temp = NamedTempFile::new_in(source_dir)
        .with_context(|| format!("allocating temp cache file in {}", source_dir.display()))?;
    temp.write_all(bytes)
        .with_context(|| format!("writing temp remote cache {}", temp.path().display()))?;
    temp.flush()
        .with_context(|| format!("flushing temp remote cache {}", temp.path().display()))?;
    match temp.persist(cache_path) {
        Ok(_) => Ok(()),
        Err(_error) if cache_path.exists() => Ok(()),
        Err(error) => Err(error.error).with_context(|| {
            format!(
                "persisting temp remote cache {} -> {}",
                error.file.path().display(),
                cache_path.display()
            )
        }),
    }
}

#[derive(Debug, Clone)]
pub struct SourceHarvest {
    pub source_key: String,
    pub display_name: String,
    pub streams: Vec<RemoteStreamSnapshot>,
}

#[derive(Debug, Clone)]
pub struct RemoteStreamSnapshot {
    pub thread_no: i64,
    pub title: String,
    pub semantic_slug: String,
    pub last_modified: i64,
    pub reply_count: u32,
    pub image_count: u32,
    pub items: Vec<RemoteItemSnapshot>,
}

#[derive(Debug, Clone)]
pub struct RemoteItemSnapshot {
    pub thread_no: i64,
    pub post_no: i64,
    pub title: String,
    pub image_url: String,
    pub thumb_url: String,
    pub ext: String,
    pub md5: Option<String>,
    pub width: u32,
    pub height: u32,
    pub file_size: u64,
    pub materialized_path: Option<PathBuf>,
}

impl RemoteItemSnapshot {
    fn from_post(board: &str, thread_no: i64, post: FourChanPost) -> Option<Self> {
        let tim = post.tim?;
        let ext = post.ext.clone()?;
        let image_url = format!("{IMAGE_ROOT}/{board}/{tim}{ext}");
        let thumb_url = format!("{IMAGE_ROOT}/{board}/{tim}s.jpg");
        Some(Self {
            thread_no,
            post_no: post.no,
            title: post_title(&post),
            image_url,
            thumb_url,
            ext,
            md5: post.md5,
            width: post.width.max(0) as u32,
            height: post.height.max(0) as u32,
            file_size: post.file_size.max(0) as u64,
            materialized_path: None,
        })
    }

    fn shortest_edge(&self) -> u32 {
        self.width.min(self.height)
    }
}

#[derive(Debug, Deserialize)]
struct FourChanCatalogPage {
    threads: Vec<FourChanCatalogThread>,
}

#[derive(Debug, Deserialize)]
struct FourChanCatalogThread {
    no: i64,
    #[serde(default, deserialize_with = "deserialize_boolish")]
    sticky: bool,
    #[serde(default, deserialize_with = "deserialize_boolish")]
    closed: bool,
    last_modified: i64,
    #[serde(default)]
    replies: i64,
    #[serde(default)]
    images: i64,
}

fn deserialize_boolish<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<i64>::deserialize(deserializer)?;
    Ok(raw.unwrap_or(0) != 0)
}

#[derive(Debug, Deserialize)]
struct FourChanThreadResponse {
    posts: Vec<FourChanPost>,
}

#[derive(Debug, Deserialize, Clone)]
struct FourChanPost {
    no: i64,
    resto: i64,
    #[serde(default)]
    sub: Option<String>,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    semantic_url: Option<String>,
    #[serde(default)]
    tim: Option<i64>,
    #[serde(default)]
    ext: Option<String>,
    #[serde(default, rename = "fsize")]
    file_size: i64,
    #[serde(default, rename = "w")]
    width: i64,
    #[serde(default, rename = "h")]
    height: i64,
    #[serde(default)]
    md5: Option<String>,
}

fn thread_title(thread: &FourChanThreadResponse) -> String {
    thread
        .posts
        .first()
        .map(post_title)
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| {
            let no = thread.posts.first().map_or(0, |post| post.no);
            format!("thread {no}")
        })
}

fn post_title(post: &FourChanPost) -> String {
    post.sub
        .as_deref()
        .map(decode_4chan_text)
        .or_else(|| post.filename.as_deref().map(decode_4chan_text))
        .or_else(|| post.semantic_url.as_deref().map(decode_4chan_text))
        .unwrap_or_else(|| {
            if post.resto == 0 {
                format!("thread {}", post.no)
            } else {
                format!("post {}", post.no)
            }
        })
}

fn decode_4chan_text(text: &str) -> String {
    if text.contains('&') {
        decode_html_entities(text).into_owned()
    } else {
        text.to_owned()
    }
}

fn sanitize_source_key(source_key: &str) -> String {
    source_key
        .chars()
        .map(|glyph| {
            if glyph.is_ascii_alphanumeric() {
                glyph
            } else {
                '-'
            }
        })
        .collect()
}

fn normalized_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .filter(|ext| !ext.is_empty())
        .unwrap_or_else(|| "bin".to_owned())
}

fn relative_component(path: &Path) -> String {
    let value = path.to_string_lossy().replace('\\', "/");
    if value.is_empty() {
        ".".to_owned()
    } else {
        value
    }
}

fn hashed_local_id(key: &str) -> i64 {
    let digest = blake3::hash(key.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    let raw = u64::from_le_bytes(bytes) & 0x7fff_ffff_ffff_ffff;
    i64::try_from(raw.max(1)).unwrap_or(1)
}

fn sniff_image_dimensions(path: &Path) -> Option<(u32, u32)> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl"))
    {
        return sniff_jxl_dimensions(path).or_else(|| {
            let bytes = fs::read(path).ok()?;
            let image = decode_image(&bytes).ok()?;
            Some((image.width(), image.height()))
        });
    }

    image_dimensions(path).ok().or_else(|| {
        let bytes = fs::read(path).ok()?;
        let image = decode_image(&bytes).ok()?;
        Some((image.width(), image.height()))
    })
}

fn sniff_jxl_dimensions(path: &Path) -> Option<(u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = JxlDecoder::new(file).ok()?;
    Some(decoder.dimensions())
}

fn supported_ext(ext: &str, allow_video: bool) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        ".jpg" | ".jpeg" | ".png" | ".gif" | ".webp" | ".bmp" | ".jxl"
    ) || (allow_video && matches!(ext.to_ascii_lowercase().as_str(), ".webm"))
}

#[cfg(test)]
mod tests {
    use std::{env, io::Cursor, path::PathBuf};

    use image::{Rgb, RgbImage};
    use ulid::Ulid;

    use super::{
        FourChanPost, FourChanThreadResponse, SourceScanner, post_title, supported_ext,
        thread_title,
    };
    use crate::config::{
        ImportPolicy, LocalDirectorySource, RemoteImageFilterConfig, SourceConfig, UpstreamSource,
    };

    fn test_root(name: &str) -> PathBuf {
        let root = env::temp_dir().join(format!("picmash-sources-{name}-{}", Ulid::new()));
        std::fs::create_dir_all(&root).expect("create temp root");
        root
    }

    fn flat_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, Rgb(rgb));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode png");
        bytes
    }

    #[test]
    fn image_source_default_keeps_video_dark() {
        assert!(supported_ext(".jpg", false));
        assert!(supported_ext(".jxl", false));
        assert!(!supported_ext(".webm", false));
        assert!(supported_ext(".webm", true));
    }

    #[test]
    fn post_title_decodes_html_entities_in_subjects() {
        let post = FourChanPost {
            no: 42,
            resto: 0,
            sub: Some("Tom &amp; Jerry &#039;96".to_owned()),
            filename: None,
            semantic_url: None,
            tim: None,
            ext: None,
            file_size: 0,
            width: 0,
            height: 0,
            md5: None,
        };

        assert_eq!(post_title(&post), "Tom & Jerry '96");
    }

    #[test]
    fn thread_title_decodes_html_entities_in_fallback_title_fields() {
        let thread = FourChanThreadResponse {
            posts: vec![FourChanPost {
                no: 777,
                resto: 0,
                sub: None,
                filename: Some("A &amp; B".to_owned()),
                semantic_url: Some("ignored".to_owned()),
                tim: None,
                ext: None,
                file_size: 0,
                width: 0,
                height: 0,
                md5: None,
            }],
        };

        assert_eq!(thread_title(&thread), "A & B");
    }

    #[test]
    fn remote_payload_matching_rejects_truncated_or_tampered_bytes() {
        let item = super::RemoteItemSnapshot {
            thread_no: 1,
            post_no: 2,
            title: "post 2".to_owned(),
            image_url: "https://example.invalid/thread/2.jpg".to_owned(),
            thumb_url: "https://example.invalid/thread/2s.jpg".to_owned(),
            ext: ".jpg".to_owned(),
            md5: Some("XUFAKrxLKna5cZ2REBfFkg==".to_owned()),
            width: 1,
            height: 1,
            file_size: 5,
            materialized_path: None,
        };

        assert!(super::remote_payload_matches_snapshot(b"hello", &item));
        assert!(!super::remote_payload_matches_snapshot(b"hell", &item));
        assert!(!super::remote_payload_matches_snapshot(b"world", &item));
    }

    #[test]
    fn local_directory_harvest_groups_by_parent_directory() {
        let root = test_root("local-harvest");
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).expect("create nested dir");
        let top_path = root.join("top.png");
        let nested_path = nested.join("child.png");
        std::fs::write(&top_path, flat_png(960, 960, [120, 80, 40])).expect("write top png");
        std::fs::write(&nested_path, flat_png(1200, 900, [80, 120, 200]))
            .expect("write nested png");

        let scanner = SourceScanner::new(&root.join(".cache")).expect("new scanner");
        let source = SourceConfig {
            weight: 1.0,
            import_policy: ImportPolicy::NotX,
            scan_interval_seconds: 600,
            upstream: UpstreamSource::LocalDirectory(LocalDirectorySource {
                root,
                recurse: true,
                filters: RemoteImageFilterConfig {
                    min_shortest_edge: 800,
                    max_download_bytes: 8 * 1024 * 1024,
                },
            }),
        };

        let harvest = scanner.harvest(&source).expect("harvest local dir");
        assert_eq!(harvest.streams.len(), 2);
        assert!(
            harvest
                .streams
                .iter()
                .any(|stream| stream.semantic_slug == ".")
        );
        assert!(
            harvest
                .streams
                .iter()
                .any(|stream| stream.semantic_slug == "nested")
        );

        let items = harvest
            .streams
            .into_iter()
            .flat_map(|stream| stream.items)
            .collect::<Vec<_>>();
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|item| {
            item.materialized_path
                .as_ref()
                .is_some_and(|path| path.exists())
        }));
    }
}
