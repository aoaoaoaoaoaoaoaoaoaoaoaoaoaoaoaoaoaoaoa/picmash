use super::*;
use std::time::Instant;
use tracing::{Instrument, info_span};

const SLOW_MEDIA_RESPONSE_MS: u128 = 100;

pub(super) async fn asset_image(
    State(state): State<SharedRuntimeState>,
    Path(asset_id): Path<String>,
    Query(query): Query<AssetQuery>,
) -> WebResult<Response> {
    let rendition = query.rendition();
    let span = info_span!(
        "media.asset",
        asset_id = %asset_id,
        rendition = rendition.as_str(),
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let Some(asset) = state.maybe_image_asset(&AssetId(asset_id.clone()))? else {
            warn!(asset_id = %asset_id, "asset image missing from corpus");
            return Ok((StatusCode::NOT_FOUND, "missing asset").into_response());
        };
        if !asset.path.exists() {
            warn!(asset_id = %asset_id, path = %asset.path.display(), "asset image path missing");
            return Ok((StatusCode::NOT_FOUND, "missing asset").into_response());
        }
        rendition_response(
            state.cache_root(),
            &asset.id.0,
            asset.rotation_quarters,
            &asset.path,
            rendition,
        )
        .await
    }
    .instrument(span)
    .await
}

pub(super) async fn remote_image(
    State(state): State<SharedRuntimeState>,
    Path(item_id): Path<i64>,
    Query(query): Query<AssetQuery>,
) -> WebResult<Response> {
    let rendition = query.rendition();
    let span = info_span!(
        "media.remote",
        remote_item_id = item_id,
        rendition = rendition.as_str(),
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let Some(item) = state.maybe_remote_item(RemoteItemId(item_id))? else {
            warn!(remote_item_id = item_id, "remote image missing from store");
            return Ok((StatusCode::NOT_FOUND, "missing remote").into_response());
        };
        if !item.path.exists() {
            warn!(
                remote_item_id = item_id,
                path = %item.path.display(),
                "remote image path missing"
            );
            return Ok((StatusCode::NOT_FOUND, "missing remote").into_response());
        }
        rendition_response(
            state.cache_root(),
            &format!("remote-{}", item.id.0),
            item.rotation_quarters,
            &item.path,
            rendition,
        )
        .await
    }
    .instrument(span)
    .await
}

pub(super) async fn font_asset(Path(font_name): Path<String>) -> WebResult<Response> {
    let path = design_language_font_path(&font_name)?;
    let bytes = fs::read(&path)
        .await
        .with_context(|| format!("reading design-language font {}", path.display()))?;

    let mut response = Response::new(bytes.into());
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("font/woff2"));
    Ok(response)
}

pub(super) async fn favicon_asset() -> WebResult<Response> {
    let mut response = Response::new(FAVICON_SVG.as_bytes().to_vec().into());
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("image/svg+xml"));
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    Ok(response)
}

fn header_value_for_path(path: &std::path::Path) -> HeaderValue {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => HeaderValue::from_static("image/jpeg"),
        Some("gif") => HeaderValue::from_static("image/gif"),
        Some("webp") => HeaderValue::from_static("image/webp"),
        Some("bmp") => HeaderValue::from_static("image/bmp"),
        Some("jxl") => HeaderValue::from_static("image/jxl"),
        _ => HeaderValue::from_static("image/png"),
    }
}

fn normalize_payload(
    bytes: Vec<u8>,
    rotation_quarters: i32,
    rendition: AssetRendition,
) -> anyhow::Result<Vec<u8>> {
    let image = crate::identity::decode_image(&bytes)?;
    let oriented = if rendition.rotates_pixels() {
        rotate_image(image, rotation_quarters)
    } else {
        image
    };
    let bounded = match rendition {
        AssetRendition::Board | AssetRendition::Explore => oriented.resize_to_fill(
            rendition.max_edge(),
            rendition.max_edge(),
            rendition.resize_filter(),
        ),
        _ if oriented.width().max(oriented.height()) > rendition.max_edge() => oriented.resize(
            rendition.max_edge(),
            rendition.max_edge(),
            rendition.resize_filter(),
        ),
        _ => oriented,
    };

    let rgba = bounded.into_rgba8();
    let mut out = Vec::new();
    PngEncoder::new_with_quality(&mut out, CompressionType::Fast, PngFilterType::Adaptive)
        .write_image(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            ColorType::Rgba8.into(),
        )
        .context("encoding rotated image as PNG")?;
    Ok(out)
}

fn rotate_image(image: DynamicImage, rotation_quarters: i32) -> DynamicImage {
    let rgba = image.into_rgba8();
    match rotation_quarters.rem_euclid(4) {
        1 => DynamicImage::ImageRgba8(imageops::rotate90(&rgba)),
        2 => DynamicImage::ImageRgba8(imageops::rotate180(&rgba)),
        3 => DynamicImage::ImageRgba8(imageops::rotate270(&rgba)),
        _ => DynamicImage::ImageRgba8(rgba),
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AssetRendition {
    Arena,
    Board,
    Face,
    Explore,
    Preview,
}

impl AssetRendition {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Arena => "arena",
            Self::Board => "board",
            Self::Face => "face",
            Self::Explore => "explore",
            Self::Preview => "preview",
        }
    }

    pub(super) fn max_edge(self) -> u32 {
        match self {
            Self::Arena => 2_560,
            Self::Board => 512,
            Self::Face => 1_600,
            Self::Explore => 96,
            Self::Preview => 3_200,
        }
    }

    pub(super) fn resize_filter(self) -> FilterType {
        match self {
            Self::Board | Self::Explore => FilterType::Triangle,
            Self::Arena | Self::Face | Self::Preview => FilterType::CatmullRom,
        }
    }

    pub(super) fn rotates_pixels(self) -> bool {
        match self {
            Self::Arena => false,
            Self::Board | Self::Face | Self::Explore | Self::Preview => true,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct AssetQuery {
    kind: Option<String>,
}

impl AssetQuery {
    fn rendition(&self) -> AssetRendition {
        match self.kind.as_deref() {
            Some("board") => AssetRendition::Board,
            Some("face") => AssetRendition::Face,
            Some("explore") => AssetRendition::Explore,
            Some("preview") => AssetRendition::Preview,
            _ => AssetRendition::Arena,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct FaceImageQuery {
    #[serde(default)]
    pub(super) debug: bool,
    #[serde(default)]
    pub(super) full: bool,
}

pub(super) fn rendition_cache_path_for_key(
    cache_root: &FsPath,
    cache_key: &str,
    rotation_quarters: i32,
    rendition: AssetRendition,
) -> PathBuf {
    let shard = &cache_key[..cache_key.len().min(2)];
    cache_root.join(shard).join(format!(
        "{}.{}.r{}.v{RENDITION_CACHE_VERSION}.png",
        cache_key,
        rendition.as_str(),
        rotation_quarters.rem_euclid(4),
    ))
}

pub(super) fn rendition_failure_marker_path_for_key(
    cache_root: &FsPath,
    cache_key: &str,
    rotation_quarters: i32,
    rendition: AssetRendition,
) -> PathBuf {
    let shard = &cache_key[..cache_key.len().min(2)];
    cache_root.join(shard).join(format!(
        "{}.{}.r{}.fail",
        cache_key,
        rendition.as_str(),
        rotation_quarters.rem_euclid(4),
    ))
}

fn legacy_rendition_failure_marker_paths_for_key(
    cache_root: &FsPath,
    cache_key: &str,
    rotation_quarters: i32,
    rendition: AssetRendition,
) -> Vec<PathBuf> {
    let shard = &cache_key[..cache_key.len().min(2)];
    let mut legacy_versions = vec![RENDITION_CACHE_VERSION];
    if RENDITION_CACHE_VERSION > 1 {
        legacy_versions.push(RENDITION_CACHE_VERSION - 1);
    }
    legacy_versions
        .into_iter()
        .map(|version| {
            cache_root.join(shard).join(format!(
                "{}.{}.r{}.v{version}.fail",
                cache_key,
                rendition.as_str(),
                rotation_quarters.rem_euclid(4),
            ))
        })
        .collect()
}

pub(super) async fn has_rendition_failure_marker_for_key(
    cache_root: &FsPath,
    cache_key: &str,
    rotation_quarters: i32,
    rendition: AssetRendition,
) -> bool {
    let marker_path =
        rendition_failure_marker_path_for_key(cache_root, cache_key, rotation_quarters, rendition);
    if fs::try_exists(&marker_path).await.unwrap_or(false) {
        return true;
    }

    for legacy_path in legacy_rendition_failure_marker_paths_for_key(
        cache_root,
        cache_key,
        rotation_quarters,
        rendition,
    ) {
        if fs::try_exists(&legacy_path).await.unwrap_or(false) {
            return true;
        }
    }
    false
}

pub(super) async fn write_rendition_cache(path: &FsPath, bytes: &[u8]) -> anyhow::Result<()> {
    let Some(parent) = path.parent() else {
        bail!("cache path has no parent: {}", path.display());
    };
    fs::create_dir_all(parent)
        .await
        .with_context(|| format!("creating rendition cache directory {}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = parent.join(format!(
        ".tmp-{nonce}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("rendition")
    ));
    fs::write(&temp_path, bytes)
        .await
        .with_context(|| format!("writing temporary rendition cache {}", temp_path.display()))?;
    if !matches!(fs::rename(&temp_path, path).await, Ok(())) {
        fs::copy(&temp_path, path)
            .await
            .with_context(|| format!("copying rendition cache into {}", path.display()))?;
        fs::remove_file(&temp_path).await.with_context(|| {
            format!("removing temporary rendition cache {}", temp_path.display())
        })?;
    }
    Ok(())
}

pub(super) async fn write_rendition_failure_marker(
    path: &FsPath,
    message: &str,
) -> anyhow::Result<()> {
    let Some(parent) = path.parent() else {
        bail!("failure marker path has no parent: {}", path.display());
    };
    fs::create_dir_all(parent).await.with_context(|| {
        format!(
            "creating rendition failure marker directory {}",
            parent.display()
        )
    })?;
    fs::write(path, message)
        .await
        .with_context(|| format!("writing rendition failure marker {}", path.display()))?;
    Ok(())
}

async fn rendition_response(
    cache_root: &FsPath,
    cache_key: &str,
    rotation_quarters: i32,
    source_path: &FsPath,
    rendition: AssetRendition,
) -> WebResult<Response> {
    let started = Instant::now();
    let span = info_span!(
        "media.rendition",
        cache_key,
        rendition = rendition.as_str(),
        rotation_quarters,
        source_path = %source_path.display(),
    );
    async move {
        let cache_path =
            rendition_cache_path_for_key(cache_root, cache_key, rotation_quarters, rendition);
        let failure_marker_path = rendition_failure_marker_path_for_key(
            cache_root,
            cache_key,
            rotation_quarters,
            rendition,
        );
        if let Ok(body) = fs::read(&cache_path).await {
            return Ok(cached_image_response(
                body,
                HeaderValue::from_static("image/png"),
            ));
        }

        let bytes = match fs::read(source_path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                warn!(path = %source_path.display(), "media source path missing during rendition");
                return Ok((StatusCode::NOT_FOUND, "missing image").into_response());
            }
            Err(error) => {
                return Err(anyhow::Error::from(error)
                    .context(format!("reading image payload {}", source_path.display()))
                    .into());
            }
        };
        let original_mime = header_value_for_path(source_path);
        let fallback_bytes = bytes.clone();
        let fallback_path = source_path.to_path_buf();
        if has_rendition_failure_marker_for_key(cache_root, cache_key, rotation_quarters, rendition)
            .await
        {
            warn!(
                cache_key,
                rendition = rendition.as_str(),
                path = %fallback_path.display(),
                "serving original media after cached rendition failure"
            );
            return Ok(cached_image_response(fallback_bytes, original_mime));
        }
        let payload = tokio::task::spawn_blocking(move || {
            normalize_payload(bytes, rotation_quarters, rendition)
        })
        .await
        .context("joining normalized image task")?;

        let (body, mime) = match payload {
            Ok(bytes) => {
                if let Err(error) = write_rendition_cache(&cache_path, &bytes).await {
                    warn!(
                        "failed to persist rendition cache {}: {error:#}",
                        cache_path.display()
                    );
                }
                (bytes, HeaderValue::from_static("image/png"))
            }
            Err(error) => {
                warn!(
                    "failed to normalize asset {}: {error:#}; serving original bytes",
                    fallback_path.display()
                );
                if let Err(marker_error) =
                    write_rendition_failure_marker(&failure_marker_path, &error.to_string()).await
                {
                    warn!(
                        "failed to persist rendition failure marker {}: {marker_error:#}",
                        failure_marker_path.display()
                    );
                }
                (fallback_bytes, original_mime)
            }
        };

        let elapsed_ms = started.elapsed().as_millis();
        if elapsed_ms > SLOW_MEDIA_RESPONSE_MS {
            warn!(
                elapsed_ms,
                cache_key,
                rendition = rendition.as_str(),
                path = %source_path.display(),
                "slow media response"
            );
        }

        Ok(cached_image_response(body, mime))
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use super::{AssetRendition, normalize_payload};
    use image::GenericImageView;

    fn probe_png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([((x * 31) % 255) as u8, ((y * 47) % 255) as u8, 200, 255])
        });
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("encode probe png");
        out.into_inner()
    }

    #[test]
    fn quarter_turn_swaps_dimensions() {
        let payload = normalize_payload(probe_png(6, 14), 1, AssetRendition::Preview)
            .expect("normalize rotated probe");
        let decoded = image::load_from_memory(&payload).expect("decode normalized probe");
        assert_eq!(decoded.dimensions(), (14, 6));
    }

    #[test]
    fn board_rendition_is_square_thumbnail() {
        let payload = normalize_payload(probe_png(6, 14), 1, AssetRendition::Board)
            .expect("normalize board probe");
        let decoded = image::load_from_memory(&payload).expect("decode board probe");
        assert_eq!(decoded.dimensions(), (512, 512));
    }

    #[test]
    fn explore_rendition_is_small_square_thumbnail() {
        let payload = normalize_payload(probe_png(6, 14), 1, AssetRendition::Explore)
            .expect("normalize explore probe");
        let decoded = image::load_from_memory(&payload).expect("decode explore probe");
        assert_eq!(decoded.dimensions(), (96, 96));
    }

    #[test]
    fn arena_rendition_leaves_pixels_unturned() {
        let payload = normalize_payload(probe_png(6, 14), 1, AssetRendition::Arena)
            .expect("normalize arena probe");
        let decoded = image::load_from_memory(&payload).expect("decode arena probe");
        assert_eq!(decoded.dimensions(), (6, 14));
    }
}
