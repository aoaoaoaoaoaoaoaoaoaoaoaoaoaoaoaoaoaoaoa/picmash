use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use image::{
    DynamicImage, GenericImageView,
    imageops::{self, FilterType},
};
use ort::{environment::Environment, ep, logging::LogLevel, session::Session, value::Tensor};
use parking_lot::Mutex;
use reqwest::blocking::Client;
use tracing::{info, warn};

use crate::{
    face::{DetectedFace, ScrfdStrideOutput, decode_scrfd_outputs, scrfd_input_tensor},
    identity::canonical_embedding_image,
    model::EmbeddingRecord,
};

// ─── DINO constants ──────────────────────────────────────────

const DEFAULT_DINO_REPO: &str =
    "https://huggingface.co/onnx-community/dinov2-base/resolve/main/onnx/model.onnx";
const DEFAULT_MODEL_NAME: &str = "dinov2-base@onnx-v1";
const DEFAULT_DINO_FILE: &str = "dinov2-base/model.onnx";
const INPUT_SIDE: u32 = 224;
const RESIZE_SHORTEST_EDGE: u32 = 256;
const INPUT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const INPUT_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ─── CLIP constants ─────────────────────────────────────────

const DEFAULT_CLIP_REPO: &str =
    "https://huggingface.co/pimash/openclip-vit-b-32-laion2b/resolve/main/openclip_vitb32.onnx";
const DEFAULT_CLIP_FILE: &str = "openclip-vitb32/model.onnx";
const DEFAULT_CLIP_MODEL_NAME: &str = "openclip-vitb32-laion2b@onnx-v1";
const CLIP_INPUT_SIDE: u32 = 224;
const CLIP_RESIZE_SHORTEST_EDGE: u32 = 224;

// ─── SCRFD constants ─────────────────────────────────────────

const DEFAULT_SCRFD_REPO: &str =
    "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx";
const DEFAULT_SCRFD_FILE: &str = "scrfd/scrfd_10g_bnkps.onnx";
const DEFAULT_SCRFD_MODEL_NAME: &str = "scrfd-10g@onnx-v2";
const SCRFD_STRIDES: [u32; 3] = [8, 16, 32];

// ─── ArcFace constants ───────────────────────────────────────

const DEFAULT_ARCFACE_REPO: &str =
    "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx";
const DEFAULT_ARCFACE_FILE: &str = "arcface/glintr100.onnx";
const DEFAULT_ARCFACE_MODEL_NAME: &str = "arcface-glintr100@onnx-v1";
const ARCFACE_INPUT_SIDE: u32 = 112;
const ARCFACE_MEAN: f32 = 127.5;
const ARCFACE_SCALE: f32 = 128.0;

// ─── shared ──────────────────────────────────────────────────

const DOWNLOAD_LOG_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

// ─── session wrappers ────────────────────────────────────────

struct DinoSession {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

struct ClipSession {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

struct ScrfdSession {
    session: Mutex<Session>,
    input_name: String,
}

struct ArcFaceSession {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

// ─── engine ──────────────────────────────────────────────────

pub struct OnnxEngine {
    dino: Option<DinoSession>,
    clip: Option<ClipSession>,
    scrfd: Option<ScrfdSession>,
    arcface: Option<ArcFaceSession>,
    model_name: String,
    clip_model_name: String,
    arcface_model_name: String,
    status_note: Option<String>,
    clip_note: Option<String>,
    scrfd_note: Option<String>,
    arcface_note: Option<String>,
}

impl OnnxEngine {
    #[must_use]
    pub fn from_env(cache_root: &Path) -> Self {
        let model_name = env::var("PICMASH_ONNX_DINO_MODEL_NAME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_owned());

        let clip_model_name = env::var("PICMASH_ONNX_CLIP_MODEL_NAME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_CLIP_MODEL_NAME.to_owned());

        let arcface_model_name = env::var("PICMASH_ONNX_ARCFACE_MODEL_NAME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ARCFACE_MODEL_NAME.to_owned());

        init_ort_environment();

        let dino = match ignite_dino(cache_root, &model_name) {
            Ok(session) => Some(session),
            Err(error) => {
                let note = format!("DINO bootstrap failed: {error:#}");
                warn!("{note}");
                return Self {
                    dino: None,
                    clip: None,
                    scrfd: None,
                    arcface: None,
                    model_name,
                    clip_model_name,
                    arcface_model_name,
                    status_note: Some(note),
                    clip_note: Some("CLIP skipped (DINO unavailable)".to_owned()),
                    scrfd_note: Some("SCRFD skipped (DINO unavailable)".to_owned()),
                    arcface_note: Some("ArcFace skipped (DINO unavailable)".to_owned()),
                };
            }
        };

        let (clip, clip_note) = match ignite_clip(cache_root, &clip_model_name) {
            Ok(session) => (Some(session), None),
            Err(error) => {
                let note = format!("CLIP bootstrap failed: {error:#}");
                warn!("{note}");
                (None, Some(note))
            }
        };

        let (scrfd, scrfd_note) = match ignite_scrfd(cache_root) {
            Ok(session) => (Some(session), None),
            Err(error) => {
                let note = format!("SCRFD bootstrap failed: {error:#}");
                warn!("{note}");
                (None, Some(note))
            }
        };

        let (arcface, arcface_note) = match ignite_arcface(cache_root, &arcface_model_name) {
            Ok(session) => (Some(session), None),
            Err(error) => {
                let note = format!("ArcFace bootstrap failed: {error:#}");
                warn!("{note}");
                (None, Some(note))
            }
        };

        Self {
            dino,
            clip,
            scrfd,
            arcface,
            model_name,
            clip_model_name,
            arcface_model_name,
            status_note: None,
            clip_note,
            scrfd_note,
            arcface_note,
        }
    }

    /// Whether the `DINOv2` embedding session is live.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.dino.is_some()
    }

    /// Whether the `OpenCLIP` embedding session is live.
    #[must_use]
    pub const fn clip_enabled(&self) -> bool {
        self.clip.is_some()
    }

    /// Whether the SCRFD face detection session is live.
    #[must_use]
    pub const fn face_detection_enabled(&self) -> bool {
        self.scrfd.is_some()
    }

    #[must_use]
    pub const fn recognition_enabled(&self) -> bool {
        self.arcface.is_some()
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    pub fn clip_model_name(&self) -> &str {
        &self.clip_model_name
    }

    #[must_use]
    pub fn recognition_model_name(&self) -> &str {
        &self.arcface_model_name
    }

    #[must_use]
    pub fn status_note(&self) -> Option<&str> {
        self.status_note.as_deref()
    }

    #[must_use]
    pub fn clip_note(&self) -> Option<&str> {
        self.clip_note.as_deref()
    }

    #[must_use]
    pub fn scrfd_note(&self) -> Option<&str> {
        self.scrfd_note.as_deref()
    }

    #[must_use]
    pub fn arcface_note(&self) -> Option<&str> {
        self.arcface_note.as_deref()
    }

    #[must_use]
    pub const fn face_detection_model_name(&self) -> &'static str {
        DEFAULT_SCRFD_MODEL_NAME
    }

    pub fn recognize_face_image(
        &self,
        image: &DynamicImage,
    ) -> anyhow::Result<Option<EmbeddingRecord>> {
        let Some(arcface) = &self.arcface else {
            return Ok(None);
        };
        let input = arcface_input_tensor(image)?;
        let mut session = arcface.session.lock();
        let outputs = session
            .run(ort::inputs![arcface.input_name.as_str() => input])
            .context("running ArcFace inference")?;
        let output = outputs
            .get(arcface.output_name.as_str())
            .unwrap_or_else(|| &outputs[0]);
        let vector = extract_embedding_vector(output).context("extracting ArcFace embedding")?;
        Ok(Some(EmbeddingRecord {
            model_name: self.arcface_model_name.clone(),
            vector: l2_normalize(vector),
        }))
    }

    /// Embed a whole image from disk (reads, decodes, normalizes, infers).
    pub fn embed(&self, path: &Path) -> anyhow::Result<Option<EmbeddingRecord>> {
        let Some(dino) = &self.dino else {
            return Ok(None);
        };

        let bytes = fs::read(path).with_context(|| {
            format!(
                "reading image bytes for ONNX embedder from {}",
                path.display()
            )
        })?;
        let image = match canonical_embedding_image(&bytes) {
            Ok(image) => image,
            Err(error) => {
                warn!(
                    "skipping ONNX embedding for {}: failed to canonicalize: {error:#}",
                    path.display()
                );
                return Ok(None);
            }
        };

        self.embed_image_inner(dino, &image)
            .with_context(|| format!("DINO embedding for {}", path.display()))
    }

    /// Embed an already-decoded image (e.g. an aligned face crop).
    pub fn embed_image(&self, image: &DynamicImage) -> anyhow::Result<Option<EmbeddingRecord>> {
        let Some(dino) = &self.dino else {
            return Ok(None);
        };
        self.embed_image_inner(dino, image)
    }

    /// CLIP-embed a whole image from disk.
    pub fn clip_embed(&self, path: &Path) -> anyhow::Result<Option<EmbeddingRecord>> {
        let Some(clip) = &self.clip else {
            return Ok(None);
        };

        let bytes = fs::read(path).with_context(|| {
            format!(
                "reading image bytes for CLIP embedder from {}",
                path.display()
            )
        })?;
        let image = match canonical_embedding_image(&bytes) {
            Ok(image) => image,
            Err(error) => {
                warn!(
                    "skipping CLIP embedding for {}: failed to canonicalize: {error:#}",
                    path.display()
                );
                return Ok(None);
            }
        };

        self.clip_embed_inner(clip, &image)
            .with_context(|| format!("CLIP embedding for {}", path.display()))
    }

    /// CLIP-embed an already-decoded image.
    pub fn clip_embed_image(
        &self,
        image: &DynamicImage,
    ) -> anyhow::Result<Option<EmbeddingRecord>> {
        let Some(clip) = &self.clip else {
            return Ok(None);
        };
        self.clip_embed_inner(clip, image)
    }

    fn clip_embed_inner(
        &self,
        clip: &ClipSession,
        image: &DynamicImage,
    ) -> anyhow::Result<Option<EmbeddingRecord>> {
        let input = clip_input_tensor(image)?;
        let mut session = clip.session.lock();
        let outputs = session
            .run(ort::inputs![clip.input_name.as_str() => input])
            .context("running CLIP inference")?;
        let output = outputs
            .get(clip.output_name.as_str())
            .unwrap_or_else(|| &outputs[0]);
        let vector = extract_embedding_vector(output).context("extracting CLIP embedding")?;
        Ok(Some(EmbeddingRecord {
            model_name: self.clip_model_name.clone(),
            vector: l2_normalize(vector),
        }))
    }

    fn embed_image_inner(
        &self,
        dino: &DinoSession,
        image: &DynamicImage,
    ) -> anyhow::Result<Option<EmbeddingRecord>> {
        let input = dino_input_tensor(image)?;
        let mut session = dino.session.lock();
        let outputs = session
            .run(ort::inputs![dino.input_name.as_str() => input])
            .context("running DINO inference")?;
        let output = outputs
            .get(dino.output_name.as_str())
            .unwrap_or_else(|| &outputs[0]);
        let vector = extract_embedding_vector(output).context("extracting DINO embedding")?;
        Ok(Some(EmbeddingRecord {
            model_name: self.model_name.clone(),
            vector: l2_normalize(vector),
        }))
    }

    /// Detect faces in an image via SCRFD.
    pub fn detect_faces(&self, image: &DynamicImage) -> anyhow::Result<Vec<DetectedFace>> {
        let Some(scrfd) = &self.scrfd else {
            return Ok(Vec::new());
        };

        let (input, scale, pad_x, pad_y) = scrfd_input_tensor(image)?;
        let mut session = scrfd.session.lock();
        let outputs = session
            .run(ort::inputs![scrfd.input_name.as_str() => input])
            .context("running SCRFD inference")?;

        let stride_outputs = parse_scrfd_outputs(&outputs)?;
        Ok(decode_scrfd_outputs(&stride_outputs, scale, pad_x, pad_y))
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn disabled_for_tests() -> Self {
        Self {
            dino: None,
            clip: None,
            scrfd: None,
            arcface: None,
            model_name: DEFAULT_MODEL_NAME.to_owned(),
            clip_model_name: DEFAULT_CLIP_MODEL_NAME.to_owned(),
            arcface_model_name: DEFAULT_ARCFACE_MODEL_NAME.to_owned(),
            status_note: Some("test ONNX engine disabled".to_owned()),
            clip_note: Some("test CLIP disabled".to_owned()),
            scrfd_note: Some("test SCRFD disabled".to_owned()),
            arcface_note: Some("test ArcFace disabled".to_owned()),
        }
    }
}

// ─── session construction ────────────────────────────────────

fn init_ort_environment() {
    ort::init().with_name("picmash-onnx").commit();
    if let Ok(env) = Environment::current() {
        env.set_log_level(LogLevel::Warning);
    }
}

fn build_session(model_path: &Path) -> anyhow::Result<Session> {
    let builder = Session::builder().context("creating ONNX session builder")?;
    let mut builder = builder
        .with_execution_providers([ep::CUDA::default().build()])
        .map_err(|error| anyhow::anyhow!("configuring execution providers: {error}"))?;
    builder
        .commit_from_file(model_path)
        .with_context(|| format!("loading ONNX model from {}", model_path.display()))
}

fn ignite_dino(cache_root: &Path, model_name: &str) -> anyhow::Result<DinoSession> {
    let model_path = resolve_model_path(
        cache_root,
        "PICMASH_ONNX_DINO_PATH",
        "PICMASH_ONNX_DINO_URL",
        DEFAULT_DINO_FILE,
        DEFAULT_DINO_REPO,
    )?;
    info!(model = %model_name, path = %model_path.display(), "loading DINO model");

    let session = build_session(&model_path)?;
    let input_name = session
        .inputs()
        .first()
        .context("DINO model exposes no inputs")?
        .name()
        .to_owned();
    let output_name = session
        .outputs()
        .first()
        .context("DINO model exposes no outputs")?
        .name()
        .to_owned();

    Ok(DinoSession {
        session: Mutex::new(session),
        input_name,
        output_name,
    })
}

fn ignite_clip(cache_root: &Path, model_name: &str) -> anyhow::Result<ClipSession> {
    let model_path = resolve_model_path(
        cache_root,
        "PICMASH_ONNX_CLIP_PATH",
        "PICMASH_ONNX_CLIP_URL",
        DEFAULT_CLIP_FILE,
        DEFAULT_CLIP_REPO,
    )?;
    info!(model = %model_name, path = %model_path.display(), "loading CLIP model");

    let session = build_session(&model_path)?;
    let input_name = session
        .inputs()
        .first()
        .context("CLIP model exposes no inputs")?
        .name()
        .to_owned();
    let output_name = session
        .outputs()
        .first()
        .context("CLIP model exposes no outputs")?
        .name()
        .to_owned();

    Ok(ClipSession {
        session: Mutex::new(session),
        input_name,
        output_name,
    })
}

fn ignite_scrfd(cache_root: &Path) -> anyhow::Result<ScrfdSession> {
    let model_path = resolve_model_path(
        cache_root,
        "PICMASH_SCRFD_PATH",
        "PICMASH_SCRFD_URL",
        DEFAULT_SCRFD_FILE,
        DEFAULT_SCRFD_REPO,
    )?;
    info!(path = %model_path.display(), "loading SCRFD face detection model");

    let session = build_session(&model_path)?;
    let input_name = session
        .inputs()
        .first()
        .context("SCRFD model exposes no inputs")?
        .name()
        .to_owned();

    Ok(ScrfdSession {
        session: Mutex::new(session),
        input_name,
    })
}

fn ignite_arcface(cache_root: &Path, model_name: &str) -> anyhow::Result<ArcFaceSession> {
    let model_path = resolve_model_path(
        cache_root,
        "PICMASH_ARCFACE_PATH",
        "PICMASH_ARCFACE_URL",
        DEFAULT_ARCFACE_FILE,
        DEFAULT_ARCFACE_REPO,
    )?;
    info!(model = %model_name, path = %model_path.display(), "loading ArcFace model");

    let session = build_session(&model_path)?;
    let input_name = session
        .inputs()
        .first()
        .context("ArcFace model exposes no inputs")?
        .name()
        .to_owned();
    let output_name = session
        .outputs()
        .first()
        .context("ArcFace model exposes no outputs")?
        .name()
        .to_owned();

    Ok(ArcFaceSession {
        session: Mutex::new(session),
        input_name,
        output_name,
    })
}

// ─── model resolution / download ─────────────────────────────

fn resolve_model_path(
    cache_root: &Path,
    path_env: &str,
    url_env: &str,
    default_file: &str,
    default_url: &str,
) -> anyhow::Result<PathBuf> {
    if let Ok(raw_path) = env::var(path_env) {
        let model_path = PathBuf::from(raw_path);
        if !model_path.exists() {
            bail!("{path_env} points at missing file {}", model_path.display());
        }
        return Ok(model_path);
    }

    let model_root = cache_root.join("models");
    let model_path = model_root.join(default_file);
    if model_path.exists() {
        return Ok(model_path);
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating model cache {}", parent.display()))?;
    }
    download_model(
        &env::var(url_env).unwrap_or_else(|_| default_url.to_owned()),
        &model_path,
    )?;
    Ok(model_path)
}

fn download_model(url: &str, model_path: &Path) -> anyhow::Result<()> {
    let temp_path = model_path.with_extension("download");
    let client = Client::builder()
        .user_agent("picmash/0.1")
        .build()
        .context("building ONNX model downloader")?;
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("downloading ONNX model from {url}"))?
        .error_for_status()
        .with_context(|| format!("reading ONNX model response from {url}"))?;
    let total = response.content_length();
    info!(
        url,
        path = %model_path.display(),
        bytes = total,
        "downloading ONNX model"
    );

    let mut file = fs::File::create(&temp_path)
        .with_context(|| format!("creating temporary model file {}", temp_path.display()))?;
    let mut downloaded = 0u64;
    let mut next_log = DOWNLOAD_LOG_CHUNK_BYTES;
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = response
            .read(&mut chunk)
            .with_context(|| format!("streaming ONNX model from {url}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&chunk[..read])
            .with_context(|| format!("writing model chunk to {}", temp_path.display()))?;
        downloaded += u64::try_from(read).unwrap_or_default();
        if downloaded >= next_log {
            info!(downloaded, total, "ONNX model download progress");
            next_log += DOWNLOAD_LOG_CHUNK_BYTES;
        }
    }
    file.sync_all()
        .with_context(|| format!("syncing model {}", temp_path.display()))?;
    fs::rename(&temp_path, model_path).with_context(|| {
        format!(
            "moving downloaded model {} → {}",
            temp_path.display(),
            model_path.display()
        )
    })?;
    info!(downloaded, path = %model_path.display(), "ONNX model download complete");
    Ok(())
}

// ─── DINO preprocessing ─────────────────────────────────────

fn dino_input_tensor(image: &DynamicImage) -> anyhow::Result<Tensor<f32>> {
    let resized = resize_shortest_edge(image, RESIZE_SHORTEST_EDGE).to_rgb8();
    let cropped = center_crop(&resized, INPUT_SIDE, INPUT_SIDE);
    let pixel_count = usize::try_from(INPUT_SIDE.saturating_mul(INPUT_SIDE)).unwrap_or_default();
    let mut tensor = Vec::with_capacity(pixel_count * 3);
    for channel in 0..3 {
        for pixel in cropped.pixels() {
            let value = f32::from(pixel[channel]) / 255.0;
            tensor.push((value - INPUT_MEAN[channel]) / INPUT_STD[channel]);
        }
    }
    Tensor::from_array((
        [1usize, 3, INPUT_SIDE as usize, INPUT_SIDE as usize],
        tensor,
    ))
    .context("building DINO input tensor")
}

fn clip_input_tensor(image: &DynamicImage) -> anyhow::Result<Tensor<f32>> {
    let resized = resize_shortest_edge(image, CLIP_RESIZE_SHORTEST_EDGE).to_rgb8();
    let cropped = center_crop(&resized, CLIP_INPUT_SIDE, CLIP_INPUT_SIDE);
    let pixel_count =
        usize::try_from(CLIP_INPUT_SIDE.saturating_mul(CLIP_INPUT_SIDE)).unwrap_or_default();
    let mut tensor = Vec::with_capacity(pixel_count * 3);
    for channel in 0..3 {
        for pixel in cropped.pixels() {
            let value = f32::from(pixel[channel]) / 255.0;
            tensor.push((value - INPUT_MEAN[channel]) / INPUT_STD[channel]);
        }
    }
    Tensor::from_array((
        [
            1usize,
            3,
            CLIP_INPUT_SIDE as usize,
            CLIP_INPUT_SIDE as usize,
        ],
        tensor,
    ))
    .context("building CLIP input tensor")
}

fn arcface_input_tensor(image: &DynamicImage) -> anyhow::Result<Tensor<f32>> {
    let resized = image
        .resize_exact(
            ARCFACE_INPUT_SIDE,
            ARCFACE_INPUT_SIDE,
            FilterType::CatmullRom,
        )
        .to_rgb8();
    let pixel_count =
        usize::try_from(ARCFACE_INPUT_SIDE.saturating_mul(ARCFACE_INPUT_SIDE)).unwrap_or_default();
    let mut tensor = Vec::with_capacity(pixel_count * 3);
    for channel in [2usize, 1, 0] {
        for pixel in resized.pixels() {
            tensor.push((f32::from(pixel[channel]) - ARCFACE_MEAN) / ARCFACE_SCALE);
        }
    }
    Tensor::from_array((
        [
            1usize,
            3,
            ARCFACE_INPUT_SIDE as usize,
            ARCFACE_INPUT_SIDE as usize,
        ],
        tensor,
    ))
    .context("building ArcFace input tensor")
}

fn resize_shortest_edge(image: &DynamicImage, target_shortest: u32) -> DynamicImage {
    let (width, height) = image.dimensions();
    let shortest = width.min(height).max(1);
    let scale = target_shortest as f32 / shortest as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(target_shortest);
    let resized_height = ((height as f32 * scale).round() as u32).max(target_shortest);
    DynamicImage::ImageRgba8(imageops::resize(
        image,
        resized_width,
        resized_height,
        FilterType::CatmullRom,
    ))
}

fn center_crop(image: &image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    let left = image.width().saturating_sub(width) / 2;
    let top = image.height().saturating_sub(height) / 2;
    imageops::crop_imm(image, left, top, width, height).to_image()
}

fn extract_embedding_vector(output: &ort::value::DynValue) -> anyhow::Result<Vec<f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .context("extracting ONNX output as f32 tensor")?;
    match shape.as_ref() {
        [_, hidden] | [_, _, hidden] | [hidden] => {
            let hidden = usize::try_from(*hidden).context("invalid hidden size")?;
            Ok(data[..hidden].to_vec())
        }
        shape => bail!("unsupported DINO output shape {shape:?}"),
    }
}

fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return vector;
    }
    for v in &mut vector {
        *v /= norm;
    }
    vector
}

// ─── SCRFD output parsing ────────────────────────────────────

/// Parse the raw ONNX output tensors from SCRFD into stride-grouped
/// structures. Outputs are auto-classified by their last dimension:
///   • dim 1 or 2 → score tensor
///   • dim 4      → bbox tensor
///   • dim 10     → keypoints tensor
/// Then grouped by descending anchor count (stride 8 → 16 → 32).
struct RawTensor {
    num_anchors: usize,
    last_dim: usize,
    data: Vec<f32>,
}

fn parse_scrfd_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
) -> anyhow::Result<Vec<ScrfdStrideOutput>> {
    let num_outputs = outputs.len();
    if num_outputs < 9 {
        bail!("SCRFD model produced {num_outputs} outputs (expected ≥ 9)");
    }

    let mut tensors = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let (shape, data) = outputs[i]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("extracting SCRFD output [{i}]"))?;
        let dims = shape.as_ref();
        let last_dim = usize::try_from(*dims.last().context("empty SCRFD output shape")?)
            .context("invalid SCRFD dim")?;
        let total = data.len();
        let num_anchors = total / last_dim.max(1);
        tensors.push(RawTensor {
            num_anchors,
            last_dim,
            data: data.to_vec(),
        });
    }

    // Group by type: scores (dim 1 or 2), bboxes (dim 4), kps (dim 10)
    let mut scores: Vec<&RawTensor> = tensors
        .iter()
        .filter(|t| t.last_dim == 1 || t.last_dim == 2)
        .collect();
    let mut bboxes: Vec<&RawTensor> = tensors.iter().filter(|t| t.last_dim == 4).collect();
    let mut kps: Vec<&RawTensor> = tensors.iter().filter(|t| t.last_dim == 10).collect();

    if scores.len() < 3 || bboxes.len() < 3 || kps.len() < 3 {
        bail!(
            "SCRFD output grouping failed: {} scores, {} bboxes, {} kps tensors",
            scores.len(),
            bboxes.len(),
            kps.len(),
        );
    }

    // Sort each group by descending anchor count (stride 8 has the most)
    scores.sort_by(|a, b| b.num_anchors.cmp(&a.num_anchors));
    bboxes.sort_by(|a, b| b.num_anchors.cmp(&a.num_anchors));
    kps.sort_by(|a, b| b.num_anchors.cmp(&a.num_anchors));

    let mut result = Vec::with_capacity(3);
    for (idx, &stride) in SCRFD_STRIDES.iter().enumerate() {
        let s = scores[idx];
        let b = bboxes[idx];
        let k = kps[idx];
        result.push(ScrfdStrideOutput {
            stride,
            num_anchors: s.num_anchors,
            score_classes: s.last_dim,
            scores: s.data.clone(),
            bboxes: b.data.clone(),
            kps: k.data.clone(),
        });
    }

    Ok(result)
}
