use std::{
    fs,
    io::Cursor,
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{Context, bail};
use image::ImageFormat;
use tempfile::Builder as TempDirBuilder;
use tracing::warn;

use crate::identity::canonical_embedding_image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrushPlan {
    JxlPassthrough,
    JpegToJxl,
    RasterToJxl,
}

#[derive(Debug, Clone)]
pub struct CrushedImage {
    bytes: Vec<u8>,
    extension: String,
}

impl CrushedImage {
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    #[must_use]
    pub fn extension(&self) -> &str {
        &self.extension
    }
}

pub fn crush_import_image(path_hint: &Path, bytes: &[u8]) -> anyhow::Result<CrushedImage> {
    let plan = crush_plan(path_hint, bytes);
    match plan {
        CrushPlan::JxlPassthrough => Ok(CrushedImage {
            bytes: bytes.to_vec(),
            extension: "jxl".to_owned(),
        }),
        CrushPlan::JpegToJxl => jxl_lossless_jpeg_candidate(path_hint, bytes),
        CrushPlan::RasterToJxl => jxl_lossless_pixels_candidate(path_hint, bytes),
    }
}

fn jxl_lossless_jpeg_candidate(path_hint: &Path, original: &[u8]) -> anyhow::Result<CrushedImage> {
    match cjxl_file_transform(
        path_hint,
        original,
        &["--lossless_jpeg=1", "-e", "10", "--quiet"],
    ) {
        Ok(bytes) => Ok(CrushedImage {
            bytes,
            extension: "jxl".to_owned(),
        }),
        Err(lossless_error) => {
            warn!(
                error = %format!("{lossless_error:#}"),
                path = %path_hint.display(),
                "cjxl lossless jpeg transcode failed; falling back to raster jxl encode"
            );
            jxl_lossless_pixels_from_decoded_image(original)
                .with_context(|| format!("lossless jpeg transcode failed: {lossless_error:#}"))
        }
    }
}

fn jxl_lossless_pixels_candidate(
    path_hint: &Path,
    original: &[u8],
) -> anyhow::Result<CrushedImage> {
    Ok(CrushedImage {
        bytes: cjxl_file_transform(path_hint, original, &["-d", "0", "-e", "10", "--quiet"])?,
        extension: "jxl".to_owned(),
    })
}

fn jxl_lossless_pixels_from_decoded_image(original: &[u8]) -> anyhow::Result<CrushedImage> {
    let image = canonical_embedding_image(original)
        .context("decoding image into canonical raster before jxl encode")?;
    let mut png = Cursor::new(Vec::new());
    image
        .write_to(&mut png, ImageFormat::Png)
        .context("encoding decoded raster as png for cjxl fallback")?;
    Ok(CrushedImage {
        bytes: cjxl_file_transform(
            Path::new("fallback.png"),
            &png.into_inner(),
            &["-d", "0", "-e", "10", "--quiet"],
        )
        .context("encoding decoded raster into jxl")?,
        extension: "jxl".to_owned(),
    })
}

fn crush_plan(path_hint: &Path, bytes: &[u8]) -> CrushPlan {
    if normalized_extension(path_hint) == "jxl" {
        return CrushPlan::JxlPassthrough;
    }
    match image::guess_format(bytes) {
        Ok(ImageFormat::Jpeg) => CrushPlan::JpegToJxl,
        Ok(
            ImageFormat::Png
            | ImageFormat::Gif
            | ImageFormat::WebP
            | ImageFormat::Bmp
            | ImageFormat::Tiff,
        ) => CrushPlan::RasterToJxl,
        _ => CrushPlan::RasterToJxl,
    }
}

fn normalized_extension(path_hint: &Path) -> String {
    path_hint
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .filter(|ext| !ext.is_empty())
        .unwrap_or_else(|| "img".to_owned())
}

fn cjxl_file_transform(path_hint: &Path, input: &[u8], args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let tempdir = TempDirBuilder::new()
        .prefix("picmash-cjxl-")
        .tempdir()
        .context("allocating cjxl tempdir")?;
    let input_path = tempdir
        .path()
        .join(format!("input.{}", normalized_extension(path_hint)));
    let output_path = tempdir.path().join("output.jxl");
    fs::write(&input_path, input)
        .with_context(|| format!("writing cjxl input {}", input_path.display()))?;
    let output = Command::new("cjxl")
        .args(args)
        .arg(&input_path)
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("running cjxl")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("cjxl exited {}: {}", output.status, stderr.trim());
    }
    let encoded = fs::read(&output_path)
        .with_context(|| format!("reading cjxl output {}", output_path.display()))?;
    if encoded.is_empty() {
        bail!("cjxl produced no output");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use image::{ImageFormat, Rgb, RgbImage};

    use super::{CrushPlan, crush_plan};

    fn flat_image_bytes(format: ImageFormat) -> Vec<u8> {
        let image = RgbImage::from_pixel(8, 8, Rgb([40, 50, 60]));
        let mut out = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut out), format)
            .expect("encode image");
        out
    }

    #[test]
    fn recognizes_png_payloads() {
        assert_eq!(
            crush_plan(Path::new("test.png"), &flat_image_bytes(ImageFormat::Png)),
            CrushPlan::RasterToJxl
        );
    }

    #[test]
    fn recognizes_jpeg_payloads() {
        assert_eq!(
            crush_plan(Path::new("test.jpg"), &flat_image_bytes(ImageFormat::Jpeg)),
            CrushPlan::JpegToJxl
        );
    }

    #[test]
    fn recognizes_webp_payloads() {
        assert_eq!(
            crush_plan(Path::new("test.webp"), &flat_image_bytes(ImageFormat::WebP)),
            CrushPlan::RasterToJxl
        );
    }

    #[test]
    fn preserves_jxl_payloads() {
        assert_eq!(
            crush_plan(Path::new("test.jxl"), b"not-a-decoded-jxl"),
            CrushPlan::JxlPassthrough
        );
    }
}
