use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    thread,
};

use anyhow::{Context, bail};
use image::ImageFormat;
use tempfile::Builder as TempDirBuilder;
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrushPlan {
    JpegToJxl,
    PngRace,
    RasterToJxl,
    Passthrough,
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

pub fn crush_import_image(path_hint: &Path, bytes: &[u8]) -> CrushedImage {
    let plan = crush_plan(bytes);
    let original = CrushedImage {
        bytes: bytes.to_vec(),
        extension: normalized_extension(path_hint),
    };
    match plan {
        CrushPlan::JpegToJxl => choose_replacement(
            original,
            jxl_lossless_jpeg_candidate(path_hint, bytes),
            path_hint,
            "cjxl --lossless_jpeg=1",
        ),
        CrushPlan::PngRace => choose_smallest(
            original,
            [
                ("oxipng", oxipng_candidate(bytes)),
                (
                    "cjxl -d 0 -e 10",
                    jxl_lossless_pixels_candidate(path_hint, bytes),
                ),
            ],
            path_hint,
        ),
        CrushPlan::RasterToJxl => choose_replacement(
            original,
            jxl_lossless_pixels_candidate(path_hint, bytes),
            path_hint,
            "cjxl -d 0 -e 10",
        ),
        CrushPlan::Passthrough => original,
    }
}

fn oxipng_candidate(original: &[u8]) -> anyhow::Result<CrushedImage> {
    Ok(CrushedImage {
        bytes: shell_filter(
            "oxipng",
            &[
                "--stdout",
                "--opt",
                "max",
                "--zopfli",
                "--strip",
                "safe",
                "--threads",
                "1",
                "--quiet",
                "-",
            ],
            original,
        )?,
        extension: "png".to_owned(),
    })
}

fn jxl_lossless_jpeg_candidate(path_hint: &Path, original: &[u8]) -> anyhow::Result<CrushedImage> {
    Ok(CrushedImage {
        bytes: cjxl_file_transform(
            path_hint,
            original,
            &["--lossless_jpeg=1", "-e", "10", "--quiet"],
        )?,
        extension: "jxl".to_owned(),
    })
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

fn crush_plan(bytes: &[u8]) -> CrushPlan {
    match image::guess_format(bytes) {
        Ok(ImageFormat::Jpeg) => CrushPlan::JpegToJxl,
        Ok(ImageFormat::Png) => CrushPlan::PngRace,
        Ok(ImageFormat::Gif | ImageFormat::WebP | ImageFormat::Bmp) => CrushPlan::RasterToJxl,
        _ => CrushPlan::Passthrough,
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

fn choose_replacement(
    original: CrushedImage,
    candidate: anyhow::Result<CrushedImage>,
    path_hint: &Path,
    tool_name: &str,
) -> CrushedImage {
    match candidate {
        Ok(candidate) if !candidate.bytes.is_empty() => candidate,
        Ok(_) => original,
        Err(error) => {
            warn!(
                path = %path_hint.display(),
                tool_name,
                error = %error,
                "lossless crush failed; admitting original bytes",
            );
            original
        }
    }
}

fn choose_smallest<const N: usize>(
    mut best: CrushedImage,
    candidates: [(&str, anyhow::Result<CrushedImage>); N],
    path_hint: &Path,
) -> CrushedImage {
    for (tool_name, candidate) in candidates {
        match candidate {
            Ok(candidate)
                if !candidate.bytes.is_empty() && candidate.bytes.len() < best.bytes.len() =>
            {
                best = candidate;
            }
            Ok(_) => {}
            Err(error) => warn!(
                path = %path_hint.display(),
                tool_name,
                error = %error,
                "lossless crush candidate failed; ignoring candidate",
            ),
        }
    }
    best
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

fn shell_filter(program: &str, args: &[&str], input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {program}"))?;
    let mut stdin = child
        .stdin
        .take()
        .with_context(|| format!("capturing stdin for {program}"))?;
    let payload = input.to_vec();
    let program_name = program.to_owned();
    let writer = thread::spawn(move || -> anyhow::Result<()> {
        stdin
            .write_all(&payload)
            .with_context(|| format!("streaming payload into {program_name}"))?;
        drop(stdin);
        Ok(())
    });
    let output = child
        .wait_with_output()
        .with_context(|| format!("waiting for {program}"))?;
    writer
        .join()
        .map_err(|_| anyhow::anyhow!("joining {program} stdin writer"))??;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("{program} exited {}: {}", output.status, stderr.trim());
    }
    if output.stdout.is_empty() {
        bail!("{program} produced no output");
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
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
            crush_plan(&flat_image_bytes(ImageFormat::Png)),
            CrushPlan::PngRace
        );
    }

    #[test]
    fn recognizes_jpeg_payloads() {
        assert_eq!(
            crush_plan(&flat_image_bytes(ImageFormat::Jpeg)),
            CrushPlan::JpegToJxl
        );
    }

    #[test]
    fn recognizes_webp_payloads() {
        assert_eq!(
            crush_plan(&flat_image_bytes(ImageFormat::WebP)),
            CrushPlan::RasterToJxl
        );
    }
}
