//! Canonical admission of a quarantined payload into the collection.

use std::{
    fs,
    io::{Cursor, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context as _, Result, ensure};
use atomic_write_file::AtomicWriteFile;
use image::ImageFormat;
use picmash_engine::{canonical_image, inspect_bytes};
use tempfile::Builder;

use super::{Prepared, slug, validate_payload_dimensions};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Promotion {
    pub path: PathBuf,
}

/// Re-encodes every admitted image, including an upstream JPEG XL payload.
/// The transform is lossless over Picmash's canonical oriented raster.
pub fn promote(
    collection: &Path,
    candidate: &Prepared,
    rotation_quarters: u8,
) -> Result<Promotion> {
    ensure!(collection.is_dir(), "collection left the filesystem");
    let original = fs::read(&candidate.cache_path).with_context(|| {
        format!(
            "read prepared remote candidate {}",
            candidate.cache_path.display()
        )
    })?;
    validate_payload_dimensions(
        &original,
        &candidate.discovery.extension,
        (candidate.discovery.width, candidate.discovery.height),
        candidate.discovery.max_pixels,
    )?;
    let mut image = canonical_image(&original).with_context(|| {
        format!(
            "decode prepared remote candidate {}",
            candidate.discovery.item_id
        )
    })?;
    image = match rotation_quarters % 4 {
        1 => image.rotate90(),
        2 => image.rotate180(),
        3 => image.rotate270(),
        _ => image,
    };
    let mut canonical_png = Cursor::new(Vec::new());
    image
        .write_to(&mut canonical_png, ImageFormat::Png)
        .context("stage canonical raster for JPEG XL encoding")?;
    let expected = inspect_bytes(canonical_png.get_ref())
        .context("inspect canonical raster before JPEG XL encoding")?;

    let forge = Builder::new()
        .prefix("picmash-promote-")
        .tempdir()
        .context("raise promotion forge")?;
    let input = forge.path().join("canonical.png");
    let output = forge.path().join("canonical.jxl");
    fs::write(&input, canonical_png.into_inner())
        .with_context(|| format!("write promotion input at {}", input.display()))?;
    let result = Command::new("cjxl")
        .args(["-d", "0", "-e", "10", "--quiet"])
        .arg(&input)
        .arg(&output)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("execute total JPEG XL promotion encoder")?;
    ensure!(
        result.status.success(),
        "JPEG XL promotion encoder failed: {}",
        String::from_utf8_lossy(&result.stderr).trim()
    );
    let encoded = fs::read(&output)
        .with_context(|| format!("read promoted JPEG XL at {}", output.display()))?;
    let actual = inspect_bytes(&encoded).context("verify promoted JPEG XL")?;
    ensure!(
        actual.render == expected.render,
        "promotion changed the canonical image"
    );

    let directory = collection
        .join(".picmash-imported")
        .join(slug(candidate.discovery.source_identity.as_str()));
    fs::create_dir_all(&directory)
        .with_context(|| format!("create promotion directory at {}", directory.display()))?;
    let identity = blake3::hash(candidate.discovery.item_id.as_str().as_bytes());
    let target = directory.join(format!("{}.jxl", &identity.to_hex()[..32]));
    let mut file = AtomicWriteFile::open(&target)
        .with_context(|| format!("stage promoted image at {}", target.display()))?;
    file.write_all(&encoded)
        .with_context(|| format!("write promoted image at {}", target.display()))?;
    file.commit()
        .with_context(|| format!("commit promoted image at {}", target.display()))?;
    Ok(Promotion { path: target })
}
