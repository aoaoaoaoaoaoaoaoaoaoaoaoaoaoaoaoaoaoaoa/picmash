use std::{
    io::{BufReader, Cursor},
    path::Path,
};

use anyhow::Context;
use exif::{In, Reader, Tag, Value};
use image::{DynamicImage, GenericImageView, imageops};
use jxl_oxide::integration::JxlDecoder;

use crate::fault::{Fault, IoResultExt, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlobDigest(String);

impl BlobDigest {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn parse(value: String) -> Result<Self> {
        if value.starts_with("blake3:") || value.starts_with("legacy:") {
            Ok(Self(value))
        } else {
            Err(Fault::Corrupt(format!("invalid blob digest `{value}`")))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderDigest(String);

impl RenderDigest {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn parse(value: String) -> Result<Self> {
        if value.starts_with("rgba-v1:") || value.starts_with("legacy:") {
            Ok(Self(value))
        } else {
            Err(Fault::Corrupt(format!("invalid render digest `{value}`")))
        }
    }

    pub(crate) fn legacy(asset_id: &str) -> Self {
        Self(format!("legacy:{asset_id}"))
    }
}

#[derive(Debug, Clone)]
pub struct ImageIdentity {
    pub blob: BlobDigest,
    pub render: RenderDigest,
    pub width: u32,
    pub height: u32,
    pub byte_len: u64,
}

impl ImageIdentity {
    #[must_use]
    pub fn pixel_area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

pub fn inspect_path(path: &Path) -> Result<ImageIdentity> {
    let bytes = std::fs::read(path).at(path)?;
    inspect_bytes(&bytes).map_err(|source| Fault::Image {
        path: path.to_path_buf(),
        source,
    })
}

pub fn inspect_bytes(bytes: &[u8]) -> anyhow::Result<ImageIdentity> {
    let blob = BlobDigest(format!("blake3:{}", blake3::hash(bytes).to_hex()));
    let image = canonical_image(bytes)?;
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(&rgba);
    Ok(ImageIdentity {
        blob,
        render: RenderDigest(format!("rgba-v1:{}", hasher.finalize().to_hex())),
        width,
        height,
        byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    })
}

pub fn canonical_image(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    let image = decode_image(bytes)?;
    if looks_like_jxl(bytes) {
        return Ok(image);
    }
    Ok(orient_by_exif(image, bytes))
}

fn decode_image(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    if looks_like_jxl(bytes) {
        let decoder =
            JxlDecoder::new(Cursor::new(bytes)).context("initializing JPEG XL decoder")?;
        return DynamicImage::from_decoder(decoder).context("decoding JPEG XL image");
    }
    match image::load_from_memory(bytes) {
        Ok(image) => Ok(image),
        Err(_) if bytes.starts_with(&[0xff, 0xd8]) => {
            let image = turbojpeg::decompress_image(bytes)
                .context("decoding JPEG through libjpeg-turbo fallback")?;
            Ok(DynamicImage::ImageRgb8(image))
        }
        Err(error) => Err(error).context("decoding image"),
    }
}

fn looks_like_jxl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0x0a])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
        ])
}

fn orient_by_exif(image: DynamicImage, bytes: &[u8]) -> DynamicImage {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let orientation = Reader::new()
        .read_from_container(&mut reader)
        .ok()
        .and_then(|exif| exif.get_field(Tag::Orientation, In::PRIMARY).cloned())
        .and_then(|field| match field.value {
            Value::Short(values) => values.first().copied(),
            Value::Long(values) => values.first().and_then(|value| u16::try_from(*value).ok()),
            _ => None,
        })
        .unwrap_or(1);
    let rgba = image.to_rgba8();
    DynamicImage::ImageRgba8(match orientation {
        2 => imageops::flip_horizontal(&rgba),
        3 => imageops::rotate180(&rgba),
        4 => imageops::flip_vertical(&rgba),
        5 => imageops::rotate90(&imageops::flip_horizontal(&rgba)),
        6 => imageops::rotate90(&rgba),
        7 => imageops::rotate270(&imageops::flip_horizontal(&rgba)),
        8 => imageops::rotate270(&rgba),
        _ => rgba,
    })
}

#[must_use]
pub fn supported_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "bmp" | "gif" | "jfif" | "jpeg" | "jpg" | "jxl" | "png" | "webp"
            )
        })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{ImageFormat, Rgb, RgbImage};

    use super::inspect_bytes;

    #[test]
    fn byte_variants_converge_on_exact_render_identity() {
        let image = RgbImage::from_pixel(96, 64, Rgb([120, 90, 30]));
        let mut png = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .expect("encode png");
        let mut bmp = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bmp), ImageFormat::Bmp)
            .expect("encode lossless bmp");

        let png = inspect_bytes(&png).expect("inspect png");
        let bmp = inspect_bytes(&bmp).expect("inspect bmp");
        assert_ne!(png.blob, bmp.blob);
        assert_eq!(png.render, bmp.render);
    }
}
