use std::io::{BufReader, Cursor};

use anyhow::Context;
use exif::{In, Reader, Tag, Value};
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use jxl_oxide::integration::JxlDecoder;
use ulid::Ulid;

use crate::model::AssetId;

const VISUAL_SIDE: u32 = 32;
const VISUAL_QUANT_SHIFT: u8 = 5;
const VISUAL_ALGO: &str = "visual:v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlobId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VisualKey(pub String);

#[derive(Debug, Clone)]
pub struct ImageIdentity {
    pub blob_id: BlobId,
    pub visual_key: VisualKey,
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

pub fn inspect_image_bytes(bytes: &[u8]) -> anyhow::Result<ImageIdentity> {
    let blob_id = BlobId(blake3::hash(bytes).to_hex().to_string());
    let oriented = canonical_embedding_image(bytes)?;
    let (width, height) = oriented.dimensions();
    let visual_key = strict_visual_key(&oriented);
    Ok(ImageIdentity {
        blob_id,
        visual_key,
        width,
        height,
        byte_len: u64::try_from(bytes.len()).unwrap_or_default(),
    })
}

pub fn canonical_embedding_payload(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    let oriented = canonical_embedding_image(bytes)?;
    let mut payload = Cursor::new(Vec::new());
    oriented
        .write_to(&mut payload, ImageFormat::Png)
        .context("encoding canonical embedding payload as PNG")?;
    Ok(payload.into_inner())
}

/// Decode image bytes, falling back to libjpeg-turbo for JPEGs that the
/// `image` crate's decoder chokes on (e.g. arithmetic-coded).
pub fn decode_image(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    if looks_like_jxl(bytes) {
        let decoder =
            JxlDecoder::new(Cursor::new(bytes)).context("initializing JPEG XL decoder")?;
        return DynamicImage::from_decoder(decoder).context("decoding JPEG XL image");
    }
    match image::load_from_memory(bytes) {
        Ok(image) => Ok(image),
        Err(_) if looks_like_jpeg(bytes) => {
            let rgb: RgbImage =
                turbojpeg::decompress_image(bytes).context("turbojpeg fallback decode failed")?;
            Ok(DynamicImage::ImageRgb8(rgb))
        }
        Err(primary_err) => Err(primary_err).context("decoding image identity payload"),
    }
}

pub(crate) fn canonical_embedding_image(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    let image = decode_image(bytes)?;
    if looks_like_jxl(bytes) {
        return Ok(image);
    }
    Ok(orient_by_exif(image, bytes))
}

fn looks_like_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8])
}

fn looks_like_jxl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
        ])
}

#[must_use]
pub fn mint_asset_id() -> AssetId {
    AssetId(Ulid::new().to_string())
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
    apply_orientation(image, orientation)
}

fn apply_orientation(image: DynamicImage, orientation: u16) -> DynamicImage {
    let rgba = image.to_rgba8();
    let oriented = match orientation {
        2 => imageops::flip_horizontal(&rgba),
        3 => imageops::rotate180(&rgba),
        4 => imageops::flip_vertical(&rgba),
        5 => imageops::rotate90(&imageops::flip_horizontal(&rgba)),
        6 => imageops::rotate90(&rgba),
        7 => imageops::rotate270(&imageops::flip_horizontal(&rgba)),
        8 => imageops::rotate270(&rgba),
        _ => rgba,
    };
    DynamicImage::ImageRgba8(oriented)
}

fn strict_visual_key(image: &DynamicImage) -> VisualKey {
    let blurred = image.blur(0.8);
    let proxy = contain_quantized_proxy(&blurred, VISUAL_SIDE, VISUAL_SIDE);
    let digest = blake3::hash(&proxy).to_hex().to_string();
    VisualKey(format!("{VISUAL_ALGO}:{digest}"))
}

fn contain_quantized_proxy(image: &DynamicImage, width: u32, height: u32) -> Vec<u8> {
    let fitted = image.resize(width, height, FilterType::Triangle).to_rgb8();
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
    let left = (width.saturating_sub(fitted.width())) / 2;
    let top = (height.saturating_sub(fitted.height())) / 2;
    canvas
        .copy_from(&fitted, left, top)
        .expect("fitted image always fits the proxy canvas");
    let mut quantized = Vec::with_capacity(
        usize::try_from(width.saturating_mul(height).saturating_mul(3)).unwrap_or_default(),
    );
    for pixel in canvas.pixels() {
        quantized.push(pixel[0] >> VISUAL_QUANT_SHIFT);
        quantized.push(pixel[1] >> VISUAL_QUANT_SHIFT);
        quantized.push(pixel[2] >> VISUAL_QUANT_SHIFT);
    }
    quantized
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{Rgb, RgbImage};

    use super::{BlobId, inspect_image_bytes};

    fn flat_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, Rgb(rgb));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode png");
        bytes
    }

    #[test]
    fn blob_identity_sees_byte_mutation() {
        let mut png = flat_png(40, 24, [200, 20, 60]);
        let left = inspect_image_bytes(&png).expect("identity");
        png[12] ^= 0x01;
        let right = BlobId(blake3::hash(&png).to_hex().to_string());
        assert_ne!(left.blob_id, right);
    }

    #[test]
    fn visual_identity_ignores_resolution_for_flat_equivalents() {
        let left = inspect_image_bytes(&flat_png(96, 64, [120, 90, 30])).expect("identity");
        let right = inspect_image_bytes(&flat_png(640, 426, [120, 90, 30])).expect("identity");
        assert_eq!(left.visual_key, right.visual_key);
    }
}
