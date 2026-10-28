mod calculus;
mod promotion;
mod reactor;
mod sources;
mod store;

use std::{
    fs,
    io::Cursor,
    os::unix::fs::MetadataExt as _,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use image::ImageDecoder as _;
use jxl_oxide::integration::JxlDecoder;

use picmash_engine::{ItemId, Provenance, Source, SourceId, Stream, StreamId};

use crate::configuration::ImportPolicy;

pub use calculus::{CatalogIntent, FetchIntent, FetchSettlement, Machine, Moment};
pub use sources::Harvester;
pub use store::RemoteStore;

pub use promotion::{ArchiveEffect, ArchiveLane, DuelVictor, PromotionIntent, PromotionJudgment};
pub use reactor::Effect;
pub use reactor::Reactor;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Epoch(u64);

impl Epoch {
    pub const INITIAL: Self = Self(1);

    pub const fn successor(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceIx(usize);

impl SourceIx {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OriginSnapshot {
    Network {
        url: String,
        expected_md5: Option<String>,
    },
    Local {
        path: PathBuf,
        stamp: FileStamp,
    },
}

impl OriginSnapshot {
    /// Where the bytes were read, without the validation the fetch needs.
    pub fn origin(&self) -> picmash_engine::Origin {
        match self {
            Self::Network { url, expected_md5 } => picmash_engine::Origin::Url {
                url: url.clone(),
                md5: expected_md5.clone(),
            },
            Self::Local { path, .. } => picmash_engine::Origin::File(path.clone()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileStamp([u8; 56]);

impl FileStamp {
    pub fn read(path: &Path) -> Result<Self> {
        Ok(Self::from_metadata(&fs::metadata(path).with_context(
            || format!("inspect remote local file {}", path.display()),
        )?))
    }

    pub fn from_metadata(metadata: &fs::Metadata) -> Self {
        let components = [
            metadata.dev().to_le_bytes(),
            metadata.ino().to_le_bytes(),
            metadata.len().to_le_bytes(),
            metadata.mtime().to_le_bytes(),
            metadata.mtime_nsec().to_le_bytes(),
            metadata.ctime().to_le_bytes(),
            metadata.ctime_nsec().to_le_bytes(),
        ];
        let mut bytes = [0; 56];
        for (index, component) in components.into_iter().enumerate() {
            bytes[index * 8..][..8].copy_from_slice(&component);
        }
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        Ok(Self(bytes.try_into().map_err(|bytes: Vec<u8>| {
            anyhow::anyhow!("invalid remote file stamp length {}", bytes.len())
        })?))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Discovery {
    pub source: SourceIx,
    pub upstream: Source,
    pub import_policy: ImportPolicy,
    pub stream_id: StreamId,
    pub stream_title: String,
    pub item_id: ItemId,
    pub title: String,
    pub origin: OriginSnapshot,
    pub extension: String,
    pub width: u32,
    pub height: u32,
    pub byte_len: u64,
    pub max_pixels: u64,
}

impl Discovery {
    /// What an admission of this discovery records about where it came from.
    pub fn provenance(&self) -> Provenance {
        Provenance {
            source: self.upstream.clone(),
            stream: Stream {
                id: self.stream_id.clone(),
                title: self.stream_title.clone(),
            },
            item: self.item_id.clone(),
            origin: self.origin.origin(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prepared {
    pub discovery: Discovery,
    pub cache_path: PathBuf,
    pub payload_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Harvest {
    pub source: SourceIx,
    pub source_identity: SourceId,
    pub discoveries: Vec<Discovery>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Summary {
    pub enabled_sources: usize,
    pub cataloging: usize,
    pub backing_off: usize,
    pub discovered: usize,
    pub fetching: usize,
    pub prepared: usize,
    pub offered: bool,
    pub promoting: usize,
}

fn slug(value: &str) -> String {
    value
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

fn validate_payload_dimensions(
    bytes: &[u8],
    extension: &str,
    expected: (u32, u32),
    max_pixels: u64,
) -> Result<()> {
    let dimensions = if extension.eq_ignore_ascii_case("jxl") {
        JxlDecoder::new(Cursor::new(bytes))
            .context("inspect JPEG XL dimensions")?
            .dimensions()
    } else {
        image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .context("identify remote image format")?
            .into_dimensions()
            .context("inspect image dimensions")?
    };
    ensure!(dimensions == expected, "remote payload dimensions changed");
    ensure!(
        u64::from(dimensions.0).saturating_mul(u64::from(dimensions.1)) <= max_pixels,
        "remote payload exceeds its pixel envelope"
    );
    Ok(())
}
