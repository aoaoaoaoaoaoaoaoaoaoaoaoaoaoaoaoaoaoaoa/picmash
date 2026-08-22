mod calculus;
mod promotion;
mod reactor;
mod sources;
mod store;

use std::{
    fmt, fs,
    io::Cursor,
    os::unix::fs::MetadataExt as _,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use image::ImageDecoder as _;
use jxl_oxide::integration::JxlDecoder;

use crate::configuration::{ImportPolicy, SourceIdentity};

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

macro_rules! textual_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

textual_id!(StreamId);
textual_id!(RemoteItemId);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Origin {
    Network {
        url: String,
        expected_md5: Option<String>,
    },
    Local {
        path: PathBuf,
        stamp: FileStamp,
    },
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
    pub source_identity: SourceIdentity,
    pub source_name: String,
    pub import_policy: ImportPolicy,
    pub stream_id: StreamId,
    pub stream_title: String,
    pub item_id: RemoteItemId,
    pub title: String,
    pub origin: Origin,
    pub extension: String,
    pub width: u32,
    pub height: u32,
    pub byte_len: u64,
    pub max_pixels: u64,
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
    pub source_identity: SourceIdentity,
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
