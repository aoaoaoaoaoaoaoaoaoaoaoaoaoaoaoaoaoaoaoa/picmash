//! The showcase specimen: freely licensed cat photographs fetched from
//! Wikimedia Commons into the host cache and admitted only by SHA-256.

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use ureq::{
    Agent,
    tls::{TlsConfig, TlsProvider},
};

const MANIFEST: &str = include_str!("../specimen/cats.toml");
const CREDITS: &str = "specimen-credits.md";
/// Ceiling on one photograph's bytes; the largest is 11 MiB.
const PHOTOGRAPH_LIMIT: u64 = 32 << 20;

#[derive(Deserialize)]
struct Manifest {
    photograph: Vec<Photograph>,
}

/// One manifest entry: a photograph's provenance, license, and seal.
#[derive(Deserialize)]
pub struct Photograph {
    pub file: String,
    title: String,
    author: String,
    attribution: Option<String>,
    license: String,
    license_url: String,
    page: String,
    source: String,
    sha256: String,
}

/// Every photograph of the specimen, sealed in the host cache.
pub struct Specimen {
    cache: PathBuf,
    photographs: Vec<Photograph>,
}

impl Specimen {
    /// Admit every photograph, fetching over the host network whichever the
    /// cache lacks or holds unsealed.
    pub fn gather() -> Result<Self> {
        let Manifest { photograph } =
            toml::from_str(MANIFEST).context("parse specimen manifest")?;
        let cache = cache_root()?.join("picmash-acceptance/specimen");
        fs::create_dir_all(&cache).with_context(|| format!("create {}", cache.display()))?;
        let mut agent = None;
        for photograph in &photograph {
            let path = cache.join(&photograph.file);
            if fs::read(&path).is_ok_and(|bytes| seal(&bytes) == photograph.sha256) {
                continue;
            }
            let bytes = agent
                .get_or_insert_with(courier)
                .get(&photograph.source)
                .call()
                .with_context(|| format!("fetch {}", photograph.source))?
                .into_body()
                .with_config()
                .limit(PHOTOGRAPH_LIMIT)
                .read_to_vec()
                .with_context(|| format!("read {}", photograph.source))?;
            ensure!(
                seal(&bytes) == photograph.sha256,
                "{} no longer matches its manifest SHA-256",
                photograph.source
            );
            let staging = path.with_extension("partial");
            fs::write(&staging, bytes).with_context(|| format!("write {}", staging.display()))?;
            fs::rename(&staging, &path).with_context(|| format!("seal {}", path.display()))?;
        }
        Ok(Self {
            cache,
            photographs: photograph,
        })
    }

    pub fn photographs(&self) -> &[Photograph] {
        &self.photographs
    }

    pub fn path(&self, photograph: &Photograph) -> PathBuf {
        self.cache.join(&photograph.file)
    }

    /// Publish the credits every still and film owes its photographs.
    pub fn credit(&self, directory: &Path) -> Result<PathBuf> {
        let mut credits = String::from(
            "# Specimen credits\n\nPhotographs from Wikimedia Commons, shown scaled and cropped by Picmash.\n\n",
        );
        for photograph in &self.photographs {
            let author = photograph
                .attribution
                .clone()
                .unwrap_or_else(|| format!("by {}", photograph.author));
            let _ = writeln!(
                credits,
                "- [“{}”](<{}>), {author}, [{}](<{}>)",
                photograph.title, photograph.page, photograph.license, photograph.license_url
            );
        }
        let path = directory.join(CREDITS);
        fs::write(&path, credits).with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    }
}

fn cache_root() -> Result<PathBuf> {
    env::var_os("XDG_CACHE_HOME")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .context("neither XDG_CACHE_HOME nor HOME names a cache root")
}

fn courier() -> Agent {
    Agent::config_builder()
        .user_agent(concat!(
            "picmash-acceptance/",
            env!("CARGO_PKG_VERSION"),
            " (https://eternalist.moe; showcase specimen)"
        ))
        .tls_config(
            TlsConfig::builder()
                .provider(TlsProvider::NativeTls)
                .build(),
        )
        .build()
        .into()
}

fn seal(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}
