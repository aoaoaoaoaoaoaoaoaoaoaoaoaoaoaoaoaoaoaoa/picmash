//! The showcase specimen: cat photographs free of attribution and share-alike,
//! fetched from Wikimedia Commons and Pexels into the host cache and admitted
//! only by SHA-256.

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use std::{
    env,
    ffi::OsStr,
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
/// Ceiling on one photograph's bytes; the largest is 5.6 MiB.
const PHOTOGRAPH_LIMIT: u64 = 16 << 20;

#[derive(Deserialize)]
struct Manifest {
    photograph: Vec<Provenance>,
}

/// One manifest record: a photograph's origin, license, and seal.
#[derive(Deserialize)]
struct Provenance {
    file: String,
    title: String,
    author: String,
    license: String,
    license_url: Option<String>,
    page: String,
    source: String,
    sha256: String,
}

/// One photograph whose exact bytes matched their manifest seal.
pub struct Photograph {
    provenance: Provenance,
    bytes: Vec<u8>,
}

impl Photograph {
    pub fn file(&self) -> &str {
        &self.provenance.file
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Every photograph of the specimen, held as the very bytes that matched their
/// seals, so nothing unsealed can reach a testbed.
pub struct Specimen {
    photographs: Vec<Photograph>,
}

impl Specimen {
    /// Seal every photograph, fetching over the host network whichever the
    /// cache lacks or holds unsealed, and evict whatever else the cache holds.
    pub fn gather() -> Result<Self> {
        let Manifest { photograph } =
            toml::from_str(MANIFEST).context("parse specimen manifest")?;
        let cache = cache_root()?.join("picmash-acceptance/specimen");
        fs::create_dir_all(&cache).with_context(|| format!("create {}", cache.display()))?;
        let mut agent = None;
        let photographs = photograph
            .into_iter()
            .map(|provenance| {
                let path = cache.join(&provenance.file);
                let bytes = match fs::read(&path) {
                    Ok(bytes) if seal(&bytes) == provenance.sha256 => bytes,
                    _ => {
                        let bytes = fetch(agent.get_or_insert_with(courier), &provenance)?;
                        let staging = cache.join(format!("{}.partial", provenance.file));
                        fs::write(&staging, &bytes)
                            .with_context(|| format!("write {}", staging.display()))?;
                        fs::rename(&staging, &path)
                            .with_context(|| format!("seal {}", path.display()))?;
                        bytes
                    }
                };
                Ok(Photograph { provenance, bytes })
            })
            .collect::<Result<Vec<_>>>()?;
        evict(&cache, &photographs)?;
        Ok(Self { photographs })
    }

    pub fn photographs(&self) -> &[Photograph] {
        &self.photographs
    }

    /// Publish the courtesy credits that accompany every still and film.
    pub fn credit(&self, directory: &Path) -> Result<PathBuf> {
        let mut credits = String::from(
            "# Specimen credits\n\nPhotographs from Wikimedia Commons and Pexels, none asking attribution or share-alike, shown scaled and cropped by Picmash.\n\n",
        );
        for Photograph { provenance, .. } in &self.photographs {
            let license = match &provenance.license_url {
                Some(url) => format!("[{}](<{url}>)", provenance.license),
                None => provenance.license.clone(),
            };
            let _ = writeln!(
                credits,
                "- [“{}”](<{}>), by {}, {license}",
                provenance.title, provenance.page, provenance.author
            );
        }
        let path = directory.join(CREDITS);
        fs::write(&path, credits).with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    }
}

fn fetch(agent: &Agent, provenance: &Provenance) -> Result<Vec<u8>> {
    let bytes = agent
        .get(&provenance.source)
        .call()
        .with_context(|| format!("fetch {}", provenance.source))?
        .into_body()
        .with_config()
        .limit(PHOTOGRAPH_LIMIT)
        .read_to_vec()
        .with_context(|| format!("read {}", provenance.source))?;
    ensure!(
        seal(&bytes) == provenance.sha256,
        "{} no longer matches its manifest SHA-256",
        provenance.source
    );
    Ok(bytes)
}

/// Remove every cache entry the manifest does not name: retired photographs
/// and abandoned partial fetches.
fn evict(cache: &Path, photographs: &[Photograph]) -> Result<()> {
    let survey = || format!("survey {}", cache.display());
    for entry in fs::read_dir(cache).with_context(survey)? {
        let path = entry.with_context(survey)?.path();
        if !photographs
            .iter()
            .any(|photograph| path.file_name() == Some(OsStr::new(photograph.file())))
        {
            fs::remove_file(&path).with_context(|| format!("evict {}", path.display()))?;
        }
    }
    Ok(())
}

fn cache_root() -> Result<PathBuf> {
    env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|root| root.is_absolute())
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .context("neither an absolute XDG_CACHE_HOME nor HOME names a cache root")
}

/// An agent that asks only for the formats seals cover: image CDNs negotiate
/// the representation they serve, and a seal names exactly one.
fn courier() -> Agent {
    Agent::config_builder()
        .accept("image/jpeg, image/png")
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
