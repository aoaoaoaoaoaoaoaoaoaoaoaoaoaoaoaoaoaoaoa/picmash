//! Canonical admission of a quarantined payload into the collection.

use std::{
    fs,
    io::{Cursor, Write as _},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context as _, Result, ensure};
use atomic_write_file::AtomicWriteFile;
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use image::ImageFormat;
pub use picmash_engine::DuelVictor;
use picmash_engine::{CommandId, PresentedAsset, SessionId, canonical_image, inspect_bytes};
use tempfile::Builder;
use wait_timeout::ChildExt as _;

use super::{Prepared, slug, validate_payload_dimensions};

pub(super) const ARCHIVE_CAPACITY: usize = 8;
const CANCELLATION_POLL: Duration = Duration::from_millis(250);
/// Tortoise: the highest effort every `cjxl` accepts. Glacier (10) needs
/// libjxl 0.10 and buys about a percent of lossless size over it.
const ENCODER_EFFORT: &str = "9";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionJudgment {
    Duel {
        session_id: SessionId,
        anchor: PresentedAsset,
        victor: DuelVictor,
        command_id: CommandId,
        response_ms: u32,
    },
    Favorite {
        session_id: SessionId,
        command_id: CommandId,
    },
}

impl PromotionJudgment {
    pub fn session_id(&self) -> &SessionId {
        match self {
            Self::Duel { session_id, .. } | Self::Favorite { session_id, .. } => session_id,
        }
    }

    pub fn command_id(&self) -> &CommandId {
        match self {
            Self::Duel { command_id, .. } | Self::Favorite { command_id, .. } => command_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromotionIntent {
    pub candidate: Prepared,
    pub collection_root: PathBuf,
    pub rotation_quarters: u8,
    pub judgment: PromotionJudgment,
}

pub struct ArchiveEffect {
    pub intent: PromotionIntent,
    pub result: std::result::Result<Promotion, String>,
}

pub struct ArchiveLane {
    cancel: Arc<AtomicBool>,
    submissions: Option<Sender<PromotionIntent>>,
    completions: Receiver<ArchiveEffect>,
    thread: Option<JoinHandle<()>>,
}

impl ArchiveLane {
    pub fn raise() -> Result<Self> {
        let cancel = Arc::new(AtomicBool::new(false));
        let (submissions, work) = bounded::<PromotionIntent>(ARCHIVE_CAPACITY);
        let (publish, completions) = bounded(ARCHIVE_CAPACITY);
        let worker_cancel = Arc::clone(&cancel);
        let thread = thread::Builder::new()
            .name("picmash-remote-archive".to_owned())
            .spawn(move || {
                for intent in work {
                    if worker_cancel.load(Ordering::Acquire) {
                        break;
                    }
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        promote(
                            &intent.collection_root,
                            &intent.candidate,
                            intent.rotation_quarters,
                            &worker_cancel,
                        )
                    }))
                    .map_or_else(
                        |_| Err("remote archive effect panicked".to_owned()),
                        |result| result.map_err(|error| format!("{error:#}")),
                    );
                    if publish.send(ArchiveEffect { intent, result }).is_err() {
                        break;
                    }
                }
            })
            .context("raise bounded remote archive lane")?;
        Ok(Self {
            cancel,
            submissions: Some(submissions),
            completions,
            thread: Some(thread),
        })
    }

    pub const fn completions(&self) -> &Receiver<ArchiveEffect> {
        &self.completions
    }

    pub fn submit(&self, intent: PromotionIntent) -> Result<()> {
        self.submissions
            .as_ref()
            .context("remote archive lane retired")?
            .try_send(intent)
            .map_err(|error| match error {
                TrySendError::Full(_) => anyhow::anyhow!("remote archive reservoir is full"),
                TrySendError::Disconnected(_) => {
                    anyhow::anyhow!("remote archive lane has stopped")
                }
            })
    }
}

impl Drop for ArchiveLane {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        drop(self.submissions.take());
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            eprintln!("Picmash remote archive lane panicked during retirement");
        }
    }
}

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
    cancel: &AtomicBool,
) -> Result<Promotion> {
    ensure!(
        !cancel.load(Ordering::Acquire),
        "JPEG XL promotion canceled"
    );
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
    let diagnostics = forge.path().join("cjxl.stderr");
    fs::write(&input, canonical_png.into_inner())
        .with_context(|| format!("write promotion input at {}", input.display()))?;
    let mut encoder = Command::new("cjxl")
        .args(["-d", "0", "-e", ENCODER_EFFORT, "--quiet"])
        .arg(&input)
        .arg(&output)
        .stdout(Stdio::null())
        .stderr(Stdio::from(
            fs::File::create(&diagnostics).context("create JPEG XL encoder diagnostics")?,
        ))
        .spawn()
        .context("raise total JPEG XL promotion encoder")?;
    let status = loop {
        if cancel.load(Ordering::Acquire) {
            encoder
                .kill()
                .context("kill canceled JPEG XL promotion encoder")?;
            let _status = encoder
                .wait()
                .context("reap canceled JPEG XL promotion encoder")?;
            anyhow::bail!("JPEG XL promotion canceled");
        }
        if let Some(status) = encoder
            .wait_timeout(CANCELLATION_POLL)
            .context("await JPEG XL promotion encoder")?
        {
            break status;
        }
    };
    ensure!(
        status.success(),
        "JPEG XL promotion encoder failed: {}",
        fs::read_to_string(&diagnostics)
            .unwrap_or_else(|_| "diagnostics unavailable".to_owned())
            .trim()
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
