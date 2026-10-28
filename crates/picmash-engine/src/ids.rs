use std::{fmt, path::Path, str::FromStr};

use thiserror::Error;
use ulid::Ulid;

#[derive(Debug, Error)]
#[error("{kind} cannot be empty")]
pub struct InvalidId {
    kind: &'static str,
}

macro_rules! text_id {
    ($name:ident, $kind:literal, fresh) => {
        text_id!($name, $kind);

        impl $name {
            #[must_use]
            pub fn fresh() -> Self {
                Self(Ulid::generate().to_string())
            }
        }
    };
    ($name:ident, $kind:literal) => {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, InvalidId> {
                let value = value.into();
                if value.is_empty() {
                    return Err(InvalidId { kind: $kind });
                }
                Ok(Self(value))
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = InvalidId;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::parse(value)
            }
        }
    };
}

macro_rules! integer_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(i64);

        impl $name {
            pub(crate) const fn from_raw(value: i64) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn get(self) -> i64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

text_id!(AssetId, "asset id", fresh);
text_id!(SessionId, "session id", fresh);
text_id!(PromptId, "prompt id", fresh);
text_id!(CommandId, "command id", fresh);
text_id!(SourceId, "source id");
text_id!(StreamId, "stream id");
text_id!(ItemId, "item id");
integer_id!(CollectionId);
integer_id!(OccurrenceId);
integer_id!(ObservationId);
integer_id!(SnapshotId);

// Remote state and provenance persist these spellings; changing one orphans
// every stored stream veto, item, and attestation minted under it.
impl SourceId {
    #[must_use]
    pub fn four_chan_board(board: &str) -> Self {
        Self(format!("4chan:{board}"))
    }

    #[must_use]
    pub fn local_directory(root: &Path) -> Self {
        Self(format!(
            "directory:{}",
            digest(&[root.as_os_str().as_encoded_bytes()], 16)
        ))
    }
}

impl StreamId {
    #[must_use]
    pub fn thread(board: &str, thread: i64) -> Self {
        Self(format!("4chan:{board}:{thread}"))
    }

    /// A folder, relative to its directory source's root.
    #[must_use]
    pub fn folder(folder: &Path) -> Self {
        Self(format!(
            "directory:{}",
            digest(&[folder.as_os_str().as_encoded_bytes()], 20)
        ))
    }
}

impl ItemId {
    #[must_use]
    pub fn post(board: &str, post: i64) -> Self {
        Self(format!("4chan:{board}:{post}"))
    }

    /// One version of a file, relative to the root of `source`.
    #[must_use]
    pub fn file(source: &SourceId, file: &Path, version: &[u8]) -> Self {
        Self(format!(
            "directory:{}",
            digest(
                &[
                    source.as_str().as_bytes(),
                    file.as_os_str().as_encoded_bytes(),
                    version,
                ],
                24,
            )
        ))
    }
}

fn digest(parts: &[&[u8]], width: usize) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        let _hasher = hasher.update(part);
    }
    hasher.finalize().to_hex()[..width].to_owned()
}
