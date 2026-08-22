use std::{io, path::PathBuf};

use thiserror::Error;

use crate::ids::{InvalidId, PromptId};

pub type Result<T> = std::result::Result<T, Fault>;

#[derive(Debug, Error)]
pub enum Fault {
    #[error("database failure: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("filesystem failure at {path}: {source}")]
    Filesystem {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("invalid identifier: {0}")]
    InvalidId(#[from] InvalidId),

    #[error("image inspection failed at {path}: {source:#}")]
    Image {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("runtime failure: {0}")]
    Runtime(String),

    #[error("durable state is corrupt: {0}")]
    Corrupt(String),

    #[error("comparison prompt {0} is stale or already answered")]
    StalePrompt(PromptId),

    #[error("command id was already used for a different operation")]
    CommandCollision,

    #[error("preference snapshot became stale while it was being built")]
    StaleSnapshot,

    #[error("legacy database is unsupported: {0}")]
    UnsupportedLegacy(String),
}

pub trait IoResultExt<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoResultExt<T> for io::Result<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|source| Fault::Filesystem {
            path: path.into(),
            source,
        })
    }
}
