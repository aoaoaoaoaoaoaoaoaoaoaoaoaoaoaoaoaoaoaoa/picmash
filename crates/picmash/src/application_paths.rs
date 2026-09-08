use anyhow::Result;
pub use eternalist_apps::ApplicationPaths;
use eternalist_apps::ProductIdentity;
use std::path::PathBuf;

pub const PRODUCT: ProductIdentity = ProductIdentity::declare(
    picmash_contract::PRODUCT_IDENTIFIER,
    picmash_contract::PRODUCT_NAME,
);

/// Resolve the platform directories and create every root Picmash writes.
pub fn claim() -> Result<ApplicationPaths> {
    let paths = ApplicationPaths::claim(PRODUCT)?;
    paths.prepare()?;
    Ok(paths)
}

/// Picmash's files beneath the platform directories.
pub trait PicmashPaths {
    fn database_path(&self) -> PathBuf;
    fn active_collection_path(&self) -> PathBuf;
    fn remote_database_path(&self) -> PathBuf;
    fn remote_cache_dir(&self) -> PathBuf;
    fn configuration_path(&self) -> PathBuf;
    fn legacy_configuration_path(&self) -> PathBuf;
}

impl PicmashPaths for ApplicationPaths {
    fn database_path(&self) -> PathBuf {
        self.local_data.join("picmash.db")
    }

    fn active_collection_path(&self) -> PathBuf {
        self.state.join("active-collection")
    }

    fn remote_database_path(&self) -> PathBuf {
        self.local_data.join("remote.db")
    }

    fn remote_cache_dir(&self) -> PathBuf {
        self.cache.join("remote")
    }

    fn configuration_path(&self) -> PathBuf {
        self.config.join("picmash.toml")
    }

    fn legacy_configuration_path(&self) -> PathBuf {
        self.config.join("config.toml")
    }
}
