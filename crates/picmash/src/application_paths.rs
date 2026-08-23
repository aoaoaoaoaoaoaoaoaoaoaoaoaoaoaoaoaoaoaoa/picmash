use anyhow::{Context as _, Result, bail};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ApplicationPaths {
    pub config: PathBuf,
    pub cache: PathBuf,
    pub data: PathBuf,
    pub state: PathBuf,
}

impl ApplicationPaths {
    pub fn claim() -> Result<Self> {
        let Some(dirs) = ProjectDirs::from("moe", "eternalist", "picmash") else {
            bail!("could not resolve Picmash's platform directories");
        };
        let paths = Self {
            config: dirs.config_dir().to_path_buf(),
            cache: dirs.cache_dir().to_path_buf(),
            data: dirs.data_local_dir().to_path_buf(),
            state: dirs
                .state_dir()
                .map_or_else(|| dirs.data_local_dir().join("state"), Path::to_path_buf),
        };
        for path in [&paths.config, &paths.cache, &paths.data, &paths.state] {
            std::fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
        }
        Ok(paths)
    }

    pub fn database_path(&self) -> PathBuf {
        self.data.join("picmash.db")
    }

    pub fn active_collection_path(&self) -> PathBuf {
        self.state.join("active-collection")
    }

    pub fn remote_database_path(&self) -> PathBuf {
        self.data.join("remote.db")
    }

    pub fn remote_cache_dir(&self) -> PathBuf {
        self.cache.join("remote")
    }

    pub fn configuration_path(&self) -> PathBuf {
        self.config.join("picmash.toml")
    }

    pub fn legacy_configuration_path(&self) -> PathBuf {
        self.config.join("config.toml")
    }
}
