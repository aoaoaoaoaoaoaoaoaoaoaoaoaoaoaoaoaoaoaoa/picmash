use anyhow::{Context as _, Result, bail};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Lair {
    pub config: PathBuf,
    pub data: PathBuf,
    pub state: PathBuf,
}

impl Lair {
    pub fn claim() -> Result<Self> {
        let Some(dirs) = ProjectDirs::from("moe", "eternalist", "picmash") else {
            bail!("could not resolve Picmash's platform directories");
        };
        let lair = Self {
            config: dirs.config_dir().to_path_buf(),
            data: dirs.data_local_dir().to_path_buf(),
            state: dirs
                .state_dir()
                .map_or_else(|| dirs.data_local_dir().join("state"), Path::to_path_buf),
        };
        for path in [&lair.config, &lair.data, &lair.state] {
            std::fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
        }
        Ok(lair)
    }

    pub fn database(&self) -> PathBuf {
        self.data.join("picmash.db")
    }

    pub fn active_collection(&self) -> PathBuf {
        self.state.join("active-collection")
    }

    pub fn configuration(&self) -> PathBuf {
        self.config.join("picmash.toml")
    }
}
