use eternalist_apps::configuration::Configuration;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub living_water: bool,
    pub images_per_row: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            living_water: true,
            images_per_row: 5,
        }
    }
}

impl Configuration for Config {}
