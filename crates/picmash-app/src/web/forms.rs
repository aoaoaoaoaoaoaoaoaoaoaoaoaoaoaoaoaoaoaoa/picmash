use std::str::FromStr;

use anyhow::bail;
use serde::Deserialize;

use crate::{
    asset_domain::AssetDomainLabel,
    model::{AssetId, ExploreMapMode},
};

#[derive(Debug, Deserialize)]
pub(super) struct ArenaCommandFields {
    pub(super) command_id: String,
    pub(super) turn_id: String,
    pub(super) action_token: String,
    #[serde(deserialize_with = "deserialize_form_u64")]
    pub(super) arena_revision: u64,
    #[serde(deserialize_with = "deserialize_form_u64")]
    pub(super) arena_sampler_epoch: u64,
}

#[derive(Debug, Deserialize)]
pub(super) struct VoteForm {
    #[serde(flatten)]
    pub(super) command: ArenaCommandFields,
    pub(super) left_id: String,
    pub(super) right_id: String,
    pub(super) winner_id: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct RotateForm {
    pub(super) asset_id: String,
    pub(super) direction: i32,
    #[serde(rename = "next_rotation")]
    pub(super) next_rotation: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct NudgeForm {
    pub(super) asset_id: String,
    pub(super) delta: i32,
}

#[derive(Debug, Deserialize)]
pub(super) struct HeartAssetForm {
    pub(super) asset_id: String,
    pub(super) active: bool,
}

#[derive(Debug, Deserialize)]
pub(super) struct HideAssetForm {
    pub(super) asset_id: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct AssetDomainForm {
    pub(super) asset_id: String,
    pub(super) label: AssetDomainLabel,
}

#[derive(Debug, Deserialize)]
pub(super) struct FacemashVoteForm {
    pub(super) winner_face_id: String,
    pub(super) loser_face_id: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct FacemashHideForm {
    pub(super) face_id: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct IdentityThresholdForm {
    pub(super) percent: u8,
}

#[derive(Debug, Deserialize)]
pub(super) struct IdentityNameForm {
    pub(super) anchor_handle: String,
    pub(super) name: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct IdentityPairForm {
    pub(super) pair_handle: String,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct IdentitiesQuery {
    pub(super) anchor: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct HideForm {
    #[serde(flatten)]
    pub(super) command: ArenaCommandFields,
    pub(super) asset_id: String,
    pub(super) hide: bool,
    #[serde(default, deserialize_with = "deserialize_comma_ids")]
    pub(super) cluster_ids: Vec<i64>,
}

fn deserialize_comma_ids<'de, D>(deserializer: D) -> Result<Vec<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|s| s.trim().parse::<i64>().map_err(serde::de::Error::custom))
        .collect()
}

fn deserialize_form_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    raw.parse::<u64>().map_err(serde::de::Error::custom)
}

#[derive(Debug, Deserialize)]
pub(super) struct HandleForm {
    #[serde(flatten)]
    pub(super) command: ArenaCommandFields,
    pub(super) asset_id: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ThreadLockForm {
    #[serde(flatten)]
    pub(super) command: ArenaCommandFields,
    pub(super) asset_id: String,
    pub(super) active: bool,
}

#[derive(Debug, Deserialize)]
pub(super) struct ExternalProbabilityForm {
    pub(super) percent: u8,
}

#[derive(Debug, Deserialize)]
pub(super) struct ArenaExploreForm {
    pub(super) percent: u8,
}

#[derive(Debug, Deserialize)]
pub(super) struct DedupRadiusForm {
    pub(super) percent: u8,
}

#[derive(Debug, Deserialize)]
pub(super) struct FacemashMinFaceSideForm {
    pub(super) px: u16,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct ClientRouteQuery {
    pub(super) focus: Option<String>,
    pub(super) mode: Option<String>,
    pub(super) triad: Option<String>,
}

impl ClientRouteQuery {
    pub(super) fn map_mode(&self) -> ExploreMapMode {
        self.mode
            .as_deref()
            .and_then(|value| ExploreMapMode::from_str(value).ok())
            .unwrap_or_default()
    }

    pub(super) fn focus_asset_id(&self) -> Option<AssetId> {
        self.focus
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(|value| AssetId(value.to_owned()))
    }

    pub(super) fn triad_assets(&self) -> anyhow::Result<Option<[AssetId; 3]>> {
        self.triad
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(parse_triad_query)
            .transpose()
    }
}

fn parse_triad_query(raw: &str) -> anyhow::Result<[AssetId; 3]> {
    let parts = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| AssetId(part.to_owned()))
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("triad query must contain exactly three asset ids");
    }
    if parts[0] == parts[1] || parts[0] == parts[2] || parts[1] == parts[2] {
        bail!("triad query must contain distinct asset ids");
    }
    Ok([parts[0].clone(), parts[1].clone(), parts[2].clone()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_command_fields_accept_stringified_form_numbers() {
        let fields: ArenaCommandFields = serde_json::from_value(serde_json::json!({
            "command_id": "cmd",
            "turn_id": "turn",
            "action_token": "token",
            "arena_revision": "42",
            "arena_sampler_epoch": "7",
        }))
        .expect("deserialize arena command fields");

        assert_eq!(fields.arena_revision, 42);
        assert_eq!(fields.arena_sampler_epoch, 7);
    }
}
