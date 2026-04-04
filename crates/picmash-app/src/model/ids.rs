use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorpusId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RemoteItemId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FaceId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FaceIdentityId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArenaHandle {
    Local(AssetId),
    Remote(RemoteItemId),
}

impl ArenaHandle {
    #[must_use]
    pub fn slug(&self) -> String {
        match self {
            Self::Local(asset_id) => format!("asset_{}", asset_id.0),
            Self::Remote(item_id) => format!("remote_{}", item_id.0),
        }
    }
}

impl std::str::FromStr for ArenaHandle {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(asset_id) = value.strip_prefix("asset_") {
            return Ok(Self::Local(AssetId(asset_id.to_owned())));
        }
        if let Some(remote_id) = value.strip_prefix("remote_") {
            let parsed = remote_id
                .parse::<i64>()
                .map_err(|_| "invalid remote arena handle")?;
            return Ok(Self::Remote(RemoteItemId(parsed)));
        }
        Err("unknown arena handle")
    }
}

#[cfg(test)]
mod tests {
    use super::AssetId;

    #[test]
    fn asset_id_roundtrips_clone() {
        let asset = AssetId("abc".to_owned());
        assert_eq!(asset, asset.clone());
    }
}
