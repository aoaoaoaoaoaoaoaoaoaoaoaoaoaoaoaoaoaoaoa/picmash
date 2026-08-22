use std::{fmt, str::FromStr};

use thiserror::Error;
use ulid::Ulid;

#[derive(Debug, Error)]
#[error("{kind} cannot be empty")]
pub struct InvalidId {
    kind: &'static str,
}

macro_rules! text_id {
    ($name:ident, $kind:literal) => {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            #[must_use]
            pub fn fresh() -> Self {
                Self(Ulid::new().to_string())
            }

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

text_id!(AssetId, "asset id");
text_id!(SessionId, "session id");
text_id!(PromptId, "prompt id");
text_id!(CommandId, "command id");
integer_id!(CollectionId);
integer_id!(OccurrenceId);
integer_id!(ObservationId);
integer_id!(SnapshotId);
