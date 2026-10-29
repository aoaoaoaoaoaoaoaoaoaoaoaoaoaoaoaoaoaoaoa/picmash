//! Tester-independent names crossing Picmash's native UI boundary.

use std::{borrow::Cow, fmt};

/// Reverse-DNS product identifier; the sole source of Picmash's platform identity.
pub const PRODUCT_IDENTIFIER: &str = "moe.eternalist.picmash";
/// Product display name.
pub const PRODUCT_NAME: &str = "Picmash";
/// Native UI contract revision.
pub const UI_FINGERPRINT: &str = "picmash.ui/10";

/// One member of the active comparison.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Side {
    /// Left image.
    Left,
    /// Right image.
    Right,
}

impl Side {
    /// Stable wire spelling.
    #[must_use]
    pub const fn wire(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// The direction of one quarter turn.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Turn {
    /// A quarter turn against the clock.
    Counterclockwise,
    /// A quarter turn with the clock.
    Clockwise,
}

impl Turn {
    /// Stable wire spelling.
    #[must_use]
    pub const fn wire(self) -> &'static str {
        match self {
            Self::Counterclockwise => "counterclockwise",
            Self::Clockwise => "clockwise",
        }
    }
}

/// Stable interactive surface identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Target {
    /// Open the platform collection chooser.
    OpenCollection,
    /// Rescan the active collection.
    Rescan,
    /// Enter the comparison chamber.
    CompareMode,
    /// Enter the collection browse chamber.
    BrowseMode,
    /// An image whose activation chooses a comparison winner.
    Choice(Side),
    /// The keycap beneath one compared image, pressed to choose it.
    ChoiceKey(Side),
    /// Favorite actuator for one compared image.
    Favorite(Side),
    /// Hide actuator for one compared image.
    Hide(Side),
    /// Reject the displayed remote candidate.
    Reject(Side),
    /// Quarter-turn actuator for one compared image.
    Rotate(Side, Turn),
    /// Latch beneath a remote challenger holding the chamber to its stream.
    Lock(Side),
    /// The stream panel's latch holding the chamber to its stream.
    StreamLock,
    /// Reject every candidate belonging to a remote stream.
    RejectStream(Side),
    /// Collection browser surface.
    Browser,
    /// Browse scope that admits every image.
    ScopeAll,
    /// Browse scope that admits the images admitted from no source.
    ScopeUnsourced,
    /// Browse scope for the n-th listed source.
    ScopeSource(usize),
    /// Browse scope for the n-th listed stream of the chosen source.
    ScopeStream(usize),
    /// One visible browser tile.
    BrowseTile,
    /// Full-image viewer surface.
    Viewer,
    /// Copy the viewed image to the clipboard.
    ViewerCopy,
    /// Close the full-image viewer.
    ViewerClose,
    /// Browse the stream the viewed image was admitted from.
    ViewerStream,
}

impl Target {
    /// Stable witness name.
    #[must_use]
    pub fn wire(&self) -> Cow<'static, str> {
        match self {
            Self::OpenCollection => Cow::Borrowed("picmash.collection.open"),
            Self::Rescan => Cow::Borrowed("picmash.collection.rescan"),
            Self::CompareMode => Cow::Borrowed("picmash.mode.compare"),
            Self::BrowseMode => Cow::Borrowed("picmash.mode.browse"),
            Self::Choice(side) => Cow::Owned(format!("picmash.comparison.choice/{}", side.wire())),
            Self::ChoiceKey(side) => {
                Cow::Owned(format!("picmash.comparison.choice-key/{}", side.wire()))
            }
            Self::Favorite(side) => {
                Cow::Owned(format!("picmash.comparison.favorite/{}", side.wire()))
            }
            Self::Hide(side) => Cow::Owned(format!("picmash.comparison.hide/{}", side.wire())),
            Self::Reject(side) => Cow::Owned(format!("picmash.comparison.reject/{}", side.wire())),
            Self::Rotate(side, turn) => Cow::Owned(format!(
                "picmash.comparison.rotate/{}/{}",
                side.wire(),
                turn.wire()
            )),
            Self::Lock(side) => Cow::Owned(format!("picmash.comparison.lock/{}", side.wire())),
            Self::StreamLock => Cow::Borrowed("picmash.stream.lock"),
            Self::RejectStream(side) => {
                Cow::Owned(format!("picmash.comparison.reject-stream/{}", side.wire()))
            }
            Self::Browser => Cow::Borrowed("picmash.collection.browser"),
            Self::ScopeAll => Cow::Borrowed("picmash.collection.scope.all"),
            Self::ScopeUnsourced => Cow::Borrowed("picmash.collection.scope.unsourced"),
            Self::ScopeSource(ordinal) => {
                Cow::Owned(format!("picmash.collection.scope.source/{ordinal}"))
            }
            Self::ScopeStream(ordinal) => {
                Cow::Owned(format!("picmash.collection.scope.stream/{ordinal}"))
            }
            Self::BrowseTile => Cow::Borrowed("picmash.collection.browser.tile"),
            Self::Viewer => Cow::Borrowed("picmash.collection.viewer"),
            Self::ViewerCopy => Cow::Borrowed("picmash.collection.viewer.copy"),
            Self::ViewerClose => Cow::Borrowed("picmash.collection.viewer.close"),
            Self::ViewerStream => Cow::Borrowed("picmash.collection.viewer.stream"),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.wire())
    }
}
