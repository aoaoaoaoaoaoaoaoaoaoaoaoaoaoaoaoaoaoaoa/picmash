//! Tester-independent names crossing Picmash's native UI boundary.

use std::{borrow::Cow, fmt};

/// Native UI contract revision.
pub const UI_FINGERPRINT: &str = "picmash.ui/5";

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
    /// Favorite actuator for one compared image.
    Favorite(Side),
    /// Hide actuator for one compared image.
    Hide(Side),
    /// Reject the displayed remote candidate.
    Reject(Side),
    /// Clockwise rotation actuator for one compared image.
    Rotate(Side),
    /// Reject every candidate belonging to a remote stream.
    RejectStream(Side),
    /// Collection browser surface.
    Browser,
    /// One visible browser tile.
    BrowseTile,
    /// Full-image viewer surface.
    Viewer,
    /// Copy the viewed image to the clipboard.
    ViewerCopy,
    /// Close the full-image viewer.
    ViewerClose,
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
            Self::Favorite(side) => {
                Cow::Owned(format!("picmash.comparison.favorite/{}", side.wire()))
            }
            Self::Hide(side) => Cow::Owned(format!("picmash.comparison.hide/{}", side.wire())),
            Self::Reject(side) => Cow::Owned(format!("picmash.comparison.reject/{}", side.wire())),
            Self::Rotate(side) => Cow::Owned(format!("picmash.comparison.rotate/{}", side.wire())),
            Self::RejectStream(side) => {
                Cow::Owned(format!("picmash.comparison.reject-stream/{}", side.wire()))
            }
            Self::Browser => Cow::Borrowed("picmash.collection.browser"),
            Self::BrowseTile => Cow::Borrowed("picmash.collection.browser.tile"),
            Self::Viewer => Cow::Borrowed("picmash.collection.viewer"),
            Self::ViewerCopy => Cow::Borrowed("picmash.collection.viewer.copy"),
            Self::ViewerClose => Cow::Borrowed("picmash.collection.viewer.close"),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.wire())
    }
}
