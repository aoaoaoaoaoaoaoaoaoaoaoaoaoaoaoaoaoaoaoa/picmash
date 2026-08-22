//! Tester-independent names crossing Picmash's native UI boundary.

use std::{borrow::Cow, fmt};

/// Native UI contract revision.
pub const UI_FINGERPRINT: &str = "picmash.ui/3";

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
    /// Clockwise rotation actuator for one compared image.
    Rotate(Side),
    /// Reject every candidate belonging to a remote stream.
    VetoStream(Side),
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
            Self::OpenCollection => Cow::Borrowed("collection.open"),
            Self::Rescan => Cow::Borrowed("collection.rescan"),
            Self::CompareMode => Cow::Borrowed("mode.compare"),
            Self::BrowseMode => Cow::Borrowed("mode.browse"),
            Self::Choice(side) => Cow::Owned(format!("comparison.choice/{}", side.wire())),
            Self::Favorite(side) => Cow::Owned(format!("comparison.favorite/{}", side.wire())),
            Self::Hide(side) => Cow::Owned(format!("comparison.hide/{}", side.wire())),
            Self::Rotate(side) => Cow::Owned(format!("comparison.rotate/{}", side.wire())),
            Self::VetoStream(side) => Cow::Owned(format!("comparison.veto_stream/{}", side.wire())),
            Self::Browser => Cow::Borrowed("collection.browser"),
            Self::BrowseTile => Cow::Borrowed("collection.browser/tile"),
            Self::Viewer => Cow::Borrowed("collection.viewer"),
            Self::ViewerCopy => Cow::Borrowed("collection.viewer/copy"),
            Self::ViewerClose => Cow::Borrowed("collection.viewer/close"),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.wire())
    }
}
