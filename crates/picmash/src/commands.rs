use std::sync::OnceLock;

use eternalist_apps::{
    command_guide::{GuideGesture, GuideGroup},
    commands::{CommandCanon, CommandScope, CommandSpec, Shortcut, ShortcutKey, ShortcutModifiers},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Edict {
    ChooseLeft,
    ChooseRight,
    RejectRemote,
    RejectStream,
    LockStream,
    CopyViewer,
    OpenCollection,
    Rescan,
    Compare,
    Browse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Context {
    Compare,
    Browse,
    Viewer,
}

const CHOOSE_LEFT: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('A'),
)];
const CHOOSE_RIGHT: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('D'),
)];
const REJECT_REMOTE: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('X'),
)];
const LOCK_STREAM: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('L'),
)];
const COPY_VIEWER: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('C'),
)];
const OPEN_COLLECTION: [Shortcut; 1] = [Shortcut::primary('O')];
const RESCAN: [Shortcut; 1] = [Shortcut::primary('R')];
const COMPARE: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('1'),
)];
const BROWSE: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('2'),
)];

const EDICTS: [CommandSpec<Edict, Context>; 10] = [
    CommandSpec::new(
        Edict::ChooseLeft,
        "comparison.choose_left",
        "Choose left image",
        CommandScope::Context(Context::Compare),
    )
    .with_detail("Records the left rendering as preferred and advances the comparison.")
    .with_default_shortcuts(&CHOOSE_LEFT),
    CommandSpec::new(
        Edict::ChooseRight,
        "comparison.choose_right",
        "Choose right image",
        CommandScope::Context(Context::Compare),
    )
    .with_detail("Records the right rendering as preferred and advances the comparison.")
    .with_default_shortcuts(&CHOOSE_RIGHT),
    CommandSpec::new(
        Edict::RejectRemote,
        "comparison.reject_remote",
        "Reject",
        CommandScope::Context(Context::Compare),
    )
    .with_detail("Rejects the displayed remote image and advances the comparison.")
    .with_default_shortcuts(&REJECT_REMOTE),
    CommandSpec::new(
        Edict::RejectStream,
        "comparison.reject_stream",
        "Reject Stream",
        CommandScope::Context(Context::Compare),
    )
    .with_detail("Rejects this remote thread and every candidate it contains.")
    .with_mnemonic('J'),
    CommandSpec::new(
        Edict::LockStream,
        "comparison.lock_stream",
        "Lock Stream",
        CommandScope::Context(Context::Compare),
    )
    .with_detail(
        "Holds the comparison to the displayed remote stream until it is released, rejected, \
         or exhausted; pressed again, releases it.",
    )
    .with_default_shortcuts(&LOCK_STREAM),
    CommandSpec::new(
        Edict::CopyViewer,
        "viewer.copy",
        "Copy",
        CommandScope::Context(Context::Viewer),
    )
    .with_detail("Copies the full-resolution image to the clipboard.")
    .with_default_shortcuts(&COPY_VIEWER),
    CommandSpec::new(
        Edict::OpenCollection,
        "collection.open",
        "Open collection",
        CommandScope::Global,
    )
    .with_detail("Chooses a local directory and makes it the active image collection.")
    .with_default_shortcuts(&OPEN_COLLECTION),
    CommandSpec::new(
        Edict::Rescan,
        "collection.rescan",
        "Rescan collection",
        CommandScope::Global,
    )
    .with_detail("Reconciles the active collection with its files without changing them.")
    .with_default_shortcuts(&RESCAN),
    CommandSpec::new(
        Edict::Compare,
        "mode.compare",
        "Compare",
        CommandScope::Global,
    )
    .with_detail("Enters the pairwise preference chamber.")
    .with_default_shortcuts(&COMPARE),
    CommandSpec::new(
        Edict::Browse,
        "mode.browse",
        "Browse collection",
        CommandScope::Global,
    )
    .with_detail("Browses the active collection in learned preference order.")
    .with_default_shortcuts(&BROWSE),
];

const COMPARISON_GESTURES: [GuideGesture; 5] = [
    GuideGesture::new(
        "Choose a winner",
        "Click an image, or press A for left and D for right.",
        &[CHOOSE_LEFT[0], CHOOSE_RIGHT[0]],
    ),
    GuideGesture::new(
        "Favorite an image",
        "Use the heart beneath either image without advancing the comparison.",
        &[],
    ),
    GuideGesture::new(
        "Rotate an image",
        "Use ↶ or ↷ beneath either image; the corrected presentation persists.",
        &[],
    ),
    GuideGesture::new(
        "Hide an image",
        "Use 👁 beneath a local image to withdraw it without deleting its file.",
        &[],
    ),
    GuideGesture::new(
        "Lock a stream",
        "Use 🔒 beneath a remote challenger to hold the comparison to its stream; \
         Picmash waits for the stream's next image and releases the lock once it is exhausted.",
        &LOCK_STREAM,
    ),
];
const COLLECTION_GESTURES: [GuideGesture; 3] = [
    GuideGesture::new(
        "Open collection",
        "Selects another local image directory.",
        &OPEN_COLLECTION,
    ),
    GuideGesture::new(
        "Rescan collection",
        "Reconciles added, changed, removed, and unreadable files.",
        &RESCAN,
    ),
    GuideGesture::new(
        "Change chamber",
        "1 opens Compare; 2 opens Browse.",
        &[COMPARE[0], BROWSE[0]],
    ),
];

const COMPARISON: GuideGroup = GuideGroup::new("COMPARISON", &COMPARISON_GESTURES);
const COLLECTION: GuideGroup = GuideGroup::new("COLLECTION", &COLLECTION_GESTURES);

pub const COMPARE_GUIDE_GROUPS: [GuideGroup; 2] = [COMPARISON, COLLECTION];
pub const BROWSE_GUIDE_GROUPS: [GuideGroup; 1] = [COLLECTION];

pub fn canon() -> &'static CommandCanon<Edict, Context> {
    static CANON: OnceLock<CommandCanon<Edict, Context>> = OnceLock::new();
    CANON.get_or_init(|| CommandCanon::new(&EDICTS))
}
