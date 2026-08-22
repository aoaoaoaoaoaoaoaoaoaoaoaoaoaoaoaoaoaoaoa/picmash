use std::sync::OnceLock;

use eternalist_apps::{
    command_guide::{GuideGesture, GuideSection},
    commands::{CommandCanon, CommandScope, CommandSpec, Shortcut, ShortcutKey, ShortcutModifiers},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Edict {
    ChooseLeft,
    ChooseRight,
    OpenCollection,
    Rescan,
    Compare,
    Browse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Context {
    Compare,
    Browse,
}

const CHOOSE_LEFT: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('A'),
)];
const CHOOSE_RIGHT: [Shortcut; 1] = [Shortcut::new(
    ShortcutModifiers::NONE,
    ShortcutKey::Character('D'),
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

const EDICTS: [CommandSpec<Edict, Context>; 6] = [
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

const COMPARISON_GESTURES: [GuideGesture; 4] = [
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
        "Use its clockwise actuator; the corrected presentation persists.",
        &[],
    ),
    GuideGesture::new(
        "Hide an image",
        "Use its eye actuator to withdraw it from comparisons without deleting its file.",
        &[],
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

const COMPARISON: GuideSection = GuideSection::new("COMPARISON", &COMPARISON_GESTURES);
const COLLECTION: GuideSection = GuideSection::new("COLLECTION", &COLLECTION_GESTURES);

pub const COMPARE_GUIDE: [GuideSection; 2] = [COMPARISON, COLLECTION];
pub const BROWSE_GUIDE: [GuideSection; 1] = [COLLECTION];

pub fn canon() -> &'static CommandCanon<Edict, Context> {
    static CANON: OnceLock<CommandCanon<Edict, Context>> = OnceLock::new();
    CANON.get_or_init(|| CommandCanon::new(&EDICTS))
}
