use anyhow::Result;
use brass_poolrooms::{
    chrome::{
        self, Checkbox, MechanismSize, Monoglyph, MonoglyphFinish, MonoglyphResponse, Symbol,
    },
    water::{Domain, Floor, Surface, Wetness},
};
use crossbeam_channel::{Receiver, bounded};
use egui::{ColorImage, TextureHandle, TextureOptions};
use eternalist_apps::{
    ApplicationHeader, Inspector, LivingWait, NativeWake,
    command_guide::CommandGuide,
    commands::{CommandDispatch, CommandStatus, ShortcutKey, ShortcutModifiers},
    configuration::ConfigurationLedger,
    panel_navigation::PanelNavigator,
    settings::{SettingSpec, SettingsFile, SettingsSheet},
};
use picmash_contract::{Side, Target, Turn};
use picmash_engine::{AssetId, ScanProgress};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use crate::{
    application_paths::{self, PicmashPaths as _},
    commands::{self, Edict},
    configuration::{Configuration, Probability, ReservoirCapacity},
    facets::{Facets, Scope, source_label},
    remote::Summary as RemoteSummary,
    viewer::{Action as ViewerAction, Viewer},
    witness,
    worker::{Blade, Card, Command, Event, Pair, PairCard, Summary, Worker},
};

const EVENT_DRAIN: usize = 24;
const CONFIG_SETTLE: Duration = Duration::from_millis(400);
const MIN_IMAGES_PER_ROW: u16 = 1;
const MAX_IMAGES_PER_ROW: u16 = 12;
const MIN_TILE_EDGE: f32 = 72.0;
const TILE_GAP: f32 = 12.0;
/// The one gauge of every duel-rail actuator, keycaps included.
const RAIL_GAUGE: MechanismSize = MechanismSize::Medium;
const DUEL_CONTROL_HEIGHT: f32 = RAIL_GAUGE.side();
const DUEL_PANEL_HEIGHT: f32 = DUEL_CONTROL_HEIGHT + 8.0;
const DUEL_PANEL_BLEED: f32 = 2.0;
const BRONZE_RIM: f32 = 4.0;
/// How long a judged pair holds the chamber before the wait for its
/// successor shows: a successor forged ahead arrives well within it.
const CHOICE_GRACE: Duration = Duration::from_millis(250);
const WATER: SettingSpec = SettingSpec::new(
    "living_water",
    "LIVING WATER",
    "Let controls and image choices displace the chamber's one water body.",
);
const REMOTE: SettingSpec = SettingSpec::new(
    "remote.enabled",
    "REMOTE SOURCES",
    "Maintain a bounded reservoir of challengers from configured sources.",
);
const REMOTE_CHANCE: SettingSpec = SettingSpec::new(
    "remote.sample_probability",
    "REMOTE CHANCE",
    "Chance that the next comparison draws a ready remote challenger.",
);
const REMOTE_RESERVOIR: SettingSpec = SettingSpec::new(
    "remote.reservoir_capacity",
    "REMOTE RESERVOIR",
    "Hard cap on downloaded candidates, including fetching and displayed images.",
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Compare,
    Browse,
}

impl Mode {
    const fn context(self) -> commands::Context {
        match self {
            Self::Compare => commands::Context::Compare,
            Self::Browse => commands::Context::Browse,
        }
    }

    #[cfg(feature = "egui-test")]
    const fn wire(self) -> &'static str {
        match self {
            Self::Compare => "compare",
            Self::Browse => "browse",
        }
    }
}

struct PairView {
    pair: Pair,
    left_texture: TextureHandle,
    right_texture: TextureHandle,
}

/// A judgment sent to the worker, holding its pair until the next arrives.
#[derive(Clone, Copy)]
struct Choice {
    sent: Instant,
    status: &'static str,
}

#[derive(Clone, Copy)]
enum Action {
    Choose(Side),
    Favorite(Side),
    Hide(Side),
    RejectRemote,
    RejectStream,
    /// Hold the chamber to the displayed stream, or release it.
    LockStream(bool),
    Rotate(Side, Turn),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ThumbKey {
    asset_id: AssetId,
    bucket: u8,
}

pub struct Picmash {
    worker: Worker,
    chooser: Option<Receiver<Option<PathBuf>>>,
    mode: Mode,
    summary: Option<Summary>,
    remote_summary: RemoteSummary,
    pending_collection: Option<PathBuf>,
    cards: Vec<Card>,
    browse_indices: Vec<usize>,
    images_per_row: u16,
    browse_scroll_offset: f32,
    favorites_only: bool,
    scope: Scope,
    /// Facets drawn in the browser. A fresh tally waits in `pending_facets`
    /// while the pointer rests on them, so no row moves under it.
    facets: Facets,
    pending_facets: Option<Facets>,
    facets_hovered: bool,
    stream_filter: String,
    /// The browser returns to its first row on the next frame.
    browse_rewind: bool,
    pair: Option<PairView>,
    thumbnails: HashMap<ThumbKey, TextureHandle>,
    thumbnails_inflight: HashSet<ThumbKey>,
    /// Thumbnails the worker failed to develop, never demanded again until a
    /// rescan or another collection.
    thumbnails_faulted: HashSet<ThumbKey>,
    viewer: Option<Viewer>,
    busy: bool,
    /// The judgment awaiting its next pair.
    choice: Option<Choice>,
    /// The chamber shows the worker's wait this frame.
    wait_shown: bool,
    /// The chamber waits for the locked stream's next challenger.
    awaiting: bool,
    /// A stream lock state requested of the worker and not yet confirmed.
    lock_pending: Option<bool>,
    scan_progress: Option<ScanProgress>,
    status: String,
    panels: PanelNavigator,
    guide: CommandGuide,
    settings: SettingsSheet,
    configuration: ConfigurationLedger<Configuration>,
    living_wait: LivingWait,
    water: Surface,
    /// Visible browse tiles lacking a texture at their requested resolution,
    /// counted by the frame that painted the browser; `None` in any other.
    tiles_wanting: Option<usize>,
}

impl Picmash {
    pub fn open(ctx: &egui::Context, initial: Option<PathBuf>) -> Result<Self> {
        let paths = application_paths::claim()?;
        let fallback = Configuration::legacy_fallback(&paths.legacy_configuration_path())?;
        let configuration: ConfigurationLedger<Configuration> =
            ConfigurationLedger::raise_with_fallback(
                "picmash-configuration",
                ctx,
                paths.configuration_path(),
                CONFIG_SETTLE,
                fallback,
            )?;
        chrome::set_font_scale(ctx, configuration.live().font_scale);
        let wetness = if configuration.live().living_water {
            Wetness::Wet
        } else {
            Wetness::Dry
        };
        let images_per_row = configuration
            .live()
            .images_per_row
            .clamp(MIN_IMAGES_PER_ROW, MAX_IMAGES_PER_ROW);
        let worker = Worker::spawn(ctx, paths, initial, configuration.live().remote.clone())?;
        Ok(Self {
            worker,
            chooser: None,
            mode: Mode::Compare,
            summary: None,
            remote_summary: RemoteSummary::default(),
            awaiting: false,
            lock_pending: None,
            pending_collection: None,
            cards: Vec::new(),
            browse_indices: Vec::new(),
            images_per_row,
            browse_scroll_offset: 0.0,
            favorites_only: false,
            scope: Scope::All,
            facets: Facets::default(),
            pending_facets: None,
            facets_hovered: false,
            stream_filter: String::new(),
            browse_rewind: false,
            pair: None,
            thumbnails: HashMap::new(),
            thumbnails_inflight: HashSet::new(),
            thumbnails_faulted: HashSet::new(),
            viewer: None,
            busy: true,
            choice: None,
            wait_shown: true,
            scan_progress: None,
            status: "WAKING ENGINE".to_owned(),
            panels: PanelNavigator::default(),
            guide: CommandGuide::default(),
            settings: SettingsSheet::default(),
            configuration,
            living_wait: LivingWait::default(),
            water: Surface::new(wetness),
            tiles_wanting: None,
        })
    }

    pub fn pulse(&mut self, ui: &mut egui::Ui) {
        self.tiles_wanting = None;
        let ctx = ui.ctx().clone();
        chrome::set_font_scale(&ctx, self.configuration.live().font_scale);
        self.absorb_chooser(&ctx);
        self.drain(&ctx);
        self.poll_viewer_copy(&ctx);
        if self.configuration.absorb() {
            self.adopt_configuration();
        }
        if self.configuration.fault().is_some() {
            self.settings.require_attention(&ctx);
        }
        let settings_invoked = !self.guide.is_open() && self.settings.take_shortcut(&ctx);
        let guide_invoked =
            !settings_invoked && !self.settings.is_open() && self.guide.take_shortcuts(&ctx);
        self.zoom_tiles(&ctx);
        if !settings_invoked
            && !guide_invoked
            && !self.settings.is_open()
            && !self.guide.is_open()
            && self.chooser.is_none()
            && self.viewer.is_none()
            && let Some(dispatch) = commands::canon().route(&ctx, &[self.mode.context()], |edict| {
                self.edict_status(edict)
            })
        {
            self.apply_edict(&ctx, dispatch);
        }
        self.paint(ui);
        self.show_viewer(&ctx);
        self.command_guide(&ctx);
        self.show_settings(&ctx);
    }

    pub fn configuration_deadline(&self) -> Option<Instant> {
        self.configuration.deadline()
    }

    pub fn service_configuration(&mut self, now: Instant) -> bool {
        let changed = self.configuration.service_deadline_reached(now);
        if changed {
            self.adopt_configuration();
        }
        changed
    }

    pub fn water_frame(
        &mut self,
        ctx: &egui::Context,
        pixels_per_point: f32,
        tooltip_rects: &[egui::Rect],
    ) -> brass_poolrooms::water::Frame {
        self.living_wait.compose(ctx, &mut self.water);
        self.water.frame(ctx, pixels_per_point, tooltip_rects, None)
    }

    fn paint(&mut self, ui: &mut egui::Ui) {
        self.wait_shown = self.shows_wait(ui.ctx());
        let mut panels = std::mem::take(&mut self.panels);
        let inspector = Inspector::new("picmash-controls")
            .scroll_id("picmash-controls-scroll")
            .show(ui, |ui| self.inspector(ui, &mut panels));
        self.panels = panels;
        inspector.agitate(&mut self.water);
        let _center = egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(chrome::PAGE)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ui, |ui| self.chamber(ui));
    }

    fn inspector(&mut self, ui: &mut egui::Ui, navigator: &mut PanelNavigator) {
        ui.set_width(ui.available_width());
        let _header = ApplicationHeader::new("PICMASH")
            .settings_attention(self.configuration.fault().is_some())
            .show(ui, &mut self.guide, &mut self.settings, &mut self.water);
        ui.add_space(6.0);
        let mut panels = navigator.frame(ui.ctx());

        let collection = panels.panel(ui, "collection", "COLLECTION", true, |ui| {
            let root = self
                .summary
                .as_ref()
                .map(|summary| &summary.root)
                .or(self.pending_collection.as_ref());
            if let Some(root) = root {
                let _root = ui.label(
                    chrome::TypeRole::Label
                        .text(file_name(root).to_uppercase())
                        .strong()
                        .color(chrome::HOT),
                );
                let _path = ui.label(
                    chrome::TypeRole::Label
                        .text(root.display().to_string())
                        .color(chrome::MUTED),
                );
                ui.add_space(5.0);
            } else {
                let _none = ui.label(chrome::muted("NO COLLECTION CLAIMED"));
                ui.add_space(5.0);
            }
            let open = plate_enabled(ui, !self.wait_shown, "OPEN COLLECTION", false);
            witness::response(ui, Target::OpenCollection, &open);
            if open.clicked() {
                self.open_collection(ui.ctx());
            }
            let rescan = plate_enabled(
                ui,
                self.summary.is_some() && !self.wait_shown,
                "RESCAN",
                false,
            );
            witness::response(ui, Target::Rescan, &rescan);
            if rescan.clicked() {
                self.send(Command::Rescan, true, "SCANNING COLLECTION");
            }
        });
        self.water.fold(collection.wake);

        let mode = panels.panel(ui, "chamber", "CHAMBER", true, |ui| {
            let compare = plate(ui, "01  COMPARE", self.mode == Mode::Compare);
            witness::response(ui, Target::CompareMode, &compare);
            if compare.clicked() {
                self.set_mode(Mode::Compare);
            }
            let browse = plate(ui, "02  BROWSE", self.mode == Mode::Browse);
            witness::response(ui, Target::BrowseMode, &browse);
            if browse.clicked() {
                self.set_mode(Mode::Browse);
            }
        });
        self.water.fold(mode.wake);

        if self.mode == Mode::Browse {
            let view = panels.panel(ui, "view", "VIEW", true, |ui| {
                let before = self.favorites_only;
                let filter = Checkbox::new(&mut self.favorites_only, "FAVORITES ONLY")
                    .size(MechanismSize::Small)
                    .show(ui);
                self.water.checkbox(&filter);
                if before != self.favorites_only {
                    self.rebuild_browse_indices();
                }
                if !self.facets_hovered
                    && let Some(facets) = self.pending_facets.take()
                {
                    self.facets = facets;
                }
                let picked =
                    ui.scope(|ui| self.facets.show(ui, &self.scope, &mut self.stream_filter));
                self.facets_hovered = picked.response.contains_pointer();
                if let Some(scope) = picked.inner {
                    self.narrow(scope);
                }
            });
            self.water.fold(view.wake);
        }

        if self.mode == Mode::Compare
            && let Some((source, title)) = self.held_stream()
        {
            let stream = panels.panel(ui, "stream", "STREAM", true, |ui| {
                let _source = ui.label(
                    chrome::TypeRole::Label
                        .text(source)
                        .strong()
                        .color(chrome::HOT),
                );
                let _title = ui.label(chrome::TypeRole::Label.text(title).color(chrome::MUTED));
                ui.add_space(5.0);
                let held = self.stream_locked();
                let mut latched = held;
                let latch = ui
                    .horizontal(|ui| {
                        let latch = Monoglyph::symbol(Symbol::Lock)
                            .size(MechanismSize::Medium)
                            .show_latched(ui, &mut latched)
                            .on_hover_text(if held {
                                "Release this stream"
                            } else {
                                "Hold the comparison to this stream"
                            });
                        let _label =
                            ui.label(chrome::section_title("LOCK STREAM").color(if held {
                                chrome::HOT
                            } else {
                                chrome::TEXT
                            }));
                        latch
                    })
                    .inner;
                self.water.monoglyph(&latch);
                witness::response(ui, Target::StreamLock, &latch);
                if latch.clicked() && !self.busy {
                    self.apply_action(Action::LockStream(latched));
                }
                if self.awaiting {
                    let _awaiting = ui.label(chrome::muted("AWAITING ITS NEXT IMAGE"));
                }
            });
            self.water.fold(stream.wake);
        }

        let status = panels.panel(ui, "status", "STATUS", true, |ui| {
            if let Some(summary) = &self.summary {
                datum(ui, "VISIBLE", summary.visible_assets.to_string());
                datum(ui, "DUELS", duel_total(&self.cards).to_string());
                datum(ui, "FAVORITES", favorite_total(&self.cards).to_string());
                if summary.scan_failures > 0 {
                    datum(ui, "UNREADABLE", summary.scan_failures.to_string());
                }
                if let Some(evaluation) = summary.evaluation {
                    ui.add_space(4.0);
                    let _term = glossary(
                        ui.label(chrome::eyebrow("CHRONOLOGICAL HOLDOUT")),
                        "CHRONOLOGICAL HOLDOUT",
                        "How well an order fit to your older duels predicts the newest fifth. \
                         It diagnoses the model; it does not grade the images.",
                    );
                    datum(
                        ui,
                        "ACCURACY",
                        format!("{:.0}%", evaluation.accuracy * 100.0),
                    );
                    datum(ui, "LOG LOSS", format!("{:.3}", evaluation.log_loss));
                }
            }
            if self.remote_summary.enabled_sources > 0 || self.remote_summary.promoting > 0 {
                ui.add_space(4.0);
                let _label = ui.label(chrome::eyebrow("REMOTE RESERVOIR"));
                datum(
                    ui,
                    "READY",
                    (self.remote_summary.prepared + usize::from(self.remote_summary.offered))
                        .to_string(),
                );
                datum(ui, "FETCHING", self.remote_summary.fetching.to_string());
                if self.remote_summary.cataloging > 0 {
                    datum(ui, "CATALOGING", self.remote_summary.cataloging.to_string());
                }
                if self.remote_summary.backing_off > 0 {
                    datum(ui, "BACKOFF", self.remote_summary.backing_off.to_string());
                }
                datum(ui, "DISCOVERED", self.remote_summary.discovered.to_string());
                if self.remote_summary.promoting > 0 {
                    datum(ui, "ARCHIVING", self.remote_summary.promoting.to_string());
                }
            }
            ui.add_space(4.0);
            let _status = ui.label(chrome::muted(&self.status));
        });
        self.water.fold(status.wake);
    }

    fn chamber(&mut self, ui: &mut egui::Ui) {
        let arena = ui.available_rect_before_wrap();
        self.water.begin(Domain::shelf(arena));
        let awaiting = self.awaiting && self.mode == Mode::Compare;
        self.water
            .set_floor((self.wait_shown || awaiting).then_some(Floor::shallow(arena)));
        match (self.wait_shown, self.summary.is_some(), self.mode) {
            (true, _, _) => {
                let _rect = match self.scan_progress {
                    Some(progress) => {
                        self.living_wait
                            .bouncer_with(ui, arena, format!("{}%", progress.percent()))
                    }
                    None => self.living_wait.bouncer(ui, arena),
                };
            }
            (false, false, _) => self.first_contact(ui),
            (false, true, Mode::Compare) if awaiting => {
                let _rect = self.living_wait.bouncer_with(ui, arena, "AWAITING");
            }
            (false, true, Mode::Compare) => self.comparison(ui),
            (false, true, Mode::Browse) => self.browser(ui),
        }
    }

    fn first_contact(&mut self, ui: &mut egui::Ui) {
        let arena = ui.available_rect_before_wrap();
        let card = egui::Rect::from_center_size(
            arena.center(),
            egui::vec2(440.0_f32.min(arena.width()), 180.0_f32.min(arena.height())),
        );
        let _placed = ui.scope_builder(egui::UiBuilder::new().max_rect(card), |ui| {
            let _frame = egui::Frame::new()
                .fill(chrome::SURFACE)
                .stroke(egui::Stroke::new(1.0_f32, chrome::EDGE_STRONG))
                .inner_margin(egui::Margin::same(20))
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        let _title = ui.label(chrome::title("CLAIM A COLLECTION"));
                        let _detail = ui.label(chrome::muted(
                            "Picmash reads local images and keeps judgments in its application data directory.",
                        ));
                        ui.add_space(14.0);
                        let open = plate(ui, "OPEN COLLECTION", false);
                        if open.clicked() {
                            self.open_collection(ui.ctx());
                        }
                    });
                });
        });
    }

    fn comparison(&mut self, ui: &mut egui::Ui) {
        let Some(pair) = self.pair.as_ref() else {
            let _empty = ui.centered_and_justified(|ui| {
                ui.label(chrome::title("TWO VISIBLE IMAGES ARE REQUIRED"));
            });
            return;
        };
        let left = (pair.pair.left.clone(), pair.left_texture.clone());
        let right = (pair.pair.right.clone(), pair.right_texture.clone());
        let arena = ui.available_rect_before_wrap();
        let image_arena = egui::Rect::from_min_max(
            arena.min,
            egui::pos2(
                arena.right(),
                (arena.bottom() - DUEL_PANEL_HEIGHT - DUEL_PANEL_BLEED).max(arena.top()),
            ),
        );
        let [left_layout, right_layout] = comparison_layouts(
            image_arena,
            left.1.size_vec2(),
            right.1.size_vec2(),
            ui.spacing().item_spacing.x,
        );
        let _advanced = ui.advance_cursor_after_rect(arena);
        let mut actions = Vec::new();
        if let Some(action) = comparison_image(
            ui,
            &mut self.water,
            Side::Left,
            &left.0,
            &left.1,
            left_layout,
        ) {
            actions.push(action);
        }
        if let Some(action) = comparison_image(
            ui,
            &mut self.water,
            Side::Right,
            &right.0,
            &right.1,
            right_layout,
        ) {
            actions.push(action);
        }
        let locked = self.stream_locked();
        actions.extend(comparison_panel(
            ui,
            &mut self.water,
            Side::Left,
            &left.0,
            locked,
            left_layout.panel,
        ));
        actions.extend(comparison_panel(
            ui,
            &mut self.water,
            Side::Right,
            &right.0,
            locked,
            right_layout.panel,
        ));
        // A judged pair holding the chamber through its grace is inert.
        if !self.busy {
            for action in actions {
                self.apply_action(action);
            }
        }
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width().max(MIN_TILE_EDGE);
        let maximum_columns = (((width + TILE_GAP) / (MIN_TILE_EDGE + TILE_GAP)) as usize).max(1);
        let columns = usize::from(self.images_per_row.max(1)).min(maximum_columns);
        let edge = tile_edge(width, columns);
        let row_height = edge + TILE_GAP;
        let rows = self.browse_indices.len().div_ceil(columns);
        let bucket = thumb_bucket(edge);
        let motion = (!self.guide.is_open()
            && !self.settings.is_open()
            && self.viewer.is_none()
            && !ui.ctx().text_edit_focused())
        .then(|| ui.ctx().input(gallery_motion))
        .flatten();
        let offset = std::mem::take(&mut self.browse_rewind)
            .then_some(0.0)
            .or_else(|| {
                motion.map(|motion| match motion {
                    GalleryMotion::PreviousRow => (self.browse_scroll_offset - row_height).max(0.0),
                    GalleryMotion::NextRow => self.browse_scroll_offset + row_height,
                    GalleryMotion::First => 0.0,
                })
            });
        let mut demands = Vec::<ThumbKey>::new();
        let mut wanting = 0;
        let mut retained = HashSet::new();
        let mut opened = None;
        let scroll = egui::ScrollArea::vertical().id_salt("picmash-browser");
        let scroll = if let Some(offset) = offset {
            scroll.vertical_scroll_offset(offset)
        } else {
            scroll
        };
        let body = scroll.show_rows(ui, row_height, rows, |ui, range| {
            ui.spacing_mut().item_spacing.x = TILE_GAP;
            for row in range {
                let _row = ui.horizontal(|ui| {
                    for column in 0..columns {
                        let slot = row * columns + column;
                        let Some(&card_index) = self.browse_indices.get(slot) else {
                            ui.allocate_space(egui::vec2(edge, 1.0));
                            continue;
                        };
                        let card = &self.cards[card_index];
                        let _retained = retained.insert(card.presentation.asset_id.clone());
                        let key = ThumbKey {
                            asset_id: card.presentation.asset_id.clone(),
                            bucket,
                        };
                        let developed = self.thumbnails.get(&key);
                        let texture = developed.or_else(|| {
                            [2, 1, 0].into_iter().find_map(|resident| {
                                self.thumbnails.get(&ThumbKey {
                                    asset_id: card.presentation.asset_id.clone(),
                                    bucket: resident,
                                })
                            })
                        });
                        if browse_tile(ui, &mut self.water, card, texture, edge, slot == 0) {
                            opened = Some(card.presentation.asset_id.clone());
                        }
                        if developed.is_none() {
                            wanting += 1;
                            if !self.thumbnails_inflight.contains(&key)
                                && !self.thumbnails_faulted.contains(&key)
                            {
                                demands.push(key);
                            }
                        }
                    }
                });
            }
        });
        self.tiles_wanting = Some(wanting);
        self.browse_scroll_offset = body.state.offset.y;
        self.water.heave(ui.ctx(), body.state.offset.y);
        witness::rect(ui.ctx(), Target::Browser, body.inner_rect);
        self.thumbnails
            .retain(|key, _texture| retained.contains(&key.asset_id));
        if let Some(asset_id) = opened {
            self.open_viewer(ui.ctx(), &asset_id);
        }
        for key in demands {
            if self
                .worker
                .send(Command::Thumbnail {
                    asset_id: key.asset_id.clone(),
                    bucket: key.bucket,
                })
                .is_ok()
            {
                let _inserted = self.thumbnails_inflight.insert(key);
            }
        }
    }

    fn edict_status(&self, edict: Edict) -> CommandStatus<'static> {
        let remote = self.remote_shown();
        match edict {
            Edict::ChooseLeft | Edict::ChooseRight if self.busy || self.pair.is_none() => {
                CommandStatus::Disabled("no comparison is ready")
            }
            Edict::Rescan if self.busy || self.summary.is_none() => {
                CommandStatus::Disabled("no collection is ready")
            }
            Edict::OpenCollection if self.busy => CommandStatus::Disabled("Picmash is busy"),
            Edict::RejectRemote | Edict::RejectStream if !remote => CommandStatus::Hidden,
            Edict::RejectRemote | Edict::RejectStream if self.busy => {
                CommandStatus::Disabled("no remote challenger is ready")
            }
            Edict::LockStream if !remote && self.remote_summary.lock.is_none() => {
                CommandStatus::Hidden
            }
            Edict::LockStream if self.busy || self.lock_pending.is_some() => {
                CommandStatus::Disabled("the stream lock is changing")
            }
            Edict::CopyViewer => CommandStatus::Hidden,
            Edict::ChooseLeft
            | Edict::ChooseRight
            | Edict::RejectRemote
            | Edict::RejectStream
            | Edict::LockStream
            | Edict::OpenCollection
            | Edict::Rescan
            | Edict::Compare
            | Edict::Browse => CommandStatus::Enabled,
        }
    }

    fn apply_edict(&mut self, ctx: &egui::Context, dispatch: CommandDispatch<'_, Edict>) {
        let edict = match dispatch {
            CommandDispatch::Invoke(edict) => edict,
            CommandDispatch::Refused { reason, .. } => {
                self.status = format!("UNAVAILABLE · {reason}");
                return;
            }
        };
        match edict {
            Edict::ChooseLeft => self.apply_action(Action::Choose(Side::Left)),
            Edict::ChooseRight => self.apply_action(Action::Choose(Side::Right)),
            Edict::RejectRemote => self.apply_action(Action::RejectRemote),
            Edict::RejectStream => self.apply_action(Action::RejectStream),
            Edict::LockStream => self.apply_action(Action::LockStream(!self.stream_locked())),
            Edict::CopyViewer => unreachable!("viewer commands are routed by the viewer"),
            Edict::OpenCollection => self.open_collection(ctx),
            Edict::Rescan => self.send(Command::Rescan, true, "SCANNING COLLECTION"),
            Edict::Compare => self.set_mode(Mode::Compare),
            Edict::Browse => self.set_mode(Mode::Browse),
        }
    }

    fn apply_action(&mut self, action: Action) {
        // An open modal owns the pointer, yet egui lets a click through to the
        // chamber during the modal's invisible first pass: no judgment may
        // come of a click aimed at the modal.
        if self.settings.is_open() || self.guide.is_open() {
            return;
        }
        match action {
            Action::Choose(side) => {
                let status = if self.remote_shown() {
                    "COMMITTING TO ARCHIVE"
                } else {
                    "FORGING NEXT PAIR"
                };
                if self.dispatch(Command::Choose(side), true) {
                    self.choice = Some(Choice {
                        sent: Instant::now(),
                        status,
                    });
                }
            }
            Action::Favorite(side) => {
                let remote = self.pair.as_ref().is_some_and(|pair| match side {
                    Side::Left => pair.pair.left.remote().is_some(),
                    Side::Right => pair.pair.right.remote().is_some(),
                });
                self.send(
                    Command::Favorite(side),
                    remote,
                    if remote {
                        "COMMITTING FAVORITE"
                    } else {
                        "MARKING FAVORITE"
                    },
                );
            }
            Action::Hide(side) => self.send(Command::Hide(side), true, "WITHDRAWING IMAGE"),
            Action::RejectRemote => {
                self.send(Command::RejectRemote, true, "REJECTING REMOTE IMAGE");
            }
            Action::RejectStream => {
                self.send(Command::RejectStream, true, "REJECTING REMOTE STREAM");
            }
            Action::LockStream(lock) => {
                if self.lock_pending.is_none()
                    && self.dispatch(
                        if lock {
                            Command::LockStream
                        } else {
                            Command::UnlockStream
                        },
                        false,
                    )
                {
                    self.lock_pending = Some(lock);
                    if lock {
                        "STREAM LOCKED"
                    } else {
                        "STREAM RELEASED"
                    }
                    .clone_into(&mut self.status);
                }
            }
            Action::Rotate(side, turn) => {
                self.send(Command::Rotate(side, turn), true, "TURNING IMAGE");
            }
        }
    }

    fn send(&mut self, command: Command, waits: bool, status: &'static str) {
        if self.dispatch(command, waits) {
            status.clone_into(&mut self.status);
        }
    }

    /// Sends `command` to the worker; one that waits holds the chamber until
    /// the worker answers.
    fn dispatch(&mut self, command: Command, waits: bool) -> bool {
        self.choice = None;
        match self.worker.send(command) {
            Ok(()) => {
                self.busy |= waits;
                true
            }
            Err(error) => {
                self.busy = false;
                self.status = format!("FAULT · {error:#}");
                false
            }
        }
    }

    /// Whether the chamber shows the worker's wait. A choice holds its pair
    /// for [`CHOICE_GRACE`] first, so a successor forged ahead replaces it
    /// without a flash of waiting.
    fn shows_wait(&mut self, ctx: &egui::Context) -> bool {
        if !self.busy {
            return false;
        }
        let Some(choice) = self.choice else {
            return true;
        };
        let remaining = CHOICE_GRACE.saturating_sub(choice.sent.elapsed());
        if !remaining.is_zero() && self.pair.is_some() {
            ctx.request_repaint_after(remaining);
            return false;
        }
        choice.status.clone_into(&mut self.status);
        self.choice = None;
        true
    }

    /// The lock state the chamber shows: a requested state until the worker
    /// confirms it.
    fn stream_locked(&self) -> bool {
        self.lock_pending
            .unwrap_or_else(|| self.remote_summary.lock.is_some())
    }

    fn remote_shown(&self) -> bool {
        self.pair
            .as_ref()
            .is_some_and(|pair| pair.pair.right.remote().is_some())
    }

    /// The source label and title of the stream the chamber is held to, else
    /// of the displayed remote challenger's.
    fn held_stream(&self) -> Option<(String, String)> {
        if let Some(lock) = &self.remote_summary.lock {
            return Some((source_label(&lock.source), lock.title.clone()));
        }
        self.pair
            .as_ref()
            .and_then(|pair| pair.pair.right.remote())
            .map(|remote| (remote.source.clone(), remote.stream.clone()))
    }

    fn open_collection(&mut self, ctx: &egui::Context) {
        if self.chooser.is_some() {
            return;
        }
        let (publish, chooser) = bounded(1);
        let wake = NativeWake::from_context(ctx);
        let spawned = thread::Builder::new()
            .name("picmash-collection-chooser".to_owned())
            .spawn(move || {
                let choice = pollster::block_on(rfd::AsyncFileDialog::new().pick_folder())
                    .map(|folder| folder.path().to_path_buf());
                let _sent = publish.send(choice);
                let _woken = wake.request_repaint();
            });
        match spawned {
            Ok(_thread) => {
                self.chooser = Some(chooser);
                self.busy = true;
                "CHOOSING COLLECTION".clone_into(&mut self.status);
            }
            Err(error) => {
                self.status = format!("FAULT · could not open collection chooser: {error}");
            }
        }
    }

    fn absorb_chooser(&mut self, _ctx: &egui::Context) {
        let Some(choice) = self
            .chooser
            .as_ref()
            .and_then(|chooser| chooser.try_recv().ok())
        else {
            return;
        };
        self.chooser = None;
        if let Some(path) = choice {
            self.send(Command::Load(path), true, "SCANNING COLLECTION");
        } else {
            self.busy = false;
            "COLLECTION SELECTION CANCELED".clone_into(&mut self.status);
        }
    }

    fn drain(&mut self, ctx: &egui::Context) {
        for _ in 0..EVENT_DRAIN {
            let Ok(event) = self.worker.events.try_recv() else {
                break;
            };
            self.absorb_event(ctx, event);
        }
        if !self.worker.events.is_empty() {
            ctx.request_repaint();
        }
    }

    fn absorb_event(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::NeedCollection => {
                self.busy = false;
                self.pending_collection = None;
                self.scan_progress = None;
                "NO COLLECTION CLAIMED".clone_into(&mut self.status);
            }
            Event::Busy(status) => {
                self.busy = true;
                self.scan_progress = None;
                self.status = status.to_owned();
            }
            Event::ScanStarted(root) => {
                self.busy = true;
                self.pending_collection = Some(root);
                self.scan_progress = None;
                "SCANNING COLLECTION".clone_into(&mut self.status);
            }
            Event::ScanProgress(progress) => {
                if self
                    .scan_progress
                    .is_none_or(|prior| progress.inspected_paths >= prior.inspected_paths)
                {
                    self.status = format!(
                        "SCANNING {}/{} · {} CACHED",
                        progress.inspected_paths, progress.total_paths, progress.reused_paths
                    );
                    self.scan_progress = Some(progress);
                }
            }
            Event::Catalog { summary, cards } => {
                self.pending_collection = None;
                self.scan_progress = None;
                self.viewer = None;
                self.water.close_pond();
                self.summary = Some(summary);
                self.cards = cards;
                self.pair = None;
                self.awaiting = false;
                self.thumbnails.clear();
                self.thumbnails_inflight.clear();
                self.thumbnails_faulted.clear();
                self.rebuild_browse_indices();
            }
            Event::CatalogRefreshed { summary, cards } => {
                self.summary = Some(summary);
                if let Some(view) = self.pair.as_mut() {
                    for card in [&mut view.pair.left, &mut view.pair.right] {
                        refresh_card(card, &cards);
                    }
                }
                self.cards = cards;
                self.thumbnails_faulted.clear();
                self.rebuild_browse_indices();
            }
            Event::Pair(pair) => {
                let remote = pair.right.remote().is_some();
                let left_texture = upload(ctx, "picmash-left", &pair.left_blade);
                let right_texture = upload(ctx, "picmash-right", &pair.right_blade);
                self.pair = Some(PairView {
                    pair: *pair,
                    left_texture,
                    right_texture,
                });
                self.busy = false;
                self.awaiting = false;
                self.choice = None;
                self.scan_progress = None;
                if !remote {
                    "CHOOSE THE STRONGER IMAGE"
                } else if self.remote_summary.lock.is_some() {
                    "LOCKED STREAM · CHOOSE THE STRONGER IMAGE"
                } else {
                    "REMOTE CHALLENGER · CHOOSE THE STRONGER IMAGE"
                }
                .clone_into(&mut self.status);
            }
            Event::Remote(summary) => {
                if self.lock_pending == Some(summary.lock.is_some()) {
                    self.lock_pending = None;
                }
                self.remote_summary = summary;
            }
            Event::RemoteFault(message) => {
                self.lock_pending = None;
                self.status = format!("REMOTE SUSPENDED · {message}");
            }
            Event::ArchiveFault(message) => {
                self.status = format!("ARCHIVE PAUSED · {message}");
            }
            Event::Awaiting => {
                self.pair = None;
                self.busy = false;
                self.awaiting = true;
                self.choice = None;
                self.scan_progress = None;
                "AWAITING THE LOCKED STREAM".clone_into(&mut self.status);
            }
            Event::Exhausted(title) => {
                self.status = format!("STREAM EXHAUSTED · LOCK RELEASED · {title}");
            }
            Event::NoComparison => {
                self.pair = None;
                self.busy = false;
                self.awaiting = false;
                self.scan_progress = None;
                "TWO VISIBLE IMAGES ARE REQUIRED".clone_into(&mut self.status);
            }
            Event::Favorite { asset_id, active } => {
                for card in &mut self.cards {
                    if card.presentation.asset_id == asset_id {
                        card.favorite = active;
                    }
                }
                if let Some(pair) = &mut self.pair {
                    for card in [&mut pair.pair.left, &mut pair.pair.right] {
                        if let Some(card) = card.local_mut()
                            && card.presentation.asset_id == asset_id
                        {
                            card.favorite = active;
                        }
                    }
                }
                self.rebuild_browse_indices();
                if active {
                    "FAVORITE MARKED"
                } else {
                    "FAVORITE WITHDRAWN"
                }
                .clone_into(&mut self.status);
            }
            Event::Thumbnail {
                asset_id,
                bucket,
                blade,
            } => {
                let key = ThumbKey { asset_id, bucket };
                let _inflight = self.thumbnails_inflight.remove(&key);
                let texture = upload(
                    ctx,
                    &format!("picmash-thumb-{}-{bucket}", key.asset_id),
                    &blade,
                );
                let _old = self.thumbnails.insert(key, texture);
            }
            Event::ThumbnailFault {
                asset_id,
                bucket,
                message,
            } => {
                let key = ThumbKey { asset_id, bucket };
                let _inflight = self.thumbnails_inflight.remove(&key);
                let _faulted = self.thumbnails_faulted.insert(key);
                self.status = format!("THUMBNAIL FAULT · {message}");
            }
            Event::Full {
                asset_id,
                source,
                display,
            } => {
                if let Some(viewer) = &mut self.viewer {
                    viewer.install(ctx, &asset_id, source, display.as_ref());
                }
            }
            Event::FullFault { asset_id, message } => {
                if let Some(viewer) = &mut self.viewer {
                    viewer.fail(&asset_id, message.clone());
                }
                self.status = format!("VIEWER FAULT · {message}");
            }
            Event::Fault(message) => {
                self.busy = false;
                self.lock_pending = None;
                self.pending_collection = None;
                self.scan_progress = None;
                self.status = format!("FAULT · {message}");
            }
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
            self.viewer = None;
            self.water.close_pond();
            self.mode = mode;
            match mode {
                Mode::Compare if self.pair.is_some() => "CHOOSE THE STRONGER IMAGE",
                Mode::Compare => "TWO VISIBLE IMAGES ARE REQUIRED",
                Mode::Browse => "BROWSING LEARNED ORDER",
            }
            .clone_into(&mut self.status);
            self.water.bump(self.water.domain());
        }
    }

    fn rebuild_browse_indices(&mut self) {
        let facets = Facets::tally(&self.cards, self.favorites_only);
        if !facets.holds(&self.scope) {
            self.scope = Scope::All;
        }
        self.pending_facets = Some(facets);
        self.browse_indices = self
            .cards
            .iter()
            .enumerate()
            .filter_map(|(index, card)| {
                ((!self.favorites_only || card.favorite) && self.scope.admits(card))
                    .then_some(index)
            })
            .collect();
    }

    /// Narrows the browser to `scope`, from its first row.
    fn narrow(&mut self, scope: Scope) {
        if scope.source() != self.scope.source() {
            self.stream_filter.clear();
        }
        self.scope = scope;
        self.browse_rewind = true;
        self.rebuild_browse_indices();
    }

    fn open_viewer(&mut self, ctx: &egui::Context, asset_id: &AssetId) {
        let sequence = self
            .browse_indices
            .iter()
            .map(|&index| self.cards[index].presentation.asset_id.clone())
            .collect::<Vec<_>>();
        let Some(slot) = sequence.iter().position(|candidate| candidate == asset_id) else {
            "VIEWER FAULT · IMAGE LEFT THE BROWSER".clone_into(&mut self.status);
            return;
        };
        self.viewer = Some(Viewer::open(sequence, slot));
        self.request_viewer_blade(ctx);
        ctx.request_repaint();
    }

    fn request_viewer_blade(&mut self, ctx: &egui::Context) {
        let demand = self.viewer.as_mut().and_then(|viewer| viewer.arm(ctx));
        let Some((asset_id, bound)) = demand else {
            return;
        };
        if let Err(error) = self.worker.send(Command::Full {
            asset_id: asset_id.clone(),
            bound,
        }) && let Some(viewer) = &mut self.viewer
        {
            viewer.fail(&asset_id, format!("{error:#}"));
        }
    }

    fn show_viewer(&mut self, ctx: &egui::Context) {
        self.request_viewer_blade(ctx);
        let Some(asset_id) = self.viewer.as_ref().map(|viewer| viewer.asset_id().clone()) else {
            return;
        };
        let Some(card) = self
            .cards
            .iter()
            .find(|card| card.presentation.asset_id == asset_id)
            .cloned()
        else {
            self.viewer = None;
            self.water.close_pond();
            "VIEWER FAULT · IMAGE LEFT THE COLLECTION".clone_into(&mut self.status);
            return;
        };
        let inputs_enabled =
            !self.guide.is_open() && !self.settings.is_open() && self.chooser.is_none();
        let actions = self.viewer.as_mut().map_or_else(Vec::new, |viewer| {
            viewer.show(ctx, &mut self.water, &card, inputs_enabled)
        });
        for action in actions {
            self.apply_viewer_action(ctx, action);
        }
    }

    fn apply_viewer_action(&mut self, ctx: &egui::Context, action: ViewerAction) {
        match action {
            ViewerAction::Close => {
                self.viewer = None;
                self.water.close_pond();
            }
            ViewerAction::Copy => {
                let result = self.viewer.as_mut().map(Viewer::begin_copy);
                match result {
                    Some(Ok(true)) => "COPYING IMAGE".clone_into(&mut self.status),
                    Some(Ok(false)) => {}
                    Some(Err(error)) => self.status = format!("COPY FAULT · {error:#}"),
                    None => {}
                }
            }
            ViewerAction::Favorite => {
                if let Some(asset_id) = self.viewer.as_ref().map(|viewer| viewer.asset_id().clone())
                {
                    self.send(Command::FavoriteAsset(asset_id), false, "MARKING FAVORITE");
                }
            }
            ViewerAction::Stream(source, stream) => {
                self.viewer = None;
                self.water.close_pond();
                self.narrow(Scope::Stream(source, stream));
            }
            ViewerAction::Previous | ViewerAction::Next => {
                if self
                    .viewer
                    .as_mut()
                    .is_some_and(|viewer| viewer.navigate(action))
                {
                    self.request_viewer_blade(ctx);
                }
            }
        }
    }

    fn poll_viewer_copy(&mut self, ctx: &egui::Context) {
        let result = self
            .viewer
            .as_mut()
            .and_then(|viewer| viewer.poll_copy(ctx));
        match result {
            Some(Ok(())) => "IMAGE COPIED".clone_into(&mut self.status),
            Some(Err(error)) => self.status = format!("COPY FAULT · {error:#}"),
            None => {}
        }
    }

    fn zoom_tiles(&mut self, ctx: &egui::Context) {
        if self.mode != Mode::Browse
            || self.viewer.is_some()
            || self.guide.is_open()
            || self.settings.is_open()
            || self.chooser.is_some()
            || ctx.text_edit_focused()
        {
            return;
        }
        let steps = ctx.input(|input| {
            input
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::MouseWheel {
                        unit,
                        delta,
                        modifiers,
                        ..
                    } if modifiers.ctrl => Some(match unit {
                        egui::MouseWheelUnit::Point => delta.y / 120.0,
                        egui::MouseWheelUnit::Line => delta.y,
                        egui::MouseWheelUnit::Page => delta.y * 4.0,
                    }),
                    _ => None,
                })
                .sum::<f32>()
        });
        if steps == 0.0 {
            return;
        }
        ctx.input_mut(|input| {
            input.events.retain(|event| {
                !matches!(event, egui::Event::MouseWheel { modifiers, .. } if modifiers.ctrl)
            });
            input.smooth_scroll_delta = egui::Vec2::ZERO;
        });
        let next = (i32::from(self.images_per_row) - steps.round() as i32)
            .clamp(i32::from(MIN_IMAGES_PER_ROW), i32::from(MAX_IMAGES_PER_ROW))
            as u16;
        if next == self.images_per_row {
            return;
        }
        self.images_per_row = next;
        if let Err(error) = self
            .configuration
            .revise(|config| config.images_per_row = next)
        {
            self.status = format!("FAULT · {error:#}");
        }
        ctx.request_repaint();
    }

    fn command_guide(&mut self, ctx: &egui::Context) {
        let context = self.mode.context();
        let guide_groups = match self.mode {
            Mode::Compare => &commands::COMPARE_GUIDE_GROUPS[..],
            Mode::Browse => &commands::BROWSE_GUIDE_GROUPS[..],
        };
        let mut guide = std::mem::take(&mut self.guide);
        guide.show(
            ctx,
            commands::canon(),
            &[context],
            |scope| match scope {
                commands::Context::Compare => "COMPARISON CHAMBER",
                commands::Context::Browse => "COLLECTION BROWSER",
                commands::Context::Viewer => "IMAGE VIEWER",
            },
            |edict| self.edict_status(edict),
            guide_groups,
        );
        self.guide = guide;
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut living_water = self.configuration.live().living_water;
        let mut font_scale = self.configuration.live().font_scale;
        let mut remote_enabled = self.configuration.live().remote.enabled;
        let mut remote_chance = self.configuration.live().remote.sample_probability.ratio();
        let mut remote_reservoir =
            f64::from(self.configuration.live().remote.reservoir_capacity.get());
        let fault = self.configuration.fault().map(ToString::to_string);
        let file = fault
            .as_deref()
            .map_or_else(
                || SettingsFile::ready(self.configuration.path()),
                |fault| SettingsFile::fault(self.configuration.path(), fault),
            )
            .reloading(self.configuration.reload_pending())
            .reloadable(self.configuration.fault().is_some() || self.configuration.settled());
        let response = self
            .settings
            .show(ctx, &mut self.water, Some(file), |settings| {
                settings.group("APPEARANCE");
                let _font_scale = settings.font_size(&mut font_scale);
                settings.group("PRESENTATION");
                let _water = settings.boolean(WATER, &mut living_water);
                settings.group("ACQUISITION");
                let _enabled = settings.boolean(REMOTE, &mut remote_enabled);
                let _chance =
                    settings.number(REMOTE_CHANCE, &mut remote_chance, 0.0..=1.0, 0.05, 2);
                let _reservoir =
                    settings.number(REMOTE_RESERVOIR, &mut remote_reservoir, 1.0..=8.0, 1.0, 0);
            });
        let remote_changed = remote_enabled != self.configuration.live().remote.enabled
            || remote_chance != self.configuration.live().remote.sample_probability.ratio()
            || remote_reservoir
                != f64::from(self.configuration.live().remote.reservoir_capacity.get());
        let font_scale_changed = font_scale != self.configuration.live().font_scale;
        if font_scale_changed
            || living_water != self.configuration.live().living_water
            || remote_changed
        {
            let remote = Probability::try_from(remote_chance).and_then(|probability| {
                Ok((
                    probability,
                    ReservoirCapacity::try_from(remote_reservoir.round() as u8)?,
                ))
            });
            match remote {
                Ok((probability, reservoir)) => {
                    match self.configuration.revise(|config| {
                        config.font_scale = font_scale;
                        config.living_water = living_water;
                        config.remote.enabled = remote_enabled;
                        config.remote.sample_probability = probability;
                        config.remote.reservoir_capacity = reservoir;
                    }) {
                        Ok(true) => {
                            if font_scale_changed {
                                chrome::set_font_scale(ctx, font_scale);
                            }
                            self.adopt_configuration();
                        }
                        Ok(false) => {}
                        Err(error) => self.status = format!("FAULT · {error:#}"),
                    }
                }
                Err(error) => self.status = format!("FAULT · {error:#}"),
            }
        }
        if response.reload_requested()
            && let Err(error) = self.configuration.request_reload()
        {
            self.status = format!("FAULT · {error:#}");
        }
    }

    fn adopt_configuration(&mut self) {
        self.water
            .set_wetness(if self.configuration.live().living_water {
                Wetness::Wet
            } else {
                Wetness::Dry
            });
        self.images_per_row = self
            .configuration
            .live()
            .images_per_row
            .clamp(MIN_IMAGES_PER_ROW, MAX_IMAGES_PER_ROW);
        if let Err(error) = self.worker.send(Command::ConfigureRemote(
            self.configuration.live().remote.clone(),
        )) {
            self.status = format!("REMOTE CONFIGURATION FAULT · {error:#}");
        }
    }

    #[cfg(feature = "egui-test")]
    pub fn observe(&self, text_edit_focused: bool) -> Observation {
        Observation {
            contract: picmash_contract::UI_FINGERPRINT,
            mode: self.mode.wire(),
            busy: self.busy,
            status: self.status.clone(),
            collection: self
                .summary
                .as_ref()
                .map(|summary| summary.root.display().to_string()),
            visible_assets: self.cards.len(),
            browse_visible: self.browse_indices.len(),
            favorites: favorite_total(&self.cards),
            duels: duel_total(&self.cards),
            pair_ready: self.pair.is_some(),
            remote_pair: self
                .pair
                .as_ref()
                .is_some_and(|pair| pair.pair.right.remote().is_some()),
            remote_ready: self.remote_summary.prepared + usize::from(self.remote_summary.offered),
            remote_promoting: self.remote_summary.promoting,
            stream_locked: self.remote_summary.lock.is_some(),
            awaiting: self.awaiting,
            pair_rotations: self.pair.as_ref().map(|pair| {
                [
                    pair.pair.left.rotation_quarters(),
                    pair.pair.right.rotation_quarters(),
                ]
            }),
            pair_files: self.pair.as_ref().map(|pair| {
                [
                    file_name(pair.pair.left.path()),
                    file_name(pair.pair.right.path()),
                ]
            }),
            images_per_row: self.images_per_row,
            tiles_wanting: self.tiles_wanting,
            thumbnails_faulted: !self.thumbnails_faulted.is_empty(),
            viewer_open: self.viewer.is_some(),
            viewer_ready: self.viewer.as_ref().is_some_and(Viewer::ready),
            guide_open: self.guide.is_open(),
            settings_open: self.settings.is_open(),
            text_edit_focused,
        }
    }
}

#[derive(Clone, Copy)]
struct ComparisonLayout {
    image: egui::Rect,
    frame: egui::Rect,
    panel: egui::Rect,
}

/// Lays the duel out on one baseline: both images take one height, the
/// largest at which both fit within their bronze rims, so their frames and
/// the rails beneath them align. Width left over is split between the slots,
/// each image centered in its own.
fn comparison_layouts(
    arena: egui::Rect,
    left: egui::Vec2,
    right: egui::Vec2,
    gap: f32,
) -> [ComparisonLayout; 2] {
    let gap = gap.clamp(0.0, arena.width());
    let width = (arena.width() - gap - 4.0 * BRONZE_RIM).max(2.0);
    let aspects = [positive_aspect(left), positive_aspect(right)];
    let height = (arena.height() - 2.0 * BRONZE_RIM)
        .max(1.0)
        .min(width / (aspects[0] + aspects[1]));
    let slack = (width - (aspects[0] + aspects[1]) * height) * 0.5;
    let mut left_edge = arena.left();
    aspects.map(|aspect| {
        let slot = aspect * height + 2.0 * BRONZE_RIM + slack;
        let image = egui::Rect::from_min_size(
            egui::pos2(
                left_edge + BRONZE_RIM + slack * 0.5,
                arena.top() + BRONZE_RIM,
            ),
            egui::vec2(aspect * height, height),
        );
        left_edge += slot + gap;
        let frame = image.expand(BRONZE_RIM);
        ComparisonLayout {
            image,
            frame,
            panel: egui::Rect::from_min_size(
                frame.left_bottom(),
                egui::vec2(frame.width(), DUEL_PANEL_HEIGHT),
            ),
        }
    })
}

fn comparison_image(
    ui: &egui::Ui,
    water: &mut Surface,
    side: Side,
    card: &PairCard,
    texture: &TextureHandle,
    layout: ComparisonLayout,
) -> Option<Action> {
    paint_bronze_frame(ui, layout.frame, chrome::PAGE);
    let response = ui.interact(
        layout.frame,
        ui.make_persistent_id(("comparison-choice", side.wire())),
        egui::Sense::click(),
    );
    ui.painter().image(
        texture.id(),
        layout.image,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    let response = response.on_hover_text(card.remote().map_or_else(
        || file_name(card.path()),
        |remote| format!("{}\n{}\n{}", remote.title, remote.source, remote.stream),
    ));
    witness::response(ui, Target::Choice(side), &response);
    if response.clicked() {
        water.select(layout.frame);
        Some(Action::Choose(side))
    } else {
        None
    }
}

fn comparison_panel(
    ui: &mut egui::Ui,
    water: &mut Surface,
    side: Side,
    card: &PairCard,
    locked: bool,
    rect: egui::Rect,
) -> Vec<Action> {
    let mut panel = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("comparison-panel", side.wire()))
            .max_rect(rect),
    );
    egui::Frame::new()
        .fill(chrome::RAISED)
        .stroke(egui::Stroke::new(1.0, chrome::EDGE))
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(&mut panel, |ui| {
            ui.set_min_width(ui.available_width());
            let (key, mut actions) = egui::containers::Sides::new()
                .height(DUEL_CONTROL_HEIGHT)
                .spacing(6.0)
                .shrink_left()
                .truncate()
                .show(
                    ui,
                    |ui| comparison_identity(ui, side, card),
                    |ui| comparison_actions(ui, water, side, card, locked),
                );
            if let Some(key) = key {
                water.monoglyph(&key);
                if key.clicked() {
                    actions.push(Action::Choose(side));
                }
            }
            actions
        })
        .inner
}

/// The rail's left side: the keycap that chooses this image, then what the
/// image is.
fn comparison_identity(
    ui: &mut egui::Ui,
    side: Side,
    card: &PairCard,
) -> Option<MonoglyphResponse> {
    let key = choice_key(side).map(|glyph| {
        let key = Monoglyph::new(glyph)
            .size(RAIL_GAUGE)
            .finish(MonoglyphFinish::Void)
            .show(ui)
            .on_hover_text("Choose this image");
        witness::response(ui, Target::ChoiceKey(side), &key);
        key
    });
    let _meta = ui.label(
        chrome::TypeRole::Label
            .text(comparison_metadata(card))
            .strong()
            .color(chrome::MUTED),
    );
    key
}

/// The bare key that chooses `side`, as its keycap is engraved.
fn choice_key(side: Side) -> Option<char> {
    let edict = match side {
        Side::Left => Edict::ChooseLeft,
        Side::Right => Edict::ChooseRight,
    };
    commands::canon()
        .shortcuts(edict)
        .iter()
        .find_map(|shortcut| match shortcut.key() {
            ShortcutKey::Character(glyph) if shortcut.modifiers() == ShortcutModifiers::NONE => {
                Some(glyph.to_ascii_uppercase())
            }
            _ => None,
        })
}

/// The rail's right side, laid out from the right edge: favorite, both
/// quarter turns, then hide, or a remote challenger's reject, stream lock,
/// and stream rejection.
fn comparison_actions(
    ui: &mut egui::Ui,
    water: &mut Surface,
    side: Side,
    card: &PairCard,
    locked: bool,
) -> Vec<Action> {
    let mut actions = Vec::new();
    let remote = card.remote().is_some();
    let favorite = card.favorite();
    let heart = Monoglyph::symbol(Symbol::Heart).finish(if favorite {
        MonoglyphFinish::Love
    } else {
        MonoglyphFinish::BrightCut
    });
    let hint = if remote {
        "Promote and mark favorite"
    } else if favorite {
        "Withdraw favorite"
    } else {
        "Mark favorite"
    };
    if rail_glyph(ui, water, heart, Target::Favorite(side), hint) {
        actions.push(Action::Favorite(side));
    }
    for (glyph, turn, hint) in [
        ('↷', Turn::Clockwise, "Rotate clockwise"),
        ('↶', Turn::Counterclockwise, "Rotate counterclockwise"),
    ] {
        if rail_glyph(
            ui,
            water,
            Monoglyph::new(glyph),
            Target::Rotate(side, turn),
            hint,
        ) {
            actions.push(Action::Rotate(side, turn));
        }
    }
    if !remote {
        let hide = Monoglyph::symbol(Symbol::Visibility);
        if rail_glyph(
            ui,
            water,
            hide,
            Target::Hide(side),
            "Hide from this collection",
        ) {
            actions.push(Action::Hide(side));
        }
        return actions;
    }
    let reject = Monoglyph::symbol(Symbol::Delete);
    if rail_glyph(
        ui,
        water,
        reject,
        Target::Reject(side),
        "Reject this remote image",
    ) {
        actions.push(Action::RejectRemote);
    }
    let mut latched = locked;
    let latch = Monoglyph::symbol(Symbol::Lock)
        .size(RAIL_GAUGE)
        .show_latched(ui, &mut latched)
        .on_hover_text(if locked {
            "Release this stream"
        } else {
            "Hold the comparison to this stream"
        });
    water.monoglyph(&latch);
    witness::response(ui, Target::Lock(side), &latch);
    if latch.clicked() {
        actions.push(Action::LockStream(latched));
    }
    let (stream, stream_activated) = command_plate(ui, Edict::RejectStream);
    let stream = stream.on_hover_text("Reject every image of this stream");
    witness::response(ui, Target::RejectStream(side), &stream);
    if stream_activated {
        actions.push(Action::RejectStream);
    }
    actions
}

/// One rail actuator; true when pressed.
fn rail_glyph(
    ui: &mut egui::Ui,
    water: &mut Surface,
    glyph: Monoglyph,
    target: Target,
    hint: &str,
) -> bool {
    let response = glyph.size(RAIL_GAUGE).show(ui).on_hover_text(hint);
    water.monoglyph(&response);
    witness::response(ui, target, &response);
    response.clicked()
}

fn comparison_metadata(card: &PairCard) -> String {
    card.remote().map_or_else(
        || {
            format!(
                "{}×{} · {} DUELS",
                card.width(),
                card.height(),
                card.duel_count().unwrap_or(0)
            )
        },
        |remote| {
            format!(
                "{} · {} · {}×{}",
                remote.source,
                remote.stream,
                card.width(),
                card.height()
            )
        },
    )
}

fn paint_bronze_frame(ui: &egui::Ui, rect: egui::Rect, fill: egui::Color32) -> egui::Rect {
    let inner = rect.shrink(BRONZE_RIM);
    let _bronze = ui.painter().rect_filled(rect, 1.0, chrome::RAISED);
    let _outer = ui.painter().rect_stroke(
        rect,
        1.0,
        egui::Stroke::new(1.0, chrome::EDGE_STRONG),
        egui::StrokeKind::Inside,
    );
    let _well = ui.painter().rect_filled(inner, 0.0, fill);
    let _inner = ui.painter().rect_stroke(
        inner,
        0.0,
        egui::Stroke::new(1.0, chrome::EDGE),
        egui::StrokeKind::Inside,
    );
    inner
}

/// A rail plate for `edict`: its label, mnemonic underlined, and any shortcut
/// legend.
fn command_plate(ui: &mut egui::Ui, edict: Edict) -> (egui::Response, bool) {
    let command = commands::canon().button_with(edict, ui, |button| {
        button.min_size(egui::vec2(24.0, DUEL_CONTROL_HEIGHT))
    });
    let activated = command.clicked();
    let response = command.into_response();
    chrome::tension(ui, &response);
    (response, activated)
}

fn positive_aspect(image: egui::Vec2) -> f32 {
    if image.x > 0.0 && image.y > 0.0 {
        image.x / image.y
    } else {
        1.0
    }
}

fn browse_tile(
    ui: &mut egui::Ui,
    water: &mut Surface,
    card: &Card,
    texture: Option<&TextureHandle>,
    edge: f32,
    witnessed: bool,
) -> bool {
    let (rect, response) = ui.allocate_exact_size(egui::Vec2::splat(edge), egui::Sense::click());
    if witnessed {
        witness::response(ui, Target::BrowseTile, &response);
    }
    if let Some(texture) = texture {
        ui.painter().image(
            texture.id(),
            rect,
            cover_uv(rect.size(), texture.size_vec2()),
            egui::Color32::WHITE,
        );
    } else {
        ui.painter().rect_filled(rect, 0.0, chrome::SURFACE);
        let _waiting = ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "DEVELOPING",
            chrome::spatial_font(ui.ctx(), 12.0, egui::FontFamily::Proportional),
            chrome::MUTED,
        );
    }
    if card.favorite {
        paint_tile_badge(ui, rect, "♥".to_owned(), chrome::HOT, BadgeCorner::Left);
    }
    if card.occurrence_count > 1 {
        paint_tile_badge(
            ui,
            rect,
            format!("◇ {}", card.occurrence_count),
            chrome::TEXT,
            BadgeCorner::Right,
        );
    }
    if response.hovered() {
        paint_browse_metadata(ui, rect, card);
        water.hover(("browse", card.presentation.asset_id.as_str()), rect);
    }
    if response.clicked() {
        water.click(rect);
        true
    } else {
        false
    }
}

#[derive(Clone, Copy)]
enum BadgeCorner {
    Left,
    Right,
}

fn paint_tile_badge(
    ui: &egui::Ui,
    tile: egui::Rect,
    text: String,
    color: egui::Color32,
    corner: BadgeCorner,
) {
    let font = chrome::spatial_font(ui.ctx(), 13.0, egui::FontFamily::Monospace);
    let galley = ui.painter().layout_no_wrap(text, font, color);
    let size = galley.size() + egui::vec2(12.0, 6.0);
    let (minimum, radius) = match corner {
        BadgeCorner::Left => (tile.left_top(), egui::CornerRadius::ZERO),
        BadgeCorner::Right => (
            egui::pos2(tile.right() - size.x, tile.top()),
            egui::CornerRadius::ZERO,
        ),
    };
    let rect = egui::Rect::from_min_size(minimum, size);
    ui.painter().rect_filled(rect, radius, chrome::RAISED);
    ui.painter().rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, chrome::EDGE_STRONG),
        egui::StrokeKind::Inside,
    );
    ui.painter()
        .galley(rect.center() - galley.size() * 0.5, galley, color);
}

fn paint_browse_metadata(ui: &egui::Ui, tile: egui::Rect, card: &Card) {
    let rect = egui::Rect::from_min_max(egui::pos2(tile.min.x, tile.max.y - 48.0), tile.max);
    let painter = ui.painter_at(tile);
    painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(184));
    let score = card.preference_score.map_or_else(
        || "UNRANKED".to_owned(),
        |score| format!("PREF {score:+.2}"),
    );
    let copies = if card.occurrence_count > 1 {
        format!(" · {} FILES", card.occurrence_count)
    } else {
        String::new()
    };
    painter.text(
        rect.left_top() + egui::vec2(8.0, 7.0),
        egui::Align2::LEFT_TOP,
        format!("{score} · {} DUELS{copies}", card.duel_count),
        chrome::spatial_font(ui.ctx(), 11.0, egui::FontFamily::Monospace),
        chrome::TEXT,
    );
    painter.text(
        rect.left_bottom() + egui::vec2(8.0, -7.0),
        egui::Align2::LEFT_BOTTOM,
        file_name(&card.path),
        chrome::spatial_font(ui.ctx(), 11.0, egui::FontFamily::Monospace),
        chrome::MUTED,
    );
}

fn tile_edge(width: f32, columns: usize) -> f32 {
    let columns = columns.max(1);
    let gaps = TILE_GAP * columns.saturating_sub(1) as f32;
    ((width - gaps) / columns as f32).max(MIN_TILE_EDGE)
}

fn thumb_bucket(edge: f32) -> u8 {
    if edge > 390.0 {
        2
    } else {
        u8::from(edge > 190.0)
    }
}

#[derive(Clone, Copy)]
enum GalleryMotion {
    PreviousRow,
    NextRow,
    First,
}

fn gallery_motion(input: &egui::InputState) -> Option<GalleryMotion> {
    [
        (egui::Key::PageUp, GalleryMotion::PreviousRow),
        (egui::Key::PageDown, GalleryMotion::NextRow),
        (egui::Key::Home, GalleryMotion::First),
    ]
    .into_iter()
    .find_map(|(key, motion)| exact_key_pressed(input, key).then_some(motion))
}

fn exact_key_pressed(input: &egui::InputState, key: egui::Key) -> bool {
    input.events.iter().any(|event| {
        matches!(
            event,
            egui::Event::Key {
                key: candidate,
                pressed: true,
                modifiers,
                ..
            } if *candidate == key && modifiers.matches_exact(egui::Modifiers::NONE)
        )
    })
}

fn cover_uv(arena: egui::Vec2, image: egui::Vec2) -> egui::Rect {
    let arena = arena.max(egui::Vec2::splat(1.0));
    let image = image.max(egui::Vec2::splat(1.0));
    let arena_aspect = arena.x / arena.y;
    let image_aspect = image.x / image.y;
    if image_aspect > arena_aspect {
        let visible = arena_aspect / image_aspect;
        let inset = (1.0 - visible) * 0.5;
        egui::Rect::from_min_max(egui::pos2(inset, 0.0), egui::pos2(1.0 - inset, 1.0))
    } else {
        let visible = image_aspect / arena_aspect;
        let inset = (1.0 - visible) * 0.5;
        egui::Rect::from_min_max(egui::pos2(0.0, inset), egui::pos2(1.0, 1.0 - inset))
    }
}

fn plate(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let button = egui::Button::new(chrome::section_title(label))
        .selected(selected)
        .min_size(egui::vec2(ui.available_width(), 28.0));
    let response = ui.add(button);
    chrome::tension(ui, &response);
    response
}

fn plate_enabled(ui: &mut egui::Ui, enabled: bool, label: &str, selected: bool) -> egui::Response {
    let button = egui::Button::new(chrome::section_title(label))
        .selected(selected)
        .min_size(egui::vec2(ui.available_width(), 28.0));
    let response = ui.add_enabled(enabled, button);
    chrome::tension(ui, &response);
    response
}

/// Attaches a glossary card to a domain term: one short title and one direct
/// definition, offered under the platform Help cursor.
fn glossary(term: egui::Response, title: &str, definition: &str) -> egui::Response {
    term.on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_ui(|ui| {
            ui.set_max_width(280.0);
            let _title = ui.label(chrome::section_title(title));
            let _definition = ui.label(chrome::muted(definition));
        })
}

fn datum(ui: &mut egui::Ui, label: &str, value: String) {
    let _row = ui.horizontal(|ui| {
        let _label = ui.label(chrome::muted(label));
        let _value = ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(chrome::section_title(value))
        });
    });
}

fn duel_total(cards: &[Card]) -> u64 {
    cards
        .iter()
        .map(|card| u64::from(card.duel_count))
        .sum::<u64>()
        / 2
}

fn favorite_total(cards: &[Card]) -> usize {
    cards.iter().filter(|card| card.favorite).count()
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Brings a shown local card's statistics up to date with a refreshed
/// catalog, provided the catalog still names the same rendering.
fn refresh_card(card: &mut PairCard, cards: &[Card]) {
    if let PairCard::Local(card) = card
        && let Some(fresh) = cards
            .iter()
            .find(|fresh| fresh.presentation == card.presentation && fresh.path == card.path)
    {
        card.clone_from(fresh);
    }
}

fn upload(ctx: &egui::Context, name: &str, blade: &Blade) -> TextureHandle {
    let image = ColorImage::from_rgba_unmultiplied([blade.width, blade.height], &blade.rgba);
    ctx.load_texture(name, image, TextureOptions::LINEAR)
}

#[cfg(feature = "egui-test")]
#[derive(serde::Serialize)]
pub struct Observation {
    contract: &'static str,
    mode: &'static str,
    busy: bool,
    status: String,
    collection: Option<String>,
    visible_assets: usize,
    browse_visible: usize,
    favorites: usize,
    duels: u64,
    pair_ready: bool,
    remote_pair: bool,
    remote_ready: usize,
    remote_promoting: usize,
    stream_locked: bool,
    awaiting: bool,
    pair_rotations: Option<[u8; 2]>,
    pair_files: Option<[String; 2]>,
    images_per_row: u16,
    tiles_wanting: Option<usize>,
    thumbnails_faulted: bool,
    viewer_open: bool,
    viewer_ready: bool,
    guide_open: bool,
    settings_open: bool,
    text_edit_focused: bool,
}
