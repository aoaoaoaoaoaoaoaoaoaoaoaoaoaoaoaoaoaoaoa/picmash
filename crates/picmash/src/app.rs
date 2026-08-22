use anyhow::Result;
use brass_poolrooms::{
    chrome::{self, Checkbox, MechanismSize, Monoglyph, MonoglyphFinish, Symbol},
    water::{Floor, Surface, Wetness},
};
use crossbeam_channel::{Receiver, bounded};
use egui::{ColorImage, TextureHandle, TextureOptions};
use eternalist_apps::{
    ApplicationHeader, Inspector, LivingWait, NativeWake,
    command_guide::CommandGuide,
    commands::{CommandDispatch, CommandStatus},
    configuration::ConfigurationLedger,
    panel_navigation::PanelNavigator,
    settings::{SettingSpec, SettingsFile, SettingsSheet},
};
use picmash_contract::{Side, Target};
use picmash_engine::AssetId;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use crate::{
    commands::{self, Edict},
    configuration::Config,
    witness,
    worker::{Blade, Card, Command, Event, Pair, Summary, Worker},
    xdg::Lair,
};

const EVENT_DRAIN: usize = 24;
const CONFIG_SETTLE: Duration = Duration::from_millis(400);
const TILE_MIN: f32 = 170.0;
const TILE_GAP: f32 = 10.0;
const TILE_CHROME: f32 = 54.0;
const WATER: SettingSpec = SettingSpec::new(
    "living_water",
    "LIVING WATER",
    "Let controls and image choices displace the chamber's one water body.",
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

#[derive(Clone, Copy)]
enum Action {
    Choose(Side),
    Favorite(Side),
    Hide(Side),
    Rotate(Side),
}

pub struct Picmash {
    worker: Worker,
    chooser: Option<Receiver<Option<PathBuf>>>,
    mode: Mode,
    summary: Option<Summary>,
    cards: Vec<Card>,
    browse_indices: Vec<usize>,
    favorites_only: bool,
    pair: Option<PairView>,
    thumbnails: HashMap<AssetId, TextureHandle>,
    thumbnails_inflight: HashSet<AssetId>,
    busy: bool,
    status: String,
    panels: PanelNavigator,
    guide: CommandGuide,
    settings: SettingsSheet,
    configuration: ConfigurationLedger<Config>,
    living_wait: LivingWait,
    water: Surface,
}

impl Picmash {
    pub fn open(ctx: &egui::Context, initial: Option<PathBuf>) -> Result<Self> {
        let lair = Lair::claim()?;
        let configuration: ConfigurationLedger<Config> = ConfigurationLedger::raise(
            "picmash-configuration",
            ctx,
            lair.configuration(),
            CONFIG_SETTLE,
        )?;
        let wetness = if configuration.live().living_water {
            Wetness::Wet
        } else {
            Wetness::Dry
        };
        let worker = Worker::spawn(ctx, lair, initial)?;
        Ok(Self {
            worker,
            chooser: None,
            mode: Mode::Compare,
            summary: None,
            cards: Vec::new(),
            browse_indices: Vec::new(),
            favorites_only: false,
            pair: None,
            thumbnails: HashMap::new(),
            thumbnails_inflight: HashSet::new(),
            busy: true,
            status: "WAKING ENGINE".to_owned(),
            panels: PanelNavigator::default(),
            guide: CommandGuide::default(),
            settings: SettingsSheet::default(),
            configuration,
            living_wait: LivingWait::default(),
            water: Surface::new(wetness),
        })
    }

    pub fn pulse(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.absorb_chooser(&ctx);
        self.drain(&ctx);
        if self.configuration.absorb() {
            self.adopt_configuration();
        }
        if self.configuration.fault().is_some() {
            self.settings.require_attention(&ctx);
        }
        let settings_invoked = !self.guide.is_open() && self.settings.take_shortcut(&ctx);
        let guide_invoked =
            !settings_invoked && !self.settings.is_open() && self.guide.take_shortcuts(&ctx);
        if !settings_invoked
            && !guide_invoked
            && !self.settings.is_open()
            && !self.guide.is_open()
            && self.chooser.is_none()
            && let Some(dispatch) = commands::canon().route(&ctx, &[self.mode.context()], |edict| {
                self.edict_status(edict)
            })
        {
            self.apply_edict(&ctx, dispatch);
        }
        self.paint(ui);
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

        let mode = panels.section(ui, "chamber", "CHAMBER", true, |ui| {
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

        let collection = panels.section(ui, "collection", "COLLECTION", true, |ui| {
            if let Some(summary) = &self.summary {
                let _root = ui.label(
                    chrome::section_title(file_name(&summary.root).to_uppercase()).size(12.0),
                );
                let _path = ui.label(chrome::muted(summary.root.display().to_string()).size(11.0));
                ui.add_space(5.0);
            } else {
                let _none = ui.label(chrome::muted("NO COLLECTION CLAIMED"));
                ui.add_space(5.0);
            }
            let open = plate_enabled(ui, !self.busy, "OPEN COLLECTION", false);
            witness::response(ui, Target::OpenCollection, &open);
            if open.clicked() {
                self.open_collection(ui.ctx());
            }
            let rescan = plate_enabled(ui, self.summary.is_some() && !self.busy, "RESCAN", false);
            witness::response(ui, Target::Rescan, &rescan);
            if rescan.clicked() {
                self.send(Command::Rescan, true, "SCANNING COLLECTION");
            }
        });
        self.water.fold(collection.wake);

        if self.mode == Mode::Browse {
            let view = panels.section(ui, "view", "VIEW", true, |ui| {
                let before = self.favorites_only;
                let filter = Checkbox::new(&mut self.favorites_only, "FAVORITES ONLY")
                    .size(MechanismSize::Small)
                    .show(ui);
                self.water.checkbox(&filter);
                if before != self.favorites_only {
                    self.rebuild_browse_indices();
                }
            });
            self.water.fold(view.wake);
        }

        let status = panels.section(ui, "status", "STATUS", true, |ui| {
            if let Some(summary) = &self.summary {
                datum(ui, "VISIBLE", summary.visible_assets.to_string());
                datum(ui, "DUELS", duel_total(&self.cards).to_string());
                datum(ui, "FAVORITES", favorite_total(&self.cards).to_string());
                if summary.scan_failures > 0 {
                    datum(ui, "UNREADABLE", summary.scan_failures.to_string());
                }
                if let Some(evaluation) = summary.evaluation {
                    ui.add_space(4.0);
                    let _label = ui.label(chrome::eyebrow("CHRONOLOGICAL HOLDOUT"));
                    datum(
                        ui,
                        "ACCURACY",
                        format!("{:.0}%", evaluation.accuracy * 100.0),
                    );
                    datum(ui, "LOG LOSS", format!("{:.3}", evaluation.log_loss));
                    let _caution = ui.label(chrome::muted("DIAGNOSTIC, NOT QUALITY"));
                }
            }
            ui.add_space(4.0);
            let _status = ui.label(chrome::muted(&self.status));
        });
        self.water.fold(status.wake);
    }

    fn chamber(&mut self, ui: &mut egui::Ui) {
        let arena = ui.available_rect_before_wrap();
        self.water
            .set_floor(self.busy.then_some(Floor::shallow(arena)));
        match (self.busy, self.summary.is_some(), self.mode) {
            (true, _, _) => {
                let _rect = self.living_wait.bouncer(ui, arena);
            }
            (false, false, _) => self.first_contact(ui),
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
        let _heading = ui.horizontal(|ui| {
            let _title = ui.label(chrome::eyebrow("PAIRWISE PREFERENCE"));
            let _hint = ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(chrome::muted("CLICK · A LEFT · D RIGHT"))
            });
        });
        ui.add_space(8.0);
        let height = (ui.available_height() - 4.0).max(220.0);
        let mut actions = Vec::new();
        ui.columns(2, |columns| {
            if let Some(action) = comparison_card(
                &mut columns[0],
                &mut self.water,
                Side::Left,
                &left.0,
                &left.1,
                height,
            ) {
                actions.push(action);
            }
            if let Some(action) = comparison_card(
                &mut columns[1],
                &mut self.water,
                Side::Right,
                &right.0,
                &right.1,
                height,
            ) {
                actions.push(action);
            }
        });
        for action in actions {
            self.apply_action(action);
        }
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        let _heading = ui.horizontal(|ui| {
            let _title = ui.label(chrome::eyebrow("COLLECTION BROWSER"));
            let _count = ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(chrome::muted(format!(
                    "{} SHOWN",
                    self.browse_indices.len()
                )))
            });
        });
        ui.add_space(8.0);
        let width = ui.available_width();
        let columns = ((width + TILE_GAP) / (TILE_MIN + TILE_GAP))
            .floor()
            .max(1.0) as usize;
        let edge =
            ((width - TILE_GAP * columns.saturating_sub(1) as f32) / columns as f32).max(96.0);
        let rows = self.browse_indices.len().div_ceil(columns);
        let mut demands = Vec::new();
        let mut retained = HashSet::new();
        let body = egui::ScrollArea::vertical()
            .id_salt("picmash-browser")
            .show_rows(ui, edge + TILE_CHROME + TILE_GAP, rows, |ui, range| {
                for row in range {
                    let _row = ui.horizontal(|ui| {
                        for column in 0..columns {
                            let slot = row * columns + column;
                            let Some(&card_index) = self.browse_indices.get(slot) else {
                                ui.allocate_space(egui::vec2(edge, 1.0));
                                continue;
                            };
                            let card = &self.cards[card_index];
                            let _retained = retained.insert(card.asset_id.clone());
                            let texture = self.thumbnails.get(&card.asset_id);
                            browse_tile(ui, card, texture, edge);
                            if texture.is_none()
                                && !self.thumbnails_inflight.contains(&card.asset_id)
                            {
                                demands.push(card.asset_id.clone());
                            }
                        }
                    });
                    ui.add_space(TILE_GAP);
                }
            });
        witness::rect(ui.ctx(), Target::Browser, body.inner_rect);
        self.thumbnails
            .retain(|asset_id, _texture| retained.contains(asset_id));
        for asset_id in demands {
            if self
                .worker
                .send(Command::Thumbnail(asset_id.clone()))
                .is_ok()
            {
                let _inserted = self.thumbnails_inflight.insert(asset_id);
            }
        }
    }

    fn edict_status(&self, edict: Edict) -> CommandStatus<'static> {
        match edict {
            Edict::ChooseLeft | Edict::ChooseRight if self.busy || self.pair.is_none() => {
                CommandStatus::Disabled("no comparison is ready")
            }
            Edict::Rescan if self.busy || self.summary.is_none() => {
                CommandStatus::Disabled("no collection is ready")
            }
            Edict::OpenCollection if self.busy => CommandStatus::Disabled("Picmash is busy"),
            Edict::ChooseLeft
            | Edict::ChooseRight
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
            Edict::OpenCollection => self.open_collection(ctx),
            Edict::Rescan => self.send(Command::Rescan, true, "SCANNING COLLECTION"),
            Edict::Compare => self.set_mode(Mode::Compare),
            Edict::Browse => self.set_mode(Mode::Browse),
        }
    }

    fn apply_action(&mut self, action: Action) {
        match action {
            Action::Choose(side) => self.send(Command::Choose(side), true, "FORGING NEXT PAIR"),
            Action::Favorite(side) => {
                self.send(Command::Favorite(side), false, "MARKING FAVORITE");
            }
            Action::Hide(side) => self.send(Command::Hide(side), true, "WITHDRAWING IMAGE"),
            Action::Rotate(side) => self.send(Command::Rotate(side), true, "TURNING IMAGE"),
        }
    }

    fn send(&mut self, command: Command, waits: bool, status: &'static str) {
        match self.worker.send(command) {
            Ok(()) => {
                self.busy |= waits;
                status.clone_into(&mut self.status);
            }
            Err(error) => {
                self.busy = false;
                self.status = format!("FAULT · {error:#}");
            }
        }
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
                "NO COLLECTION CLAIMED".clone_into(&mut self.status);
            }
            Event::Busy(status) => {
                self.busy = true;
                self.status = status.to_owned();
            }
            Event::Catalog { summary, cards } => {
                self.summary = Some(summary);
                self.cards = cards;
                self.pair = None;
                self.thumbnails.clear();
                self.thumbnails_inflight.clear();
                self.rebuild_browse_indices();
            }
            Event::Pair(pair) => {
                let left_texture = upload(ctx, "picmash-left", &pair.left_blade);
                let right_texture = upload(ctx, "picmash-right", &pair.right_blade);
                self.pair = Some(PairView {
                    pair,
                    left_texture,
                    right_texture,
                });
                self.busy = false;
                "CHOOSE THE STRONGER IMAGE".clone_into(&mut self.status);
            }
            Event::NoComparison => {
                self.pair = None;
                self.busy = false;
                "TWO VISIBLE IMAGES ARE REQUIRED".clone_into(&mut self.status);
            }
            Event::Favorite { asset_id, active } => {
                for card in &mut self.cards {
                    if card.asset_id == asset_id {
                        card.favorite = active;
                    }
                }
                if let Some(pair) = &mut self.pair {
                    for card in [&mut pair.pair.left, &mut pair.pair.right] {
                        if card.asset_id == asset_id {
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
            Event::Thumbnail { asset_id, blade } => {
                let _inflight = self.thumbnails_inflight.remove(&asset_id);
                let texture = upload(ctx, &format!("picmash-thumb-{asset_id}"), &blade);
                let _old = self.thumbnails.insert(asset_id, texture);
            }
            Event::Fault(message) => {
                self.busy = false;
                self.status = format!("FAULT · {message}");
            }
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
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
        self.browse_indices = self
            .cards
            .iter()
            .enumerate()
            .filter_map(|(index, card)| (!self.favorites_only || card.favorite).then_some(index))
            .collect();
    }

    fn command_guide(&mut self, ctx: &egui::Context) {
        let context = self.mode.context();
        let idioms = match self.mode {
            Mode::Compare => &commands::COMPARE_GUIDE[..],
            Mode::Browse => &commands::BROWSE_GUIDE[..],
        };
        let mut guide = std::mem::take(&mut self.guide);
        guide.show(
            ctx,
            commands::canon(),
            &[context],
            |scope| match scope {
                commands::Context::Compare => "COMPARISON CHAMBER",
                commands::Context::Browse => "COLLECTION BROWSER",
            },
            |edict| self.edict_status(edict),
            idioms,
        );
        self.guide = guide;
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut living_water = self.configuration.live().living_water;
        let fault = self.configuration.fault().map(ToString::to_string);
        let file = fault.as_deref().map_or_else(
            || SettingsFile::ready(self.configuration.path()),
            |fault| SettingsFile::fault(self.configuration.path(), fault),
        );
        let response = self.settings.show(ctx, &mut self.water, file, |settings| {
            settings.section("PRESENTATION");
            let _water = settings.boolean(WATER, &mut living_water);
        });
        if living_water != self.configuration.live().living_water
            && let Err(error) = self
                .configuration
                .revise(|config| config.living_water = living_water)
        {
            self.status = format!("FAULT · {error:#}");
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
            favorites: favorite_total(&self.cards),
            duels: duel_total(&self.cards),
            pair_ready: self.pair.is_some(),
            pair_rotations: self.pair.as_ref().map(|pair| {
                [
                    pair.pair.left.rotation_quarters,
                    pair.pair.right.rotation_quarters,
                ]
            }),
            guide_open: self.guide.is_open(),
            settings_open: self.settings.is_open(),
            text_edit_focused,
        }
    }
}

fn comparison_card(
    ui: &mut egui::Ui,
    water: &mut Surface,
    side: Side,
    card: &Card,
    texture: &TextureHandle,
    height: f32,
) -> Option<Action> {
    let mut action = None;
    let _card = egui::Frame::new()
        .fill(chrome::SURFACE)
        .stroke(egui::Stroke::new(1.0_f32, chrome::EDGE_STRONG))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.set_min_height(height - 16.0);
            let _title = ui.horizontal(|ui| {
                let label = match side {
                    Side::Left => "A  LEFT",
                    Side::Right => "D  RIGHT",
                };
                let _label = ui.label(chrome::section_title(label));
                let _meta =
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(chrome::muted(format!(
                            "{}×{} · {} DUELS",
                            card.width, card.height, card.duel_count
                        )))
                    });
            });
            ui.add_space(6.0);
            let image_height = (height - 104.0).max(140.0);
            let arena = egui::vec2(ui.available_width(), image_height);
            let (rect, response) = ui.allocate_exact_size(arena, egui::Sense::click());
            let image_rect = contain(rect, texture.size_vec2());
            ui.painter().rect_filled(rect, 1.0, chrome::CONTROL);
            ui.painter().image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            chrome::tension(ui, &response);
            let response = response.on_hover_text("Choose this image");
            witness::response(ui, Target::Choice(side), &response);
            if response.clicked() {
                water.select(image_rect);
                action = Some(Action::Choose(side));
            } else if response.hovered() {
                water.hover(("comparison", side.wire()), image_rect);
            }
            ui.add_space(6.0);
            let _bar = ui.horizontal(|ui| {
                let favorite = Monoglyph::symbol(Symbol::Heart)
                    .finish(if card.favorite {
                        MonoglyphFinish::Love
                    } else {
                        MonoglyphFinish::BrightCut
                    })
                    .size(MechanismSize::Medium)
                    .show(ui)
                    .on_hover_text(if card.favorite {
                        "Withdraw favorite"
                    } else {
                        "Mark favorite"
                    });
                water.monoglyph(&favorite);
                witness::response(ui, Target::Favorite(side), &favorite);
                if favorite.clicked() {
                    action = Some(Action::Favorite(side));
                }
                let rotate = Monoglyph::new('↻')
                    .size(MechanismSize::Medium)
                    .show(ui)
                    .on_hover_text("Rotate clockwise");
                water.monoglyph(&rotate);
                witness::response(ui, Target::Rotate(side), &rotate);
                if rotate.clicked() {
                    action = Some(Action::Rotate(side));
                }
                let hide = Monoglyph::symbol(Symbol::Visibility)
                    .finish(MonoglyphFinish::BrightCut)
                    .size(MechanismSize::Medium)
                    .show(ui)
                    .on_hover_text("Hide from this collection");
                water.monoglyph(&hide);
                witness::response(ui, Target::Hide(side), &hide);
                if hide.clicked() {
                    action = Some(Action::Hide(side));
                }
                let _path = ui
                    .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(chrome::muted(file_name(&card.path)))
                    });
            });
        });
    action
}

fn browse_tile(ui: &mut egui::Ui, card: &Card, texture: Option<&TextureHandle>, edge: f32) {
    let _slot = ui.allocate_ui_with_layout(
        egui::vec2(edge, edge + TILE_CHROME),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            let _card = egui::Frame::new()
                .fill(chrome::SURFACE)
                .stroke(egui::Stroke::new(1.0_f32, chrome::EDGE))
                .inner_margin(egui::Margin::same(5))
                .show(ui, |ui| {
                    ui.set_width(edge - 10.0);
                    let (rect, _response) = ui.allocate_exact_size(
                        egui::vec2(edge - 10.0, edge - 10.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(rect, 1.0, chrome::CONTROL);
                    if let Some(texture) = texture {
                        ui.painter().image(
                            texture.id(),
                            contain(rect, texture.size_vec2()),
                            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    } else {
                        let _waiting = ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "DEVELOPING",
                            egui::FontId::proportional(12.0),
                            chrome::MUTED,
                        );
                    }
                    let favorite = if card.favorite { "♥  " } else { "" };
                    let score = card.preference_score.map_or_else(
                        || "UNRANKED".to_owned(),
                        |score| format!("PREF {score:+.2}"),
                    );
                    let _meta = ui.label(chrome::muted(format!(
                        "{favorite}{score} · {} DUELS",
                        card.duel_count
                    )));
                    let _name = ui.label(chrome::muted(file_name(&card.path)).size(11.0));
                });
        },
    );
}

fn contain(arena: egui::Rect, image: egui::Vec2) -> egui::Rect {
    let image = image.max(egui::Vec2::splat(1.0));
    let scale = (arena.width() / image.x).min(arena.height() / image.y);
    egui::Rect::from_center_size(arena.center(), image * scale)
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
    favorites: usize,
    duels: u64,
    pair_ready: bool,
    pair_rotations: Option<[u8; 2]>,
    guide_open: bool,
    settings_open: bool,
    text_edit_focused: bool,
}
