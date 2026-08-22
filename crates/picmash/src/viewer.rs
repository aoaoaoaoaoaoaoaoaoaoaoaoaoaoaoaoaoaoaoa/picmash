use anyhow::{Context as _, Result, anyhow};
use arboard::{Clipboard, ImageData};
use brass_poolrooms::{
    chrome::{self, MechanismSize, Monoglyph, MonoglyphFinish, Symbol},
    water::Surface,
};
use crossbeam_channel::{Receiver, TryRecvError, bounded};
use egui::{ColorImage, TextureHandle, TextureOptions};
use picmash_contract::Target;
use picmash_engine::AssetId;
use std::{borrow::Cow, thread};

use crate::{
    witness,
    worker::{Blade, Card},
};

const VIEWER_ID: &str = "picmash-viewer";
const VIEWER_CHROME: f32 = 40.0;
const VIEWER_MARGIN: f32 = 14.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Close,
    Copy,
    Favorite,
    Previous,
    Next,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Gate {
    #[default]
    Fresh,
    Settling,
    Armed,
}

pub struct Viewer {
    sequence: Vec<AssetId>,
    slot: usize,
    source: Option<Blade>,
    texture: Option<TextureHandle>,
    loading: bool,
    fault: Option<String>,
    gate: Gate,
    copy: Option<Receiver<Result<()>>>,
}

impl Viewer {
    pub fn open(sequence: Vec<AssetId>, slot: usize) -> Self {
        Self {
            sequence,
            slot,
            source: None,
            texture: None,
            loading: false,
            fault: None,
            gate: Gate::Fresh,
            copy: None,
        }
    }

    pub fn asset_id(&self) -> &AssetId {
        &self.sequence[self.slot]
    }

    #[cfg(feature = "egui-test")]
    pub fn ready(&self) -> bool {
        self.texture.is_some()
    }

    pub fn arm(&mut self, ctx: &egui::Context) -> Option<(AssetId, [u32; 2])> {
        if self.loading || self.texture.is_some() || self.fault.is_some() {
            return None;
        }
        self.loading = true;
        let screen = ctx.content_rect().size();
        let points = egui::vec2(
            (screen.x - VIEWER_MARGIN * 2.0).max(64.0),
            (screen.y - VIEWER_MARGIN * 2.0 - VIEWER_CHROME).max(64.0),
        );
        let pixels = points * ctx.pixels_per_point();
        Some((
            self.asset_id().clone(),
            [
                pixels.x.ceil().max(1.0) as u32,
                pixels.y.ceil().max(1.0) as u32,
            ],
        ))
    }

    pub fn install(
        &mut self,
        ctx: &egui::Context,
        asset_id: &AssetId,
        source: Blade,
        display: Option<&Blade>,
    ) {
        if self.asset_id() != asset_id {
            return;
        }
        let raster = display.unwrap_or(&source);
        let image = ColorImage::from_rgba_unmultiplied([raster.width, raster.height], &raster.rgba);
        self.texture = Some(ctx.load_texture(
            format!("picmash-viewer-{asset_id}"),
            image,
            TextureOptions::LINEAR,
        ));
        self.source = Some(source);
        self.loading = false;
        self.fault = None;
    }

    pub fn fail(&mut self, asset_id: &AssetId, message: String) {
        if self.asset_id() == asset_id {
            self.loading = false;
            self.fault = Some(message);
        }
    }

    pub fn navigate(&mut self, action: Action) -> bool {
        let target = match action {
            Action::Previous => self.slot.checked_sub(1),
            Action::Next => self
                .slot
                .checked_add(1)
                .filter(|slot| *slot < self.sequence.len()),
            Action::Close | Action::Copy | Action::Favorite => None,
        };
        let Some(target) = target else {
            return false;
        };
        self.slot = target;
        self.source = None;
        self.texture = None;
        self.loading = false;
        self.fault = None;
        self.gate = Gate::Fresh;
        true
    }

    pub fn begin_copy(&mut self) -> Result<bool> {
        let Some(blade) = self.source.clone().filter(|_| self.copy.is_none()) else {
            return Ok(false);
        };
        let (send, receive) = bounded(1);
        thread::Builder::new()
            .name("picmash-clipboard".to_owned())
            .spawn(move || {
                let result =
                    Clipboard::new()
                        .context("open clipboard")
                        .and_then(|mut clipboard| {
                            clipboard
                                .set_image(ImageData {
                                    width: blade.width,
                                    height: blade.height,
                                    bytes: Cow::Owned(blade.rgba),
                                })
                                .context("copy image")
                        });
                let _sent = send.send(result);
            })
            .context("spawn clipboard worker")?;
        self.copy = Some(receive);
        Ok(true)
    }

    pub fn poll_copy(&mut self, ctx: &egui::Context) -> Option<Result<()>> {
        let receive = self.copy.as_ref()?;
        match receive.try_recv() {
            Ok(result) => {
                self.copy = None;
                Some(result)
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(24));
                None
            }
            Err(TryRecvError::Disconnected) => {
                self.copy = None;
                Some(Err(anyhow!("clipboard worker stopped without a result")))
            }
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        water: &mut Surface,
        card: &Card,
        inputs_enabled: bool,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        if inputs_enabled && !ctx.text_edit_focused() {
            ctx.input(|input| {
                for (key, action) in [
                    (egui::Key::ArrowLeft, Action::Previous),
                    (egui::Key::ArrowRight, Action::Next),
                    (egui::Key::C, Action::Copy),
                    (egui::Key::Escape, Action::Close),
                ] {
                    if exact_key_pressed(input, key) {
                        actions.push(action);
                    }
                }
            });
        }

        let layer = layer();
        ctx.memory_mut(|memory| memory.set_modal_layer(layer));
        water.begin_pond(true);
        let screen = ctx.content_rect();
        let image_box = image_box(card, screen.size());
        let body = egui::vec2(image_box.x, image_box.y + VIEWER_CHROME);
        let window_frame = egui::Frame::window(&ctx.global_style());
        let window_size = body + window_frame.total_margin().sum();
        let window = egui::Window::new(VIEWER_ID)
            .id(egui::Id::new(VIEWER_ID))
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .frame(window_frame)
            .fixed_size(window_size)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                viewer_bar(ui, water, card, self, &mut actions);
                let (rect, response) = ui.allocate_exact_size(image_box, egui::Sense::click());
                witness::response(ui, Target::Viewer, &response);
                if let Some(texture) = &self.texture {
                    ui.painter().image(
                        texture.id(),
                        rect,
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    water.pond_surface(rect);
                    if response.clicked()
                        && let Some(position) = response.interact_pointer_pos()
                    {
                        water.touch(position);
                    }
                } else {
                    let text = self.fault.as_deref().unwrap_or("LOADING");
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        text,
                        egui::FontId::new(13.0, egui::FontFamily::Monospace),
                        if self.fault.is_some() {
                            chrome::HOT
                        } else {
                            chrome::MUTED
                        },
                    );
                }
            });

        let outside = inputs_enabled
            && self.gate == Gate::Armed
            && window
                .as_ref()
                .is_some_and(|window| outside_click(ctx, window.response.rect));
        if outside {
            actions.push(Action::Close);
        }
        self.gate = match self.gate {
            Gate::Fresh => Gate::Settling,
            Gate::Settling | Gate::Armed => Gate::Armed,
        };
        if self.gate != Gate::Armed {
            ctx.request_repaint();
        }
        actions
    }
}

fn viewer_bar(
    ui: &mut egui::Ui,
    water: &mut Surface,
    card: &Card,
    viewer: &Viewer,
    actions: &mut Vec<Action>,
) {
    let _bar = egui::Frame::new()
        .fill(chrome::RAISED)
        .stroke(egui::Stroke::new(1.0, chrome::EDGE))
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            let _row = ui.horizontal(|ui| {
                let _name = ui.label(chrome::section_title(file_name(card)).size(13.0));
                let score = card.preference_score.map_or_else(
                    || "UNRANKED".to_owned(),
                    |score| format!("PREF {score:+.2}"),
                );
                let _meta = ui.label(chrome::muted(format!(
                    "{score} · {} DUELS · {}×{}",
                    card.duel_count, card.width, card.height
                )));
                let _controls =
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let close = Monoglyph::symbol(Symbol::Remove)
                            .finish(MonoglyphFinish::BrightCut)
                            .size(MechanismSize::Medium)
                            .show(ui)
                            .on_hover_text("Close viewer");
                        water.monoglyph(&close);
                        witness::response(ui, Target::ViewerClose, &close);
                        if close.clicked() {
                            actions.push(Action::Close);
                        }
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
                        if favorite.clicked() {
                            actions.push(Action::Favorite);
                        }
                        let copy = command_plate(ui, viewer.source.is_some(), "COPY [C]")
                            .on_hover_text("Copy the full-resolution image");
                        witness::response(ui, Target::ViewerCopy, &copy);
                        if copy.clicked() {
                            actions.push(Action::Copy);
                        }
                    });
            });
        });
}

fn command_plate(ui: &mut egui::Ui, enabled: bool, text: &str) -> egui::Response {
    let text = egui::RichText::new(text)
        .size(13.0)
        .strong()
        .color(chrome::TEXT);
    let response = ui.add_enabled(
        enabled,
        egui::Button::new(text).min_size(egui::vec2(24.0, 20.0)),
    );
    chrome::tension(ui, &response);
    response
}

fn image_box(card: &Card, screen: egui::Vec2) -> egui::Vec2 {
    let bounds = egui::vec2(
        (screen.x - VIEWER_MARGIN * 2.0).max(64.0),
        (screen.y - VIEWER_MARGIN * 2.0 - VIEWER_CHROME).max(64.0),
    );
    let image = if card.rotation_quarters.is_multiple_of(2) {
        egui::vec2(card.width as f32, card.height as f32)
    } else {
        egui::vec2(card.height as f32, card.width as f32)
    };
    contain_native(image, bounds)
}

fn contain_native(image: egui::Vec2, bounds: egui::Vec2) -> egui::Vec2 {
    if image.x <= 0.0 || image.y <= 0.0 {
        return bounds;
    }
    image * (bounds.x / image.x).min(bounds.y / image.y).min(1.0)
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

fn outside_click(ctx: &egui::Context, rect: egui::Rect) -> bool {
    ctx.input(|input| {
        input.pointer.any_click()
            && input
                .pointer
                .interact_pos()
                .is_some_and(|position| !rect.contains(position))
    })
}

fn layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Middle, egui::Id::new(VIEWER_ID))
}

fn file_name(card: &Card) -> String {
    card.path.file_name().map_or_else(
        || card.path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}
