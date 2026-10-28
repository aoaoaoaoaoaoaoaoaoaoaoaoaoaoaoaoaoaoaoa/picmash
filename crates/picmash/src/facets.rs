//! Browsing by provenance: the scope that selects which images the gallery
//! shows, by the source or stream they were admitted from.

use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

use brass_poolrooms::chrome;
use picmash_contract::Target;
use picmash_engine::{Source, SourceId, Stream, StreamId};

use crate::{witness, worker::Card};

const FACET_HEIGHT: f32 = 24.0;
const STREAM_LIST_HEIGHT: f32 = 360.0;
/// Streams beyond which the list offers a title filter.
const STREAM_FILTER_THRESHOLD: usize = 12;

/// Which images the gallery admits, by provenance.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum Scope {
    #[default]
    All,
    /// Images admitted from no source.
    Unsourced,
    Source(SourceId),
    Stream(SourceId, StreamId),
}

impl Scope {
    pub fn admits(&self, card: &Card) -> bool {
        match self {
            Self::All => true,
            Self::Unsourced => card.provenance.is_empty(),
            Self::Source(source) => card
                .provenance
                .iter()
                .any(|provenance| provenance.source.id() == *source),
            Self::Stream(source, stream) => card.provenance.iter().any(|provenance| {
                provenance.stream.id == *stream && provenance.source.id() == *source
            }),
        }
    }

    pub const fn source(&self) -> Option<&SourceId> {
        match self {
            Self::All | Self::Unsourced => None,
            Self::Source(source) | Self::Stream(source, _) => Some(source),
        }
    }
}

/// How many images a facet admits: overall, which orders the list so that it
/// never reorders as the favorites filter toggles, and under that filter,
/// which is shown.
#[derive(Clone, Copy, Default)]
struct Tally {
    total: usize,
    shown: usize,
}

impl Tally {
    fn count(&mut self, shown: bool) {
        self.total += 1;
        self.shown += usize::from(shown);
    }
}

struct SourceFacet {
    id: SourceId,
    source: Source,
    tally: Tally,
    streams: Vec<(Stream, Tally)>,
}

/// The sources and streams of a collection's visible images.
#[derive(Default)]
pub struct Facets {
    all: Tally,
    unsourced: Tally,
    sources: Vec<SourceFacet>,
}

impl Facets {
    pub fn tally(cards: &[Card], favorites_only: bool) -> Self {
        let mut facets = Self::default();
        let mut sources =
            HashMap::<SourceId, (Source, Tally, HashMap<StreamId, (Stream, Tally)>)>::new();
        for card in cards {
            let shown = !favorites_only || card.favorite;
            facets.all.count(shown);
            if card.provenance.is_empty() {
                facets.unsourced.count(shown);
            }
            let mut counted = HashSet::new();
            for provenance in &card.provenance {
                let id = provenance.source.id();
                let (_, tally, streams) = sources.entry(id.clone()).or_insert_with(|| {
                    (provenance.source.clone(), Tally::default(), HashMap::new())
                });
                if counted.insert((id.clone(), None)) {
                    tally.count(shown);
                }
                if counted.insert((id, Some(provenance.stream.id.clone()))) {
                    streams
                        .entry(provenance.stream.id.clone())
                        .or_insert_with(|| (provenance.stream.clone(), Tally::default()))
                        .1
                        .count(shown);
                }
            }
        }
        facets.sources = sources
            .into_iter()
            .map(|(id, (source, tally, streams))| {
                let mut streams = streams.into_values().collect::<Vec<_>>();
                streams.sort_by(|(left, left_tally), (right, right_tally)| {
                    (Reverse(left_tally.total), &left.title, &left.id).cmp(&(
                        Reverse(right_tally.total),
                        &right.title,
                        &right.id,
                    ))
                });
                SourceFacet {
                    id,
                    source,
                    tally,
                    streams,
                }
            })
            .collect();
        facets.sources.sort_by_cached_key(|facet| {
            (
                Reverse(facet.tally.total),
                source_label(&facet.source),
                facet.id.clone(),
            )
        });
        facets
    }

    /// Whether `scope` still names something these facets hold.
    pub fn holds(&self, scope: &Scope) -> bool {
        match scope {
            Scope::All => true,
            Scope::Unsourced => self.unsourced.total > 0,
            Scope::Source(source) => self.sources.iter().any(|facet| facet.id == *source),
            Scope::Stream(source, stream) => self.sources.iter().any(|facet| {
                facet.id == *source
                    && facet
                        .streams
                        .iter()
                        .any(|(candidate, _)| candidate.id == *stream)
            }),
        }
    }

    /// Draws the scope chooser and returns the scope picked, if any. Nothing
    /// is drawn for a collection admitted from no source.
    pub fn show(&self, ui: &mut egui::Ui, scope: &Scope, filter: &mut String) -> Option<Scope> {
        if self.sources.is_empty() {
            return None;
        }
        let mut picked = None;
        ui.add_space(6.0);
        let _heading = ui.label(chrome::eyebrow("SOURCE"));
        if facet(
            ui,
            Target::ScopeAll,
            chrome::section_title("ALL"),
            self.all,
            *scope == Scope::All,
        ) {
            picked = Some(Scope::All);
        }
        for (ordinal, source) in self.sources.iter().enumerate() {
            let selected = scope.source() == Some(&source.id);
            let label = chrome::section_title(source_label(&source.source));
            if facet(
                ui,
                Target::ScopeSource(ordinal),
                label,
                source.tally,
                selected,
            ) {
                picked = Some(if *scope == Scope::Source(source.id.clone()) {
                    Scope::All
                } else {
                    Scope::Source(source.id.clone())
                });
            }
        }
        if self.unsourced.total > 0 {
            let selected = *scope == Scope::Unsourced;
            let label = chrome::section_title("NOT IMPORTED");
            if facet(ui, Target::ScopeUnsourced, label, self.unsourced, selected) {
                picked = Some(if selected {
                    Scope::All
                } else {
                    Scope::Unsourced
                });
            }
        }
        let chosen = scope
            .source()
            .and_then(|id| self.sources.iter().find(|source| source.id == *id));
        if let Some(source) = chosen {
            ui.add_space(6.0);
            let _heading = ui.label(chrome::eyebrow(stream_noun(&source.source)));
            if source.streams.len() > STREAM_FILTER_THRESHOLD {
                let _filter = ui.add(
                    egui::TextEdit::singleline(filter)
                        .hint_text("FILTER")
                        .desired_width(f32::INFINITY),
                );
            }
            let needle = filter.to_lowercase();
            let _streams = egui::ScrollArea::vertical()
                .id_salt("picmash-streams")
                .max_height(STREAM_LIST_HEIGHT)
                .show(ui, |ui| {
                    for (ordinal, (stream, tally)) in source.streams.iter().enumerate() {
                        let selected = matches!(scope, Scope::Stream(_, id) if *id == stream.id);
                        if !selected
                            && source.streams.len() > STREAM_FILTER_THRESHOLD
                            && !stream.title.to_lowercase().contains(&needle)
                        {
                            continue;
                        }
                        let label = chrome::TypeRole::Label
                            .text(&stream.title)
                            .color(chrome::TEXT);
                        let target = Target::ScopeStream(ordinal);
                        if facet(ui, target, label, *tally, selected) {
                            picked = Some(if selected {
                                Scope::Source(source.id.clone())
                            } else {
                                Scope::Stream(source.id.clone(), stream.id.clone())
                            });
                        }
                    }
                });
        }
        picked
    }
}

/// How a source is named wherever it is shown.
pub fn source_label(source: &Source) -> String {
    match source {
        Source::FourChanBoard(board) => format!("4CHAN /{}/", board.to_ascii_uppercase()),
        Source::LocalDirectory(root) => root.file_name().map_or_else(
            || root.display().to_string(),
            |name| name.to_string_lossy().to_uppercase(),
        ),
    }
}

/// What a source's streams are called where the user meets them.
pub const fn stream_noun(source: &Source) -> &'static str {
    match source {
        Source::FourChanBoard(_) => "THREAD",
        Source::LocalDirectory(_) => "FOLDER",
    }
}

fn facet(
    ui: &mut egui::Ui,
    target: Target,
    label: egui::RichText,
    tally: Tally,
    selected: bool,
) -> bool {
    let button = egui::Button::new(label)
        .right_text(chrome::muted(tally.shown.to_string()))
        .selected(selected)
        .truncate()
        .min_size(egui::vec2(ui.available_width(), FACET_HEIGHT));
    let response = ui.add(button);
    chrome::tension(ui, &response);
    witness::response(ui, target, &response);
    response.clicked()
}
