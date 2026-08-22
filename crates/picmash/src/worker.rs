use anyhow::{Context as _, Result, anyhow, bail};
use atomic_write_file::AtomicWriteFile;
use crossbeam_channel::{Receiver, Sender, TrySendError, after, bounded, select};
use eternalist_apps::NativeWake;
use image::{RgbaImage, imageops};
use picmash_contract::Side;
use picmash_engine::{
    AssetId, AssetView, CollectionId, CommandId, ComparisonPrompt, Engine, PreferenceEvaluation,
    PresentedAsset, ScanProgress, SessionId, canonical_image,
};
use std::{
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    configuration::{ImportPolicy, Probability, RemoteConfig},
    remote::{
        ArchiveEffect, ArchiveLane, DuelVictor, Effect as RemoteEffect, Prepared, PromotionIntent,
        PromotionJudgment, Reactor, Summary as RemoteSummary,
    },
    xdg::Lair,
};

const COMMAND_CAPACITY: usize = 32;
const EVENT_CAPACITY: usize = 64;
const PROMPT_EDGE: u32 = 1_800;
const THUMB_EDGE: u32 = 360;
const CONTEXT_REVISION: &str = "poolrooms-pairwise-v1";

#[derive(Clone, Debug)]
pub struct Card {
    pub presentation: PresentedAsset,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub occurrence_count: u32,
    pub favorite: bool,
    pub duel_count: u32,
    pub preference_score: Option<f64>,
}

impl From<&AssetView> for Card {
    fn from(view: &AssetView) -> Self {
        Self {
            presentation: PresentedAsset {
                asset_id: view.id.clone(),
                occurrence_id: view.occurrence.id,
                render: view.occurrence.render.clone(),
                rotation_quarters: view.occurrence.rotation_quarters,
            },
            path: view.occurrence.path.clone(),
            width: view.occurrence.width,
            height: view.occurrence.height,
            occurrence_count: view.occurrence_count,
            favorite: view.favorite,
            duel_count: view.duel_count,
            preference_score: view.preference_score,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Blade {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Pair {
    pub left: PairCard,
    pub right: PairCard,
    pub left_blade: Blade,
    pub right_blade: Blade,
}

#[derive(Clone, Debug)]
pub enum PairCard {
    Local(Card),
    Remote(RemoteCard),
}

impl PairCard {
    pub const fn width(&self) -> u32 {
        match self {
            Self::Local(card) if card.presentation.rotation_quarters % 2 == 0 => card.width,
            Self::Local(card) => card.height,
            Self::Remote(card) => card.width,
        }
    }

    pub const fn height(&self) -> u32 {
        match self {
            Self::Local(card) if card.presentation.rotation_quarters % 2 == 0 => card.height,
            Self::Local(card) => card.width,
            Self::Remote(card) => card.height,
        }
    }

    #[cfg(feature = "egui-test")]
    pub const fn rotation_quarters(&self) -> u8 {
        match self {
            Self::Local(card) => card.presentation.rotation_quarters,
            Self::Remote(card) => card.rotation_quarters,
        }
    }

    pub const fn favorite(&self) -> bool {
        match self {
            Self::Local(card) => card.favorite,
            Self::Remote(_) => false,
        }
    }

    pub const fn duel_count(&self) -> Option<u32> {
        match self {
            Self::Local(card) => Some(card.duel_count),
            Self::Remote(_) => None,
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::Local(card) => &card.path,
            Self::Remote(card) => &card.path,
        }
    }

    pub const fn remote(&self) -> Option<&RemoteCard> {
        match self {
            Self::Local(_) => None,
            Self::Remote(card) => Some(card),
        }
    }

    pub(crate) fn local_mut(&mut self) -> Option<&mut Card> {
        match self {
            Self::Local(card) => Some(card),
            Self::Remote(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RemoteCard {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    #[cfg(feature = "egui-test")]
    pub rotation_quarters: u8,
    pub title: String,
    pub source: String,
    pub stream: String,
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub root: PathBuf,
    pub visible_assets: usize,
    pub scan_failures: usize,
    pub evaluation: Option<PreferenceEvaluation>,
}

#[derive(Debug)]
pub enum Command {
    Load(PathBuf),
    Rescan,
    Choose(Side),
    Favorite(Side),
    FavoriteAsset(AssetId),
    Hide(Side),
    Rotate(Side),
    VetoStream(Side),
    ConfigureRemote(RemoteConfig),
    Thumbnail { asset_id: AssetId, bucket: u8 },
    Full { asset_id: AssetId, bound: [u32; 2] },
}

#[derive(Debug)]
pub enum Event {
    NeedCollection,
    Busy(&'static str),
    ScanStarted(PathBuf),
    ScanProgress(ScanProgress),
    Catalog {
        summary: Summary,
        cards: Vec<Card>,
    },
    CatalogRefreshed {
        summary: Summary,
        cards: Vec<Card>,
    },
    Pair(Box<Pair>),
    Remote(RemoteSummary),
    RemoteFault(String),
    ArchiveFault(String),
    NoComparison,
    Favorite {
        asset_id: AssetId,
        active: bool,
    },
    Thumbnail {
        asset_id: AssetId,
        bucket: u8,
        blade: Blade,
    },
    ThumbnailFault {
        asset_id: AssetId,
        bucket: u8,
        message: String,
    },
    Full {
        asset_id: AssetId,
        source: Blade,
        display: Option<Blade>,
    },
    FullFault {
        asset_id: AssetId,
        message: String,
    },
    Fault(String),
}

pub struct Worker {
    commands: Sender<Command>,
    pub events: Receiver<Event>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(
        ctx: &egui::Context,
        lair: Lair,
        initial: Option<PathBuf>,
        remote: RemoteConfig,
    ) -> Result<Self> {
        let (commands, command_rx) = bounded(COMMAND_CAPACITY);
        let (event_tx, events) = bounded(EVENT_CAPACITY);
        let wake = NativeWake::from_context(ctx);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("picmash-engine".to_owned())
            .spawn(move || {
                run(
                    command_rx,
                    event_tx,
                    wake,
                    lair,
                    initial,
                    remote,
                    worker_stop,
                );
            })
            .context("spawn Picmash engine worker")?;
        Ok(Self {
            commands,
            events,
            stop,
            thread: Some(thread),
        })
    }

    pub fn send(&self, command: Command) -> Result<()> {
        self.commands
            .try_send(command)
            .map_err(|error| match error {
                TrySendError::Full(_) => anyhow!("Picmash is still tending its previous commands"),
                TrySendError::Disconnected(_) => anyhow!("Picmash's engine worker has stopped"),
            })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let (grave, _receiver) = bounded(0);
        let commands = std::mem::replace(&mut self.commands, grave);
        drop(commands);
        let (_sender, grave) = bounded(0);
        let events = std::mem::replace(&mut self.events, grave);
        drop(events);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            eprintln!("Picmash engine worker panicked during retirement");
        }
    }
}

struct EngineState {
    engine: Engine,
    lair: Lair,
    stop: Arc<AtomicBool>,
    collection: Option<CollectionId>,
    session: Option<SessionId>,
    prompt: Option<LivePrompt>,
    cards: Vec<Card>,
    scan_failures: usize,
    remote: Reactor,
    archive: ArchiveLane,
    remote_probability: Probability,
    lottery: Lottery,
    remote_anchor_cursor: usize,
}

enum LivePrompt {
    Local {
        prompt: ComparisonPrompt,
        born: Instant,
    },
    Remote {
        candidate: Box<Prepared>,
        anchor: PresentedAsset,
        rotation_quarters: u8,
        born: Instant,
        duel_command: CommandId,
        duel_response_ms: Option<u32>,
        favorite_command: CommandId,
    },
}

struct Lottery(u64);

impl Lottery {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos() as u64);
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn draw(&mut self) -> u16 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 1_000) as u16
    }
}

fn run(
    commands: Receiver<Command>,
    events: Sender<Event>,
    wake: NativeWake,
    lair: Lair,
    initial: Option<PathBuf>,
    remote_config: RemoteConfig,
    stop: Arc<AtomicBool>,
) {
    let result = Engine::open(lair.database())
        .map_err(anyhow::Error::from)
        .and_then(|engine| {
            let mut remote = Reactor::open(&lair, &remote_config)?;
            let archive = ArchiveLane::raise()?;
            for intent in remote.take_restored_promotions() {
                reserve_promotion(&engine, &intent)?;
                archive.submit(intent)?;
            }
            Ok(EngineState {
                engine,
                lair,
                stop: Arc::clone(&stop),
                collection: None,
                session: None,
                prompt: None,
                cards: Vec::new(),
                scan_failures: 0,
                remote,
                archive,
                remote_probability: remote_config.sample_probability,
                lottery: Lottery::new(),
                remote_anchor_cursor: 0,
            })
        });
    let mut state = match result {
        Ok(state) => state,
        Err(error) => {
            publish(&events, &wake, Event::Fault(format!("{error:#}")));
            return;
        }
    };
    let restored = initial.or_else(|| restore_collection(&state.lair.active_collection()));
    match restored {
        Some(root) => conduct(&mut state, &events, &wake, Command::Load(root)),
        None => publish(&events, &wake, Event::NeedCollection),
    }
    let remote_completions = state.remote.completions().clone();
    let archive_completions = state.archive.completions().clone();
    while !stop.load(Ordering::Acquire) {
        let deadline = after(state.remote.wait());
        select! {
            recv(commands) -> command => match command {
                Ok(command) => conduct(&mut state, &events, &wake, command),
                Err(_) => break,
            },
            recv(remote_completions) -> completion => match completion {
                Ok(completion) => service_remote(&mut state, &events, &wake, Some(completion)),
                Err(_) => break,
            },
            recv(archive_completions) -> completion => match completion {
                Ok(completion) => service_archive(&mut state, &events, &wake, completion),
                Err(_) => break,
            },
            recv(deadline) -> _ => service_remote(&mut state, &events, &wake, None),
        }
    }
    retire_session(&state);
}

fn conduct(state: &mut EngineState, events: &Sender<Event>, wake: &NativeWake, command: Command) {
    let result = match command {
        Command::Load(root) => load_collection(state, events, wake, &root),
        Command::Rescan => rescan(state, events, wake),
        Command::Choose(side) => choose(state, events, wake, side),
        Command::Favorite(side) => favorite(state, events, wake, side),
        Command::FavoriteAsset(asset_id) => favorite_asset(state, events, wake, &asset_id),
        Command::Hide(side) => hide(state, events, wake, side),
        Command::Rotate(side) => rotate(state, events, wake, side),
        Command::VetoStream(side) => veto_stream(state, events, wake, side),
        Command::ConfigureRemote(config) => configure_remote(state, events, wake, &config),
        Command::Thumbnail { asset_id, bucket } => {
            let result = thumbnail(state, events, wake, &asset_id, bucket);
            if let Err(error) = result {
                publish(
                    events,
                    wake,
                    Event::ThumbnailFault {
                        asset_id,
                        bucket,
                        message: format!("{error:#}"),
                    },
                );
            }
            return;
        }
        Command::Full { asset_id, bound } => {
            let result = full(state, events, wake, &asset_id, bound);
            if let Err(error) = result {
                publish(
                    events,
                    wake,
                    Event::FullFault {
                        asset_id,
                        message: format!("{error:#}"),
                    },
                );
            }
            return;
        }
    };
    if let Err(error) = result {
        publish(events, wake, Event::Fault(format!("{error:#}")));
    }
}

fn load_collection(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    root: &Path,
) -> Result<()> {
    publish(events, wake, Event::ScanStarted(root.to_path_buf()));
    let stop = Arc::clone(&state.stop);
    let Some(scan) = state.engine.scan_with_control(
        root,
        || stop.load(Ordering::Acquire),
        |progress| publish(events, wake, Event::ScanProgress(progress)),
    )?
    else {
        return Ok(());
    };
    if let Some(session) = &state.session {
        state.engine.end_session(session)?;
    }
    let session = state
        .engine
        .start_session(scan.collection_id, CONTEXT_REVISION)?;
    persist_collection(&state.lair.active_collection(), root)?;
    state.collection = Some(scan.collection_id);
    state.session = Some(session.id);
    state.prompt = None;
    state.scan_failures = scan.failures.len();
    state.remote.activate()?;
    publish(events, wake, Event::Remote(state.remote.summary()));
    publish_collection(state, events, wake)
}

fn rescan(state: &mut EngineState, events: &Sender<Event>, wake: &NativeWake) -> Result<()> {
    let collection = active_collection(state)?;
    let root = state.engine.collection(collection)?.root;
    load_collection(state, events, wake, &root)
}

fn choose(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    if let Some(LivePrompt::Remote {
        born,
        duel_response_ms,
        ..
    }) = state.prompt.as_mut()
        && duel_response_ms.is_none()
    {
        *duel_response_ms = Some(elapsed_ms(*born));
    }
    let prompt = state
        .prompt
        .as_ref()
        .context("there is no live comparison to judge")?;
    match prompt {
        LivePrompt::Local { prompt, born } => {
            publish(events, wake, Event::Busy("FORGING NEXT PAIR"));
            let prompt = prompt.clone();
            let response_ms = elapsed_ms(*born);
            let winner = match side {
                Side::Left => &prompt.left.asset_id,
                Side::Right => &prompt.right.asset_id,
            };
            state.engine.record_comparison(
                &prompt.id,
                winner,
                &CommandId::fresh(),
                Some(response_ms),
            )?;
            state.prompt = None;
            publish_collection(state, events, wake)
        }
        LivePrompt::Remote {
            candidate,
            anchor,
            rotation_quarters,
            born,
            duel_command,
            duel_response_ms,
            ..
        } => {
            let candidate = candidate.clone();
            let anchor = anchor.clone();
            let rotation_quarters = *rotation_quarters;
            let duel_command = duel_command.clone();
            let response_ms = duel_response_ms.unwrap_or_else(|| elapsed_ms(*born));
            let remote_won = side == Side::Right;
            if remote_won || candidate.discovery.import_policy == ImportPolicy::NotX {
                let session_id = active_session(state)?.clone();
                queue_promotion(
                    state,
                    &candidate,
                    rotation_quarters,
                    PromotionJudgment::Duel {
                        session_id,
                        anchor,
                        victor: if remote_won {
                            DuelVictor::Challenger
                        } else {
                            DuelVictor::Anchor
                        },
                        command_id: duel_command,
                        response_ms,
                    },
                )?;
                publish(events, wake, Event::Remote(state.remote.summary()));
            } else {
                state.remote.note_duel(&candidate, false)?;
                state.remote.reject_offer()?;
            }
            state.prompt = None;
            forge_pair(state, events, wake)
        }
    }
}

fn favorite(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    match state
        .prompt
        .as_ref()
        .context("there is no live comparison")?
    {
        LivePrompt::Local { prompt, .. } => {
            let asset_id = match side {
                Side::Left => prompt.left.asset_id.clone(),
                Side::Right => prompt.right.asset_id.clone(),
            };
            favorite_asset(state, events, wake, &asset_id)
        }
        LivePrompt::Remote {
            candidate,
            anchor,
            rotation_quarters,
            favorite_command,
            ..
        } => match side {
            Side::Left => {
                let anchor = anchor.clone();
                favorite_asset(state, events, wake, &anchor.asset_id)
            }
            Side::Right => {
                let candidate = candidate.clone();
                let rotation_quarters = *rotation_quarters;
                let favorite_command = favorite_command.clone();
                let session_id = active_session(state)?.clone();
                queue_promotion(
                    state,
                    &candidate,
                    rotation_quarters,
                    PromotionJudgment::Favorite {
                        session_id,
                        command_id: favorite_command,
                    },
                )?;
                state.prompt = None;
                publish(events, wake, Event::Remote(state.remote.summary()));
                forge_pair(state, events, wake)
            }
        },
    }
}

fn favorite_asset(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    asset_id: &AssetId,
) -> Result<()> {
    let active = !state
        .cards
        .iter()
        .find(|card| &card.presentation.asset_id == asset_id)
        .context("comparison asset left the active catalog")?
        .favorite;
    state.engine.set_favorite(
        active_session(state)?,
        asset_id,
        active,
        &CommandId::fresh(),
    )?;
    if let Some(card) = state
        .cards
        .iter_mut()
        .find(|card| &card.presentation.asset_id == asset_id)
    {
        card.favorite = active;
    }
    publish(
        events,
        wake,
        Event::Favorite {
            asset_id: asset_id.clone(),
            active,
        },
    );
    Ok(())
}

fn hide(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    match state
        .prompt
        .as_ref()
        .context("there is no live comparison")?
    {
        LivePrompt::Local { prompt, .. } => {
            let asset = match side {
                Side::Left => prompt.left.asset_id.clone(),
                Side::Right => prompt.right.asset_id.clone(),
            };
            publish(events, wake, Event::Busy("WITHDRAWING IMAGE"));
            state
                .engine
                .set_hidden(active_collection(state)?, &asset, true)?;
            state.prompt = None;
            publish_collection(state, events, wake)
        }
        LivePrompt::Remote { anchor, .. } => match side {
            Side::Left => {
                let anchor = anchor.clone();
                publish(events, wake, Event::Busy("WITHDRAWING IMAGE"));
                state
                    .engine
                    .set_hidden(active_collection(state)?, &anchor.asset_id, true)?;
                state.prompt = None;
                publish_collection(state, events, wake)
            }
            Side::Right => {
                state.remote.reject_offer()?;
                state.prompt = None;
                forge_pair(state, events, wake)
            }
        },
    }
}

fn rotate(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    if side == Side::Right
        && let Some(LivePrompt::Remote {
            candidate,
            rotation_quarters,
            ..
        }) = state.prompt.as_mut()
    {
        let candidate = candidate.clone();
        *rotation_quarters = rotation_quarters.wrapping_add(1) % 4;
        return publish_remote_pair(state, events, wake, &candidate);
    }
    let occurrence = match state
        .prompt
        .as_ref()
        .context("there is no live comparison")?
    {
        LivePrompt::Local { prompt, .. } => match side {
            Side::Left => prompt.left.occurrence_id,
            Side::Right => prompt.right.occurrence_id,
        },
        LivePrompt::Remote { anchor, .. } => anchor.occurrence_id,
    };
    publish(events, wake, Event::Busy("TURNING IMAGE"));
    let _rotation = state.engine.rotate_occurrence(occurrence, 1)?;
    state.prompt = None;
    publish_collection(state, events, wake)
}

fn thumbnail(
    state: &EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    asset_id: &AssetId,
    bucket: u8,
) -> Result<()> {
    let card = state
        .cards
        .iter()
        .find(|card| &card.presentation.asset_id == asset_id)
        .context("thumbnail asset left the active catalog")?;
    let blade = decode_thumbnail(
        &card.path,
        card.presentation.rotation_quarters,
        THUMB_EDGE.saturating_mul(1_u32 << bucket.min(2)),
    )?;
    publish(
        events,
        wake,
        Event::Thumbnail {
            asset_id: asset_id.clone(),
            bucket,
            blade,
        },
    );
    Ok(())
}

fn full(
    state: &EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    asset_id: &AssetId,
    bound: [u32; 2],
) -> Result<()> {
    let card = state
        .cards
        .iter()
        .find(|card| &card.presentation.asset_id == asset_id)
        .context("viewer asset left the active catalog")?;
    let image = decode_rgba(&card.path, card.presentation.rotation_quarters)?;
    let display = (image.width() > bound[0] || image.height() > bound[1])
        .then(|| blade(fitted_thumbnail(&image, bound[0], bound[1])));
    publish(
        events,
        wake,
        Event::Full {
            asset_id: asset_id.clone(),
            source: blade(image),
            display,
        },
    );
    Ok(())
}

fn publish_collection(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
) -> Result<()> {
    let collection = active_collection(state)?;
    let (summary, cards) = rebuild_collection(state, collection)?;
    publish(events, wake, Event::Catalog { summary, cards });
    forge_pair(state, events, wake)
}

fn refresh_collection(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    collection: CollectionId,
) -> Result<()> {
    if state.collection != Some(collection) {
        return Ok(());
    }
    let (summary, cards) = rebuild_collection(state, collection)?;
    publish(events, wake, Event::CatalogRefreshed { summary, cards });
    if state.prompt.is_none() {
        forge_pair(state, events, wake)?;
    }
    Ok(())
}

fn rebuild_collection(
    state: &mut EngineState,
    collection: CollectionId,
) -> Result<(Summary, Vec<Card>)> {
    let snapshot = state.engine.rebuild_preferences(collection)?;
    let assets = state.engine.assets(collection)?;
    state.cards = assets.iter().map(Card::from).collect();
    let root = state.engine.collection(collection)?.root;
    Ok((
        Summary {
            root,
            visible_assets: state.cards.len(),
            scan_failures: state.scan_failures,
            evaluation: snapshot.evaluation,
        },
        state.cards.clone(),
    ))
}

fn forge_pair(state: &mut EngineState, events: &Sender<Event>, wake: &NativeWake) -> Result<()> {
    if !state.cards.is_empty()
        && (state.cards.len() < 2 || state.remote_probability.draw(state.lottery.draw()))
        && let Some(candidate) = state.remote.offer()?
    {
        let slot = state.remote_anchor_cursor % state.cards.len();
        state.remote_anchor_cursor = state.remote_anchor_cursor.wrapping_add(1);
        let anchor = state.cards[slot].presentation.clone();
        state.prompt = Some(LivePrompt::Remote {
            candidate: Box::new(candidate.clone()),
            anchor,
            rotation_quarters: 0,
            born: Instant::now(),
            duel_command: CommandId::fresh(),
            duel_response_ms: None,
            favorite_command: CommandId::fresh(),
        });
        publish_remote_pair(state, events, wake, &candidate)?;
        publish(events, wake, Event::Remote(state.remote.summary()));
        return Ok(());
    }
    if state.cards.len() < 2 {
        state.prompt = None;
        publish(events, wake, Event::NoComparison);
        return Ok(());
    }
    let prompt = state.engine.propose_comparison(active_session(state)?)?;
    let left = card_for(state, &prompt.left.asset_id)?.clone();
    let right = card_for(state, &prompt.right.asset_id)?.clone();
    let left_blade =
        decode_thumbnail(&left.path, left.presentation.rotation_quarters, PROMPT_EDGE)?;
    let right_blade = decode_thumbnail(
        &right.path,
        right.presentation.rotation_quarters,
        PROMPT_EDGE,
    )?;
    state.prompt = Some(LivePrompt::Local {
        prompt,
        born: Instant::now(),
    });
    publish(
        events,
        wake,
        Event::Pair(Box::new(Pair {
            left: PairCard::Local(left),
            right: PairCard::Local(right),
            left_blade,
            right_blade,
        })),
    );
    Ok(())
}

fn publish_remote_pair(
    state: &EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    candidate: &Prepared,
) -> Result<()> {
    let LivePrompt::Remote {
        anchor,
        rotation_quarters,
        ..
    } = state
        .prompt
        .as_ref()
        .context("remote pair lost its live prompt")?
    else {
        bail!("remote pair was replaced by a local prompt");
    };
    let left = card_for(state, &anchor.asset_id)?.clone();
    let left_blade =
        decode_thumbnail(&left.path, left.presentation.rotation_quarters, PROMPT_EDGE)?;
    let right_blade = decode_thumbnail(&candidate.cache_path, *rotation_quarters, PROMPT_EDGE)?;
    let (width, height) = if rotation_quarters % 2 == 0 {
        (candidate.discovery.width, candidate.discovery.height)
    } else {
        (candidate.discovery.height, candidate.discovery.width)
    };
    publish(
        events,
        wake,
        Event::Pair(Box::new(Pair {
            left: PairCard::Local(left),
            right: PairCard::Remote(RemoteCard {
                path: candidate.cache_path.clone(),
                width,
                height,
                #[cfg(feature = "egui-test")]
                rotation_quarters: *rotation_quarters,
                title: candidate.discovery.title.clone(),
                source: candidate.discovery.source_name.clone(),
                stream: candidate.discovery.stream_title.clone(),
            }),
            left_blade,
            right_blade,
        })),
    );
    Ok(())
}

fn queue_promotion(
    state: &mut EngineState,
    candidate: &Prepared,
    rotation_quarters: u8,
    judgment: PromotionJudgment,
) -> Result<()> {
    let collection = active_collection(state)?;
    let root = state.engine.collection(collection)?.root;
    let intent = state
        .remote
        .begin_promotion(candidate, root, rotation_quarters, judgment)?;
    if let Err(error) = reserve_promotion(&state.engine, &intent) {
        state.remote.abort_promotion(&intent)?;
        return Err(error);
    }
    if let Err(error) = state.archive.submit(intent.clone()) {
        state.remote.abort_promotion(&intent)?;
        state
            .engine
            .cancel_promoted_reservation(intent.judgment.command_id())?;
        return Err(error);
    }
    state.remote.drive()
}

fn reserve_promotion(engine: &Engine, intent: &PromotionIntent) -> Result<()> {
    let identity = intent.candidate.discovery.item_id.as_str();
    match &intent.judgment {
        PromotionJudgment::Duel {
            session_id,
            anchor,
            victor,
            command_id,
            response_ms,
        } => {
            engine.reserve_promoted_comparison(
                session_id,
                identity,
                anchor,
                intent.rotation_quarters,
                *victor,
                command_id,
                *response_ms,
            )?;
        }
        PromotionJudgment::Favorite {
            session_id,
            command_id,
        } => {
            engine.reserve_promoted_favorite(session_id, identity, command_id)?;
        }
    }
    Ok(())
}

fn commit_promotion(
    state: &mut EngineState,
    intent: &PromotionIntent,
    path: &Path,
) -> Result<CollectionId> {
    let session_id = intent.judgment.session_id();
    let collection = state.engine.session_collection(session_id)?;
    let root = state.engine.collection(collection)?.root;
    anyhow::ensure!(
        root == intent.collection_root,
        "promotion collection changed before admission"
    );
    let challenger = state.engine.ingest_occurrence(collection, path)?;
    let identity = intent.candidate.discovery.item_id.as_str();
    match &intent.judgment {
        PromotionJudgment::Duel {
            anchor,
            victor,
            command_id,
            response_ms,
            ..
        } => {
            if challenger == anchor.asset_id {
                state.engine.cancel_promoted_reservation(command_id)?;
            } else {
                state.engine.record_promoted_comparison(
                    session_id,
                    identity,
                    anchor,
                    &challenger,
                    intent.rotation_quarters,
                    *victor,
                    command_id,
                    *response_ms,
                )?;
            }
        }
        PromotionJudgment::Favorite { command_id, .. } => {
            state
                .engine
                .set_promoted_favorite(session_id, identity, &challenger, command_id)?;
        }
    }
    state.remote.seal_promoted(intent)?;
    Ok(collection)
}

fn veto_stream(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    ensure_remote_side(state, side)?;
    state.remote.veto_offer_stream()?;
    state.prompt = None;
    forge_pair(state, events, wake)
}

fn configure_remote(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    config: &RemoteConfig,
) -> Result<()> {
    state.remote_probability = config.sample_probability;
    state.remote.reconfigure(config)?;
    if matches!(state.prompt.as_ref(), Some(LivePrompt::Remote { .. })) {
        state.prompt = None;
    }
    if state.prompt.is_none() && state.collection.is_some() {
        forge_pair(state, events, wake)?;
    }
    publish(events, wake, Event::Remote(state.remote.summary()));
    Ok(())
}

fn service_remote(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    completion: Option<RemoteEffect>,
) {
    let result = match completion {
        Some(completion) => state.remote.settle(completion),
        None => state.remote.drive(),
    };
    match result {
        Ok(()) => {
            publish_foreground(events, wake, Event::Remote(state.remote.summary()));
            if state.prompt.is_none()
                && state.collection.is_some()
                && let Err(error) = forge_pair(state, events, wake)
            {
                publish(events, wake, Event::RemoteFault(format!("{error:#}")));
            }
        }
        Err(error) => {
            state.remote.poison();
            publish(events, wake, Event::RemoteFault(format!("{error:#}")));
        }
    }
}

fn service_archive(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    effect: ArchiveEffect,
) {
    let item = effect.intent.candidate.discovery.item_id.clone();
    let result = effect
        .result
        .map_err(anyhow::Error::msg)
        .and_then(|promotion| commit_promotion(state, &effect.intent, &promotion.path));
    match result {
        Ok(collection) => {
            if let Err(error) = state.remote.drive() {
                state.remote.poison();
                publish(events, wake, Event::RemoteFault(format!("{error:#}")));
            }
            publish_foreground(events, wake, Event::Remote(state.remote.summary()));
            if let Err(error) = refresh_collection(state, events, wake, collection) {
                publish(
                    events,
                    wake,
                    Event::ArchiveFault(format!("admitted archive could not refresh: {error:#}")),
                );
            }
        }
        Err(error) => {
            let message = format!("{error:#}");
            if let Err(store_error) = state.remote.promotion_failed(&item, &message) {
                publish(
                    events,
                    wake,
                    Event::ArchiveFault(format!("{message}; persistence fault: {store_error:#}")),
                );
            } else {
                publish(
                    events,
                    wake,
                    Event::ArchiveFault(format!("{message}; restart Picmash to retry")),
                );
            }
            publish_foreground(events, wake, Event::Remote(state.remote.summary()));
        }
    }
}

fn ensure_remote_side(state: &EngineState, side: Side) -> Result<()> {
    if side == Side::Right && matches!(state.prompt.as_ref(), Some(LivePrompt::Remote { .. })) {
        Ok(())
    } else {
        bail!("only a remote challenger stream can be vetoed")
    }
}

fn elapsed_ms(born: Instant) -> u32 {
    u32::try_from(born.elapsed().as_millis()).unwrap_or(u32::MAX)
}

fn decode_thumbnail(path: &Path, rotation_quarters: u8, edge: u32) -> Result<Blade> {
    let image = decode_rgba(path, rotation_quarters)?;
    Ok(blade(fitted_thumbnail(&image, edge, edge)))
}

fn fitted_thumbnail(image: &RgbaImage, bound_width: u32, bound_height: u32) -> RgbaImage {
    let scale = (f64::from(bound_width.max(1)) / f64::from(image.width()))
        .min(f64::from(bound_height.max(1)) / f64::from(image.height()))
        .min(1.0);
    let width = (f64::from(image.width()) * scale).floor().max(1.0) as u32;
    let height = (f64::from(image.height()) * scale).floor().max(1.0) as u32;
    imageops::thumbnail(image, width, height)
}

fn decode_rgba(path: &Path, rotation_quarters: u8) -> Result<RgbaImage> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let image = canonical_image(&bytes).with_context(|| format!("decode {}", path.display()))?;
    let rgba = image.to_rgba8();
    Ok(match rotation_quarters % 4 {
        1 => imageops::rotate90(&rgba),
        2 => imageops::rotate180(&rgba),
        3 => imageops::rotate270(&rgba),
        _ => rgba,
    })
}

fn blade(image: RgbaImage) -> Blade {
    Blade {
        width: image.width() as usize,
        height: image.height() as usize,
        rgba: image.into_raw(),
    }
}

fn active_collection(state: &EngineState) -> Result<CollectionId> {
    state.collection.context("no collection is active")
}

fn active_session(state: &EngineState) -> Result<&SessionId> {
    state
        .session
        .as_ref()
        .context("no judgment session is active")
}

fn card_for<'a>(state: &'a EngineState, asset: &AssetId) -> Result<&'a Card> {
    state
        .cards
        .iter()
        .find(|card| &card.presentation.asset_id == asset)
        .context("prompt asset left the active catalog")
}

fn publish(events: &Sender<Event>, wake: &NativeWake, event: Event) {
    if events.send(event).is_ok() {
        let _woken = wake.request_repaint();
    }
}

fn publish_foreground(events: &Sender<Event>, wake: &NativeWake, event: Event) {
    if events.send(event).is_ok() {
        let _woken = wake.request_foreground_repaint();
    }
}

fn retire_session(state: &EngineState) {
    if let Some(session) = &state.session
        && let Err(error) = state.engine.end_session(session)
    {
        eprintln!("could not retire Picmash judgment session: {error}");
    }
}

fn persist_collection(path: &Path, root: &Path) -> Result<()> {
    let mut file = AtomicWriteFile::open(path)
        .with_context(|| format!("stage active collection at {}", path.display()))?;
    file.write_all(&path_bytes(root))
        .with_context(|| format!("write active collection at {}", path.display()))?;
    file.commit()
        .with_context(|| format!("commit active collection at {}", path.display()))
}

fn restore_collection(path: &Path) -> Option<PathBuf> {
    fs::read(path).ok().map(path_from_bytes)
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt as _;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().into_owned().into_bytes()
}

#[cfg(unix)]
fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt as _};
    PathBuf::from(OsString::from_vec(bytes))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(&bytes).into_owned())
}
