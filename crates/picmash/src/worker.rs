use anyhow::{Context as _, Result, anyhow, bail};
use atomic_write_file::AtomicWriteFile;
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use eternalist_apps::NativeWake;
use image::{DynamicImage, imageops};
use picmash_contract::Side;
use picmash_engine::{
    AssetId, AssetView, CollectionId, CommandId, ComparisonPrompt, Engine, PreferenceEvaluation,
    SessionId, canonical_image,
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
    time::Instant,
};

use crate::xdg::Lair;

const COMMAND_CAPACITY: usize = 32;
const PROMPT_EDGE: u32 = 1_800;
const THUMB_EDGE: u32 = 360;
const CONTEXT_REVISION: &str = "poolrooms-pairwise-v1";

#[derive(Clone, Debug)]
pub struct Card {
    pub asset_id: AssetId,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub rotation_quarters: u8,
    pub favorite: bool,
    pub duel_count: u32,
    pub preference_score: Option<f64>,
}

impl From<&AssetView> for Card {
    fn from(view: &AssetView) -> Self {
        Self {
            asset_id: view.id.clone(),
            path: view.occurrence.path.clone(),
            width: view.occurrence.width,
            height: view.occurrence.height,
            rotation_quarters: view.occurrence.rotation_quarters,
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
    pub left: Card,
    pub right: Card,
    pub left_blade: Blade,
    pub right_blade: Blade,
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
    Hide(Side),
    Rotate(Side),
    Thumbnail(AssetId),
}

#[derive(Debug)]
pub enum Event {
    NeedCollection,
    Busy(&'static str),
    Catalog { summary: Summary, cards: Vec<Card> },
    Pair(Pair),
    NoComparison,
    Favorite { asset_id: AssetId, active: bool },
    Thumbnail { asset_id: AssetId, blade: Blade },
    Fault(String),
}

pub struct Worker {
    commands: Sender<Command>,
    pub events: Receiver<Event>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(ctx: &egui::Context, lair: Lair, initial: Option<PathBuf>) -> Result<Self> {
        let (commands, command_rx) = bounded(COMMAND_CAPACITY);
        let (event_tx, events) = unbounded();
        let wake = NativeWake::from_context(ctx);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("picmash-engine".to_owned())
            .spawn(move || run(command_rx, event_tx, wake, lair, initial, worker_stop))
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
    prompt: Option<(ComparisonPrompt, Instant)>,
    cards: Vec<Card>,
    scan_failures: usize,
}

fn run(
    commands: Receiver<Command>,
    events: Sender<Event>,
    wake: NativeWake,
    lair: Lair,
    initial: Option<PathBuf>,
    stop: Arc<AtomicBool>,
) {
    let result = Engine::open(lair.database())
        .map(|engine| EngineState {
            engine,
            lair,
            stop: Arc::clone(&stop),
            collection: None,
            session: None,
            prompt: None,
            cards: Vec::new(),
            scan_failures: 0,
        })
        .map_err(anyhow::Error::from);
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
    while !stop.load(Ordering::Acquire)
        && let Ok(command) = commands.recv()
    {
        conduct(&mut state, &events, &wake, command);
    }
    retire_session(&state);
}

fn conduct(state: &mut EngineState, events: &Sender<Event>, wake: &NativeWake, command: Command) {
    let result = match command {
        Command::Load(root) => load_collection(state, events, wake, &root),
        Command::Rescan => rescan(state, events, wake),
        Command::Choose(side) => choose(state, events, wake, side),
        Command::Favorite(side) => favorite(state, events, wake, side),
        Command::Hide(side) => hide(state, events, wake, side),
        Command::Rotate(side) => rotate(state, events, wake, side),
        Command::Thumbnail(asset_id) => thumbnail(state, events, wake, &asset_id),
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
    publish(events, wake, Event::Busy("SCANNING COLLECTION"));
    let stop = Arc::clone(&state.stop);
    let Some(scan) = state
        .engine
        .scan_with_halt(root, || stop.load(Ordering::Acquire))?
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
    let Some((prompt, born)) = state.prompt.as_ref() else {
        bail!("there is no live comparison to judge");
    };
    let prompt = prompt.clone();
    let born = *born;
    publish(events, wake, Event::Busy("FORGING NEXT PAIR"));
    let winner = match side {
        Side::Left => &prompt.left.asset_id,
        Side::Right => &prompt.right.asset_id,
    };
    let response_ms = u32::try_from(born.elapsed().as_millis()).unwrap_or(u32::MAX);
    state
        .engine
        .record_comparison(&prompt.id, winner, &CommandId::fresh(), Some(response_ms))?;
    state.prompt = None;
    publish_collection(state, events, wake)
}

fn favorite(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    let prompt = live_prompt(state)?;
    let asset_id = match side {
        Side::Left => prompt.left.asset_id.clone(),
        Side::Right => prompt.right.asset_id.clone(),
    };
    let active = !state
        .cards
        .iter()
        .find(|card| card.asset_id == asset_id)
        .context("comparison asset left the active catalog")?
        .favorite;
    state.engine.set_favorite(
        active_session(state)?,
        &asset_id,
        active,
        &CommandId::fresh(),
    )?;
    if let Some(card) = state
        .cards
        .iter_mut()
        .find(|card| card.asset_id == asset_id)
    {
        card.favorite = active;
    }
    publish(events, wake, Event::Favorite { asset_id, active });
    Ok(())
}

fn hide(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    let collection = active_collection(state)?;
    let prompt = live_prompt(state)?;
    let asset = match side {
        Side::Left => &prompt.left.asset_id,
        Side::Right => &prompt.right.asset_id,
    };
    publish(events, wake, Event::Busy("WITHDRAWING IMAGE"));
    state.engine.set_hidden(collection, asset, true)?;
    state.prompt = None;
    publish_collection(state, events, wake)
}

fn rotate(
    state: &mut EngineState,
    events: &Sender<Event>,
    wake: &NativeWake,
    side: Side,
) -> Result<()> {
    let occurrence = match side {
        Side::Left => live_prompt(state)?.left.occurrence_id,
        Side::Right => live_prompt(state)?.right.occurrence_id,
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
) -> Result<()> {
    let card = state
        .cards
        .iter()
        .find(|card| &card.asset_id == asset_id)
        .context("thumbnail asset left the active catalog")?;
    let blade = decode(&card.path, card.rotation_quarters, THUMB_EDGE)?;
    publish(
        events,
        wake,
        Event::Thumbnail {
            asset_id: asset_id.clone(),
            blade,
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
    let snapshot = state.engine.rebuild_preferences(collection)?;
    let assets = state.engine.assets(collection)?;
    state.cards = assets.iter().map(Card::from).collect();
    let root = state.engine.collection(collection)?.root;
    publish(
        events,
        wake,
        Event::Catalog {
            summary: Summary {
                root,
                visible_assets: state.cards.len(),
                scan_failures: state.scan_failures,
                evaluation: snapshot.evaluation,
            },
            cards: state.cards.clone(),
        },
    );
    if state.cards.len() < 2 {
        state.prompt = None;
        publish(events, wake, Event::NoComparison);
        return Ok(());
    }
    let prompt = state.engine.propose_comparison(active_session(state)?)?;
    let left = card_for(state, &prompt.left.asset_id)?.clone();
    let right = card_for(state, &prompt.right.asset_id)?.clone();
    let left_blade = decode(&left.path, left.rotation_quarters, PROMPT_EDGE)?;
    let right_blade = decode(&right.path, right.rotation_quarters, PROMPT_EDGE)?;
    state.prompt = Some((prompt, Instant::now()));
    publish(
        events,
        wake,
        Event::Pair(Pair {
            left,
            right,
            left_blade,
            right_blade,
        }),
    );
    Ok(())
}

fn decode(path: &Path, rotation_quarters: u8, edge: u32) -> Result<Blade> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let image = canonical_image(&bytes).with_context(|| format!("decode {}", path.display()))?;
    let rgba = image.to_rgba8();
    let rotated = match rotation_quarters % 4 {
        1 => imageops::rotate90(&rgba),
        2 => imageops::rotate180(&rgba),
        3 => imageops::rotate270(&rgba),
        _ => rgba,
    };
    let image = DynamicImage::ImageRgba8(rotated).thumbnail(edge, edge);
    let rgba = image.to_rgba8();
    Ok(Blade {
        width: rgba.width() as usize,
        height: rgba.height() as usize,
        rgba: rgba.into_raw(),
    })
}

fn live_prompt(state: &EngineState) -> Result<&ComparisonPrompt> {
    state
        .prompt
        .as_ref()
        .map(|(prompt, _)| prompt)
        .context("there is no live comparison")
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
        .find(|card| &card.asset_id == asset)
        .context("prompt asset left the active catalog")
}

fn publish(events: &Sender<Event>, wake: &NativeWake, event: Event) {
    if events.send(event).is_ok() {
        let _woken = wake.request_repaint();
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
