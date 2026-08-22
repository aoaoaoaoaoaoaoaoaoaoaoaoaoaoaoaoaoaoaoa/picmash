use anyhow::{Context as _, Result, ensure};
use egui_tester::{
    AppCommand, Application, Button, Graphics, Key, Network, Probe, Testbed, WindowQuery,
};
use image::{ImageFormat, Rgba, RgbaImage};
use picmash as _;
use picmash_contract::{Side, Target, UI_FINGERPRINT};
use serde::Deserialize;
use std::{
    env,
    io::Cursor,
    path::{Path, PathBuf},
    time::Duration,
};

const WAIT: Duration = Duration::from_secs(8);
const STARTUP: Duration = Duration::from_secs(45);
const TITLE: &str = "picmash";
const WITNESS: &str = "probes/picmash.observations";

#[derive(Debug, Deserialize)]
struct Observation {
    contract: String,
    mode: String,
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

fn main() -> Result<()> {
    let binary = env::var_os("PICMASH_ACCEPTANCE_BINARY")
        .map(PathBuf::from)
        .context("PICMASH_ACCEPTANCE_BINARY must name the instrumented Picmash binary")?;
    let artifacts = env::args_os().nth(1).map(PathBuf::from);
    ensure!(
        binary.is_file(),
        "Picmash binary not found: {}",
        binary.display()
    );
    let testbed = Testbed::raise().context("raise hermetic X11 testbed")?;
    seed(&testbed)?;
    empty_session(&testbed, &binary, artifacts.as_deref())?;
    first_session(&testbed, &binary, artifacts.as_deref())?;
    restored_session(&testbed, &binary, artifacts.as_deref())?;
    println!("picmash acceptance passed under {}", testbed.id());
    Ok(())
}

fn empty_session(testbed: &Testbed, binary: &Path, artifacts: Option<&Path>) -> Result<()> {
    let app = launch(testbed, binary, false)?;
    let session = testbed.x11_session(
        &app,
        WindowQuery::title_exact(TITLE),
        Duration::from_secs(20),
    )?;
    session.focus()?;
    let mut probe: Probe<Observation> = app.witness()?.typed();
    let _presented = probe.wait_surface_presented(&app, STARTUP)?;
    let empty = probe.wait(&app, WAIT, "empty first contact", |frame| {
        !frame.state.busy && frame.state.collection.is_none() && !frame.state.pair_ready
    })?;
    ensure!(
        empty.state.visible_assets == 0 && empty.state.favorites == 0 && empty.state.duels == 0,
        "empty first contact projected collection state"
    );
    let target = Target::OpenCollection.to_string();
    let _open = probe.wait_anchor(&app, &target, WAIT)?;
    capture(&session, artifacts, "picmash-first-contact.png")?;
    app.terminate()?;
    Ok(())
}

fn first_session(testbed: &Testbed, binary: &Path, artifacts: Option<&Path>) -> Result<()> {
    let app = launch(testbed, binary, true)?;
    let session = testbed.x11_session(
        &app,
        WindowQuery::title_exact(TITLE),
        Duration::from_secs(20),
    )?;
    session.focus()?;
    let mut probe: Probe<Observation> = app.witness()?.typed();
    let _presented = probe.wait_surface_presented(&app, STARTUP)?;
    let ready = probe.wait(&app, STARTUP, "four-image comparison", |frame| {
        !frame.state.busy && frame.state.pair_ready && frame.state.visible_assets == 4
    })?;
    ensure!(
        ready.state.contract == UI_FINGERPRINT,
        "UI contract mismatch"
    );
    ensure!(
        ready.state.mode == "compare",
        "Picmash did not open in Compare"
    );
    ensure!(
        ready.state.collection.is_some(),
        "collection path was not projected"
    );
    ensure!(!ready.state.guide_open, "Help opened without an intent");
    ensure!(
        !ready.state.settings_open,
        "Settings opened without an intent"
    );
    ensure!(
        !ready.state.text_edit_focused,
        "a hidden editor stole keyboard focus"
    );
    capture(&session, artifacts, "picmash-compare.png")?;

    let _help_key = session.key(Key::Function(1))?;
    let _guide = probe.wait(&app, WAIT, "generated command guide", |frame| {
        frame.state.guide_open
    })?;
    let _escape = session.key(Key::Escape)?;
    let _guide_closed = probe.wait(&app, WAIT, "closed command guide", |frame| {
        !frame.state.guide_open
    })?;
    let _settings_key = session.key(Key::Function(2))?;
    let _settings = probe.wait(&app, WAIT, "application settings", |frame| {
        frame.state.settings_open
    })?;
    let _escape = session.key(Key::Escape)?;
    let _settings_closed = probe.wait(&app, WAIT, "closed settings", |frame| {
        !frame.state.settings_open
    })?;

    click(&session, &app, &mut probe, Target::Favorite(Side::Left))?;
    let _favorited = probe.wait(&app, WAIT, "favorite to persist", |frame| {
        frame.state.favorites == 1 && frame.state.pair_ready
    })?;

    click(&session, &app, &mut probe, Target::Rotate(Side::Left))?;
    let _rotated = probe.wait(&app, STARTUP, "rotated comparison", |frame| {
        !frame.state.busy
            && frame.state.pair_ready
            && frame
                .state
                .pair_rotations
                .is_some_and(|rotations| rotations.contains(&1))
    })?;

    click(&session, &app, &mut probe, Target::Choice(Side::Left))?;
    let _judged = probe.wait(&app, STARTUP, "recorded duel", |frame| {
        !frame.state.busy && frame.state.pair_ready && frame.state.duels == 1
    })?;

    click(&session, &app, &mut probe, Target::Hide(Side::Right))?;
    let _hidden = probe.wait(&app, STARTUP, "withdrawn image", |frame| {
        !frame.state.busy && frame.state.pair_ready && frame.state.visible_assets == 3
    })?;

    click(&session, &app, &mut probe, Target::BrowseMode)?;
    let browsed = probe.wait(&app, WAIT, "collection browser", |frame| {
        frame.state.mode == "browse" && !frame.state.busy
    })?;
    ensure!(
        !browsed.state.status.starts_with("FAULT"),
        "browser opened in fault: {}",
        browsed.state.status
    );
    let browser_target = Target::Browser.to_string();
    let _browser = probe.wait_anchor(&app, &browser_target, WAIT)?;
    for _frame in 0..4 {
        let _fresh = probe.wait_fresh(&app, WAIT)?;
    }
    let tile_target = Target::BrowseTile.to_string();
    let tile = probe.wait_anchor(&app, &tile_target, WAIT)?;
    let (x, y) = tile.center();
    let _hovered = session.move_to(x, y)?;
    for _frame in 0..4 {
        let _fresh = probe.wait_fresh(&app, WAIT)?;
    }
    capture(&session, artifacts, "picmash-browse.png")?;
    app.terminate()?;
    Ok(())
}

fn restored_session(testbed: &Testbed, binary: &Path, artifacts: Option<&Path>) -> Result<()> {
    let app = launch(testbed, binary, false)?;
    let session = testbed.x11_session(
        &app,
        WindowQuery::title_exact(TITLE),
        Duration::from_secs(20),
    )?;
    session.focus()?;
    let mut probe: Probe<Observation> = app.witness()?.typed();
    let _presented = probe.wait_surface_presented(&app, STARTUP)?;
    let restored = probe.wait(&app, STARTUP, "restored collection and evidence", |frame| {
        !frame.state.busy
            && frame.state.pair_ready
            && frame.state.visible_assets == 3
            && frame.state.favorites == 1
            && frame.state.duels == 1
    })?;
    ensure!(
        restored.state.contract == UI_FINGERPRINT,
        "restored UI contract mismatch"
    );
    capture(&session, artifacts, "picmash-restored.png")?;
    app.terminate()?;
    Ok(())
}

fn launch<'a>(
    testbed: &'a Testbed,
    binary: &Path,
    with_collection: bool,
) -> Result<Application<'a>> {
    let mut command = AppCommand::new(binary)
        .witness(WITNESS)
        .graphics(Graphics::Software)
        .network(Network::Deny)
        .runtime(Duration::from_secs(90));
    if with_collection {
        command = command.arg(Testbed::guest_path("collection"));
    }
    testbed.launch(command).map_err(anyhow::Error::from)
}

fn click(
    session: &egui_tester::X11Session<'_, '_>,
    app: &Application<'_>,
    probe: &mut Probe<Observation>,
    target: Target,
) -> Result<()> {
    let target = target.to_string();
    let anchor = probe.wait_anchor(app, &target, WAIT)?;
    let (x, y) = anchor.center();
    let _clicked = session.click(x, y, Button::Primary)?;
    Ok(())
}

fn capture(
    session: &egui_tester::X11Session<'_, '_>,
    artifacts: Option<&Path>,
    name: &str,
) -> Result<()> {
    if let Some(artifacts) = artifacts {
        std::fs::create_dir_all(artifacts)
            .with_context(|| format!("create {}", artifacts.display()))?;
        session.capture()?.save_png(artifacts.join(name))?;
    }
    Ok(())
}

fn seed(testbed: &Testbed) -> Result<()> {
    let _collection = testbed.create_private_dir("collection")?;
    for (index, dimensions) in [(720, 960), (1_100, 680), (820, 820), (640, 1_040)]
        .into_iter()
        .enumerate()
    {
        let bytes = fixture(index as u8, dimensions)?;
        let _written = testbed.write_private(format!("collection/{index}.png"), &bytes)?;
        let _copy = testbed.write_private(format!("collection/{index}-copy.png"), &bytes)?;
    }
    Ok(())
}

fn fixture(seed: u8, (width, height): (u32, u32)) -> Result<Vec<u8>> {
    let mut image = RgbaImage::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let band = ((x / 48 + y / 52 + u32::from(seed)) % 4) as u8;
        let radial = (((x.abs_diff(width / 2) + y.abs_diff(height / 2)) / 7) % 180) as u8;
        *pixel = Rgba([
            36_u8
                .saturating_add(seed.saturating_mul(41))
                .saturating_add(radial / 3),
            24_u8.saturating_add(band.saturating_mul(34)),
            18_u8.saturating_add((3 - band).saturating_mul(29)),
            255,
        ]);
    }
    let mut bytes = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)?;
    Ok(bytes)
}
