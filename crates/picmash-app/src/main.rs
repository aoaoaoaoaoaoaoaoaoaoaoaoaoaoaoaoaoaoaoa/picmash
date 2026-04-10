#![allow(unused_crate_dependencies)]

use std::{net::SocketAddr, sync::Arc, time::Duration as StdDuration};

use anyhow::Context;
use clap::Parser;
use directories::ProjectDirs;
use picmash_app::{
    app::{AppState, RuntimeState, StartupSummary},
    config::AppConfig,
    telemetry, web,
};
use tokio::net::TcpListener;
use tracing::{Instrument, debug_span, error, info, info_span};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let boot_id = telemetry::fresh_boot_id();
    telemetry::install_boot_id(boot_id.clone())?;
    telemetry::init_subscriber();
    let dirs = ProjectDirs::from("moe", "swarm", "picmash")
        .context("resolving XDG directories for picmash")?;
    let (mut config, config_path, mut config_digest) = AppConfig::load_or_init(dirs.config_dir())?;
    let configured_root = config.corpus_root().map(ToOwned::to_owned);
    let requested_root = cli.root_path.or(configured_root).context(
        "no corpus root configured; pass IMAGE_ROOT once or set runtime.corpus_root in config.toml",
    )?;
    let root_path = requested_root
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", requested_root.display()))?;
    if config.corpus_root() != Some(root_path.as_path()) {
        config.shove_corpus_root(root_path.clone());
        config_digest = config.write(&config_path)?;
    }

    let addr: SocketAddr = config
        .bind_addr()
        .parse()
        .with_context(|| format!("parsing runtime.bind_addr `{}`", config.bind_addr()))?;
    let runtime = Arc::new(RuntimeState::loading());
    let app = web::router(runtime.clone());
    let boot_id_text = boot_id.0.clone();
    let boot_span = info_span!(
        "boot",
        boot_id = %boot_id_text,
        root = %root_path.display(),
        bind_addr = %addr,
    );
    let _boot_guard = boot_span.enter();

    info!(boot_id = %boot_id.0, root = %root_path.display(), "booting picmash");
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding server on {addr}"))?;
    info!(boot_id = %boot_id.0, url = %format!("http://{addr}"), "picmash bootstrap shell ready");

    let boot_runtime = runtime.clone();
    tokio::spawn(async move {
        let boot_root = root_path.clone();
        let boot_config = config.clone();
        let boot_config_path = config_path.clone();
        let booted = tokio::task::spawn_blocking(move || {
            AppState::boot(&boot_root, boot_config, boot_config_path, config_digest)
        })
        .await;
        match booted {
            Ok(Ok(state)) => {
                let state = Arc::new(state);
                spawn_config_reload_loop(state.clone());
                spawn_external_source_loop(state.clone());
                match state.startup_summary().context("summarizing startup state") {
                    Ok(StartupSummary {
                        corpus_id,
                        session_id,
                        visible_assets,
                        embedded_assets,
                    }) => {
                        info!(
                            boot_id = %boot_id.0,
                            root = %root_path.display(),
                            corpus_id = corpus_id.0,
                            session_id = session_id.0,
                            visible_assets,
                            embedded_assets,
                            "picmash state loaded"
                        );
                        boot_runtime.install_ready(state.clone());
                        spawn_background_maintenance_loop(state.clone());
                        state.schedule_corpus_ingest();
                        state.schedule_bootstrap_maintenance();
                        info!(boot_id = %boot_id.0, url = %format!("http://{addr}"), "picmash ready");
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        error!(boot_id = %boot_id.0, root = %root_path.display(), error = %message, "picmash boot summary failed");
                        boot_runtime.install_failed(message);
                    }
                }
            }
            Ok(Err(error)) => {
                let message = format!("{error:#}");
                error!(boot_id = %boot_id.0, root = %root_path.display(), error = %message, "picmash boot failed");
                boot_runtime.install_failed(message);
            }
            Err(error) => {
                let message = format!("joining picmash boot task: {error:#}");
                error!(boot_id = %boot_id.0, root = %root_path.display(), error = %message, "picmash boot task crashed");
                boot_runtime.install_failed(message);
            }
        }
    }
    .instrument(info_span!("boot.load_state", boot_id = %boot_id_text)));

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("running server")?;
    runtime.close_ready().context("closing active session")?;

    Ok(())
}

fn spawn_background_maintenance_loop(state: Arc<AppState>) {
    tokio::spawn(async move {
        let poll = u64::try_from(state.maintenance_idle_poll().whole_seconds().max(1)).unwrap_or(2);
        loop {
            let worker = state.clone();
            match tokio::task::spawn_blocking(move || worker.devour_one_maintenance_job())
                .instrument(debug_span!("maintenance.loop.tick"))
                .await
            {
                Ok(Ok(true)) => continue,
                Ok(Ok(false)) => {}
                Ok(Err(error)) => {
                    error!(error = %format!("{error:#}"), "background maintenance loop failed");
                }
                Err(error) => {
                    error!(error = %format!("{error:#}"), "background maintenance task crashed");
                }
            }
            tokio::select! {
                _ = state.wait_for_maintenance_signal() => {}
                _ = tokio::time::sleep(StdDuration::from_secs(poll)) => {}
            }
        }
    });
}

fn spawn_external_source_loop(state: Arc<AppState>) {
    tokio::spawn(async move {
        let initial = state.clone();
        match tokio::task::spawn_blocking(move || initial.refresh_external_sources_if_due(false))
            .instrument(debug_span!("external.refresh.loop.initial"))
            .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                error!(error = %format!("{error:#}"), "initial external refresh failed");
            }
            Err(error) => {
                error!(error = %format!("{error:#}"), "initial external refresh task crashed");
            }
        }
        let pulse = u64::try_from(state.external_scan_pulse().whole_seconds().max(5)).unwrap_or(20);
        loop {
            tokio::time::sleep(StdDuration::from_secs(pulse)).await;
            let state = state.clone();
            match tokio::task::spawn_blocking(move || state.refresh_external_sources_if_due(false))
                .instrument(debug_span!("external.refresh.loop.tick"))
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    error!(error = %format!("{error:#}"), "external refresh loop failed");
                }
                Err(error) => {
                    error!(error = %format!("{error:#}"), "external refresh task crashed");
                }
            }
        }
    });
}

fn spawn_config_reload_loop(state: Arc<AppState>) {
    tokio::spawn(async move {
        let pulse = u64::try_from(state.config_reload_pulse().whole_seconds().max(1)).unwrap_or(2);
        loop {
            tokio::time::sleep(StdDuration::from_secs(pulse)).await;
            let state = state.clone();
            match tokio::task::spawn_blocking(move || state.reload_config_if_changed())
                .instrument(debug_span!("config.reload.tick"))
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    error!(error = %format!("{error:#}"), "config reload loop failed");
                }
                Err(error) => {
                    error!(error = %format!("{error:#}"), "config reload task crashed");
                }
            }
        }
    });
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[derive(Debug, Parser)]
#[command(version, about = "Image-first pairwise ranking darkroom")]
struct Cli {
    #[arg(value_name = "IMAGE_ROOT")]
    root_path: Option<std::path::PathBuf>,
}
