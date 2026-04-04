use std::{
    path::{Path as FsPath, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use axum::{
    Form, Json, Router,
    extract::{Path, Query, State},
    http::{
        HeaderMap, StatusCode,
        header::{CACHE_CONTROL, CONTENT_TYPE, HeaderValue, REFERER},
    },
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
};
use image::{
    ColorType, DynamicImage, ImageEncoder,
    codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder},
    imageops::{self, FilterType},
};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use serde::Deserialize;
use tokio::fs;
use tracing::{info, warn};

use crate::{
    api::{
        ExploreBootstrapDto, ExploreNeighborDto, ExplorePointDto, ExploreSelectionDto,
        ExploreSelectionResponseDto, FocusAssetDto, HeartAssetRequestDto, HideAssetRequestDto,
        RotateAssetRequestDto, TriadAssetDto, TriadBootstrapDto, TriadHandleDto,
        TriadTrainRequestDto,
    },
    app::{
        FacemashFaceView, FacemashLocalAssetView, FacemashPairView, FacemashStatus,
        IdentityReviewStatus, IdentityReviewView,
    },
    app::{RedirectTarget, RuntimePhase, RuntimeSnapshot, SharedAppState, SharedRuntimeState},
    asset_domain::AssetDomainLabel,
    model::{
        ArenaCard, ArenaHandle, ArenaView, AssetDomainView, AssetId, AssetQualitySummary,
        AssetRecord, BoardEntry, DuplicateCluster, ExploreMapMode, ExploreSelection, ExploreView,
        ExternalArenaStatus, FaceId, PosteriorSummary, RemoteItemId, RemoteItemRecord,
    },
    store::FaceRecord,
};

mod arena;
mod board;
mod explore;
mod facemash;
mod identities;
mod media;
mod vocab;

use self::{
    arena::{
        api_arena_next, arena, arena_reroll, arena_root, heart, hide, lock_thread, rotate,
        veto_thread, vote,
    },
    board::{
        asset_domain, asset_heart, asset_hide, asset_rotate, board, board_nudge, rescan,
        set_arena_explore, set_dedup_radius, set_external_probability, set_facemash_min_face_side,
    },
    explore::{
        api_explore_bootstrap, api_explore_domain, api_explore_heart, api_explore_hide,
        api_explore_rotate, api_explore_selection, api_triad_bootstrap, api_triad_domain,
        api_triad_heart, api_triad_hide, api_triad_rotate, api_triad_train, explore_root,
        triad_reroll, triad_root,
    },
    facemash::{
        face_image, facemash_hide_face, facemash_pair, facemash_reroll, facemash_root,
        facemash_status, facemash_vote,
    },
    identities::{
        identities_confirm, identities_focus, identities_name, identities_root,
        identities_set_threshold, identities_veto,
    },
    media::{AssetRendition, asset_image, favicon_asset, font_asset, remote_image},
    vocab::{vocab_lab_reroll, vocab_lab_root},
};

pub fn router(state: SharedRuntimeState) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/favicon.svg", get(favicon_asset))
        .route("/__status", get(runtime_status))
        .route("/__frontend/picmash-client.js", get(frontend_javascript))
        .route("/__frontend/picmash-client.css", get(frontend_stylesheet))
        .route("/arena", get(arena_root))
        .route("/arena/reroll", get(arena_reroll))
        .route("/arena/{left_id}/{right_id}", get(arena))
        .route("/arena/{left_id}/{right_id}/vote", post(vote))
        .route("/api/arena/next", get(api_arena_next))
        .route("/arena/{left_id}/{right_id}/rotate", post(rotate))
        .route("/arena/{left_id}/{right_id}/heart", post(heart))
        .route("/arena/{left_id}/{right_id}/hide", post(hide))
        .route("/arena/{left_id}/{right_id}/lock-thread", post(lock_thread))
        .route("/arena/{left_id}/{right_id}/veto-thread", post(veto_thread))
        .route("/board", get(board))
        .route("/board/nudge", post(board_nudge))
        .route("/asset/heart", post(asset_heart))
        .route("/asset/rotate", post(asset_rotate))
        .route("/asset/hide", post(asset_hide))
        .route("/asset/domain", post(asset_domain))
        .route("/rescan", post(rescan))
        .route("/external/probability", post(set_external_probability))
        .route("/arena/explore", post(set_arena_explore))
        .route("/external/dedup-radius", post(set_dedup_radius))
        .route("/explore", get(explore_root))
        .route("/triad", get(triad_root))
        .route("/triad/reroll", get(triad_reroll))
        .route("/vocab-lab", get(vocab_lab_root))
        .route("/vocab-lab/reroll", get(vocab_lab_reroll))
        .route("/api/explore/bootstrap", get(api_explore_bootstrap))
        .route(
            "/api/explore/selection/{asset_id}",
            get(api_explore_selection),
        )
        .route("/api/explore/rotate", post(api_explore_rotate))
        .route("/api/explore/heart", post(api_explore_heart))
        .route("/api/explore/domain", post(api_explore_domain))
        .route("/api/explore/hide", post(api_explore_hide))
        .route("/api/triad/bootstrap", get(api_triad_bootstrap))
        .route("/api/triad/train", post(api_triad_train))
        .route("/api/triad/rotate", post(api_triad_rotate))
        .route("/api/triad/heart", post(api_triad_heart))
        .route("/api/triad/domain", post(api_triad_domain))
        .route("/api/triad/hide", post(api_triad_hide))
        .route("/assets/{asset_id}", get(asset_image))
        .route("/remote/{item_id}", get(remote_image))
        .route("/faces/{face_id}", get(face_image))
        .route("/facemash", get(facemash_root))
        .route("/facemash/reroll", get(facemash_reroll))
        .route(
            "/facemash/{left_face_id}/{right_face_id}",
            get(facemash_pair),
        )
        .route("/facemash/min-face-side", post(set_facemash_min_face_side))
        .route("/facemash/vote", post(facemash_vote))
        .route("/facemash/hide", post(facemash_hide_face))
        .route("/facemash/status", get(facemash_status))
        .route("/identities", get(identities_root))
        .route("/identities/{subject_slug}", get(identities_focus))
        .route(
            "/identities/match-threshold",
            post(identities_set_threshold),
        )
        .route("/identities/name", post(identities_name))
        .route("/identities/confirm", post(identities_confirm))
        .route("/identities/veto", post(identities_veto))
        .route("/__swarm/fonts/{font_name}", get(font_asset))
        .with_state(state)
}

static SITE_LOAD_LOGGED: AtomicBool = AtomicBool::new(false);
const FAVICON_SVG: &str = include_str!("../assets/favicon.svg");
const FRONTEND_JS: &str = include_str!("../assets/web/picmash-client.js");
const FRONTEND_CSS: &str = concat!(
    include_str!("../assets/picmash.css"),
    "\n",
    include_str!("../assets/web/picmash-client.css"),
);

fn log_site_loaded(route: &'static str) {
    if SITE_LOAD_LOGGED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        info!(route, "site loaded and ready");
    }
}

async fn runtime_status(State(state): State<SharedRuntimeState>) -> Response {
    no_store_response(Json(state.snapshot()).into_response())
}

async fn frontend_javascript() -> Response {
    cached_text_response(
        FRONTEND_JS,
        HeaderValue::from_static("text/javascript; charset=utf-8"),
    )
}

async fn frontend_stylesheet() -> Response {
    cached_text_response(
        FRONTEND_CSS,
        HeaderValue::from_static("text/css; charset=utf-8"),
    )
}

fn ready_app_or_snapshot(state: &SharedRuntimeState) -> Result<SharedAppState, RuntimeSnapshot> {
    state.ready_app().ok_or_else(|| state.snapshot())
}

fn boot_response(snapshot: RuntimeSnapshot) -> Response {
    let status = match snapshot.phase {
        RuntimePhase::Loading | RuntimePhase::Ready => StatusCode::OK,
        RuntimePhase::Failed => StatusCode::SERVICE_UNAVAILABLE,
    };
    no_store_response(
        (
            status,
            Html(layout("loading-page", boot_markup(&snapshot)).into_string()),
        )
            .into_response(),
    )
}

fn boot_markup(snapshot: &RuntimeSnapshot) -> Markup {
    let status_label = match snapshot.phase {
        RuntimePhase::Loading => "warming the machine",
        RuntimePhase::Ready => "ready",
        RuntimePhase::Failed => "boot fault",
    };
    let detail = match snapshot.phase {
        RuntimePhase::Loading => {
            "ingesting the corpus, extracting embeddings, and forging the live state."
        }
        RuntimePhase::Ready => "the app is ready; this page should reload immediately.",
        RuntimePhase::Failed => snapshot.message.as_deref().unwrap_or("picmash boot failed"),
    };
    html! {
        section.loading-shell {
            div.loading-panel.swarm-surface-elevated {
                div.loading-kicker { "picmash" }
                h1.loading-title { (status_label) }
                p.loading-detail data-loading-message="" { (detail) }
                div.loading-progress aria-hidden="true" {
                    span.loading-bar {}
                }
            }
        }
    }
}

fn service_unavailable_response() -> Response {
    no_store_response((StatusCode::SERVICE_UNAVAILABLE, "picmash is still booting").into_response())
}

async fn home(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/");
    let href = state.home_target()?.href();
    Ok(Redirect::to(&href).into_response())
}

fn stale_arena_action_response<const N: usize>(
    state: &SharedAppState,
    handles: [&ArenaHandle; N],
) -> anyhow::Result<Option<Response>> {
    for handle in handles {
        if !state.arena_handle_is_live(handle)? {
            return Ok(Some(
                Redirect::to(&state.arena_target()?.href()).into_response(),
            ));
        }
    }
    Ok(None)
}

fn stale_facemash_action_response(
    state: &SharedAppState,
    left_face_id: FaceId,
    right_face_id: FaceId,
) -> anyhow::Result<Option<Response>> {
    if !state.facemash_pair_is_live(left_face_id, right_face_id)? {
        return Ok(Some(Redirect::to("/facemash").into_response()));
    }
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
enum ImageToolKind {
    RotateLeft,
    RotateRight,
    Heart,
    Hide,
    LockThread,
    VetoThread,
    Less,
    More,
    Close,
}

impl ImageToolKind {
    const fn label(self) -> &'static str {
        match self {
            Self::RotateLeft => "L",
            Self::RotateRight => "R",
            Self::Heart => "♥",
            Self::Hide | Self::Close => "X",
            Self::LockThread => "🔒",
            Self::VetoThread => "TX",
            Self::Less => "-",
            Self::More => "+",
        }
    }

    const fn title(self) -> &'static str {
        match self {
            Self::RotateLeft => "rotate left",
            Self::RotateRight => "rotate right",
            Self::Heart => "bless image",
            Self::Hide => "hide image",
            Self::LockThread => "lock to this sub-source",
            Self::VetoThread => "veto this thread",
            Self::Close => "close preview",
            Self::Less => "show less like this this session",
            Self::More => "show more like this this session",
        }
    }

    const fn class(self, mini: bool) -> &'static str {
        match (self, mini) {
            (Self::RotateLeft | Self::RotateRight, true) => "tool mini rotate-tool",
            (Self::RotateLeft | Self::RotateRight, false) => "tool rotate-tool",
            (Self::Heart, true) => "tool mini heart",
            (Self::Heart, false) => "tool heart",
            (Self::Hide | Self::VetoThread, true) => "tool mini danger",
            (Self::Hide | Self::VetoThread, false) => "tool danger",
            (Self::LockThread, true) => "tool mini",
            (Self::LockThread, false) => "tool",
            (_, true) => "tool mini",
            (_, false) => "tool",
        }
    }
}

fn tool_button(kind: ImageToolKind, mini: bool, active: bool) -> Markup {
    let class = if active {
        format!("{} active", kind.class(mini))
    } else {
        kind.class(mini).to_owned()
    };
    html! {
        button
            class=(class)
            type="submit"
            title=(kind.title())
            aria-label=(kind.title()) {
            (kind.label())
        }
    }
}

fn rail(active: NavPage, mode_menu: Option<Markup>) -> Markup {
    html! {
        header.rail {
            (global_menu(active))
            (ops_menu())
            @if let Some(reroll_href) = reroll_href(active) {
                a.rail-reroll.swarm-frame-header href=(reroll_href) { "reroll" }
            }
            @if let Some(mode_menu) = mode_menu {
                (mode_menu)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PageGeometry {
    Bare,
    Document,
    Viewport,
}

impl PageGeometry {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Bare => "bare",
            Self::Document => "document",
            Self::Viewport => "viewport",
        }
    }
}

fn reroll_href(active: NavPage) -> Option<&'static str> {
    match active {
        NavPage::Arena => Some("/arena/reroll"),
        NavPage::Facemash => Some("/facemash/reroll"),
        NavPage::Triad => Some("/triad/reroll"),
        NavPage::Vocab => Some("/vocab-lab/reroll"),
        NavPage::Board | NavPage::Explore | NavPage::Identities => None,
    }
}

fn global_menu(active: NavPage) -> Markup {
    html! {
        details.rail-menu {
            summary.swarm-frame-header { "menu" }
            nav.menu-panel.swarm-frame {
                a href="/arena" class=(nav_class(matches!(active, NavPage::Arena))) { "arena" }
                a href="/board" class=(nav_class(matches!(active, NavPage::Board))) { "board" }
                a href="/facemash" class=(nav_class(matches!(active, NavPage::Facemash))) { "facemash" }
                a href="/identities" class=(nav_class(matches!(active, NavPage::Identities))) { "identities" }
                a href="/explore?mode=raw" class=(nav_class(matches!(active, NavPage::Explore))) { "explore" }
                a href="/triad" class=(nav_class(matches!(active, NavPage::Triad))) { "triad" }
                a href="/vocab-lab" class=(nav_class(matches!(active, NavPage::Vocab))) { "vocab" }
            }
        }
    }
}

fn ops_menu() -> Markup {
    html! {
        details.rail-menu {
            summary.swarm-frame-header { "ops" }
            div.menu-panel.swarm-frame {
                form action="/rescan" method="post" {
                    button type="submit" class="menu-action" { "rescan" }
                }
            }
        }
    }
}

fn nav_class(active: bool) -> &'static str {
    if active { "active" } else { "" }
}

fn referer_redirect_target(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(REFERER)?.to_str().ok()?;
    if raw.starts_with('/') {
        return Some(raw.to_owned());
    }

    let (_, rest) = raw.split_once("://")?;
    let slash = rest.find('/')?;
    Some(rest[slash..].to_owned())
}

fn asset_src(asset: &AssetRecord, rendition: AssetRendition) -> String {
    format!(
        "/assets/{}?r={}&kind={}",
        asset.id.0,
        asset.rotation_quarters,
        rendition.as_str()
    )
}

fn remote_src(item: &RemoteItemRecord, rendition: AssetRendition) -> String {
    format!(
        "/remote/{}?r={}&kind={}",
        item.id.0,
        item.rotation_quarters,
        rendition.as_str()
    )
}

fn board_tooltip(rank: usize, entry: &BoardEntry) -> String {
    let file_name = entry
        .asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let mut lines = vec![
        format!("#{rank}"),
        file_name.to_owned(),
        format!("global {:+.2}", entry.global_score),
        format!("session {:+.2}", entry.session_utility),
        format!("focus {:+.2}", entry.session_focus),
        format!("pull {:.2}", entry.sampling_pull),
        format!(
            "{} wins / {} duels",
            entry.asset.win_count, entry.asset.compare_count
        ),
        format!("certainty {:.0}%", entry.certainty * 100.0),
    ];
    if entry.residual_score.abs() >= 0.02 {
        lines.push(format!("residual {:+.2}", entry.residual_score));
    }
    if entry.session_offset.abs() >= 0.02 {
        lines.push(format!("exact offset {:+.2}", entry.session_offset));
    }
    lines.extend(asset_quality_lines(entry.quality));
    lines.join("\n")
}

fn posterior_brief(summary: PosteriorSummary) -> String {
    format_posterior(summary.mean, summary.sigma)
}

fn format_posterior(mean: f32, sigma: f32) -> String {
    format!("{} ± {}", format_scalar(mean), format_scalar(sigma))
}

fn format_scalar(value: f32) -> String {
    let abs = value.abs();
    if abs >= 100.0 {
        format!("{value:.0}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

fn asset_quality_lines(summary: AssetQualitySummary) -> Vec<String> {
    let mut lines = vec![
        format!("asset {}", posterior_brief(summary.asset)),
        format!("base {}", posterior_brief(summary.baseline)),
    ];
    if let Some(semantic) = summary.semantic {
        lines.push(format!("semantic {}", posterior_brief(semantic)));
    }
    if let Some(vibe) = summary.vibe {
        lines.push(format!("vibe {}", posterior_brief(vibe)));
    }
    if let Some(technical) = summary.technical {
        lines.push(format!("k {}", posterior_brief(technical)));
    }
    if let Some(face) = summary.face {
        lines.push(format!("face {}", posterior_brief(face)));
    }
    lines
}

fn quality_chip(label: &str, summary: PosteriorSummary, class: &str) -> Markup {
    html! {
        span class=(format!("meta-chip quality-chip {class}")) {
            (label) " " (posterior_brief(summary))
        }
    }
}

fn asset_quality_chips(summary: AssetQualitySummary) -> Markup {
    html! {
        div.quality-chip-row {
            (quality_chip("q", summary.asset, "quality-chip-asset"))
            (quality_chip("b", summary.baseline, "quality-chip-baseline"))
            @if let Some(semantic) = summary.semantic {
                (quality_chip("m", semantic, "quality-chip-semantic"))
            }
            @if let Some(vibe) = summary.vibe {
                (quality_chip("s", vibe, "quality-chip-vibe"))
            }
            @if let Some(technical) = summary.technical {
                (quality_chip("k", technical, "quality-chip-technical"))
            }
            @if let Some(face) = summary.face {
                (quality_chip("f", face, "quality-chip-face"))
            }
        }
    }
}

fn asset_domain_button_classes(
    label: AssetDomainLabel,
    domain: AssetDomainView,
    mini: bool,
) -> String {
    let mut classes = if mini {
        "tool mini domain-tool".to_owned()
    } else {
        "tool domain-tool".to_owned()
    };
    classes.push_str(match label {
        AssetDomainLabel::Real => " domain-tool-3d",
        AssetDomainLabel::Anime => " domain-tool-2d",
    });
    if domain
        .predicted
        .is_some_and(|prediction| prediction.label() == label)
    {
        classes.push_str(" predicted");
    }
    if domain.manual == Some(label) {
        classes.push_str(" manual active");
    }
    classes
}

fn asset_domain_button_title(label: AssetDomainLabel, domain: AssetDomainView) -> String {
    let mut cues = Vec::new();
    if domain.manual == Some(label) {
        cues.push(format!("manual {}", label.display_str()));
    }
    if let Some(prediction) = domain
        .predicted
        .filter(|prediction| prediction.label() == label)
    {
        cues.push(format!(
            "model {} {}%",
            label.display_str(),
            prediction.display_percent()
        ));
    }
    if cues.is_empty() {
        label.title().to_owned()
    } else {
        format!("{} · {}", label.title(), cues.join(" · "))
    }
}

fn asset_domain_controls(
    asset_id: &AssetId,
    domain: AssetDomainView,
    mini: bool,
    facemash_aux: bool,
) -> Markup {
    let render_forms = |aux: bool| {
        html! {
            @for label in [AssetDomainLabel::Real, AssetDomainLabel::Anime] {
                @if aux {
                    form action="/asset/domain" method="post" data-facemash-aux="" {
                        input type="hidden" name="asset_id" value=(asset_id.0);
                        input type="hidden" name="label" value=(label.as_str());
                        button
                            class=(asset_domain_button_classes(label, domain, mini))
                            type="submit"
                            title=(asset_domain_button_title(label, domain))
                            aria-label=(asset_domain_button_title(label, domain)) {
                            (label.display_str())
                        }
                    }
                } @else {
                    form action="/asset/domain" method="post" {
                        input type="hidden" name="asset_id" value=(asset_id.0);
                        input type="hidden" name="label" value=(label.as_str());
                        button
                            class=(asset_domain_button_classes(label, domain, mini))
                            type="submit"
                            title=(asset_domain_button_title(label, domain))
                            aria-label=(asset_domain_button_title(label, domain)) {
                            (label.display_str())
                        }
                    }
                }
            }
        }
    };
    html! {
        @if facemash_aux {
            div.asset-domain-controls data-facemash-aux="" {
                (render_forms(true))
            }
        } @else {
            div.asset-domain-controls {
                (render_forms(false))
            }
        }
    }
}

fn selection_name(asset: &AssetRecord) -> &str {
    asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
}

fn frontend_host_markup(
    active: NavPage,
    page_class: &'static str,
    frontend_page: &'static str,
) -> Markup {
    shell_layout(
        page_class,
        PageGeometry::Viewport,
        Some(active),
        None,
        html! {
            script type="module" src="/__frontend/picmash-client.js" {}
        },
        html! {
            section.frontend-host-shell {
                div id="picmash-app" data-picmash-client-page=(frontend_page) {}
                noscript {
                    p.frontend-noscript { "picmash frontend requires javascript." }
                }
            }
        },
    )
}

fn layout(page_class: &'static str, body: Markup) -> Markup {
    bare_layout(page_class, body)
}

fn routed_layout(
    page_class: &'static str,
    geometry: PageGeometry,
    active: NavPage,
    mode_menu: Option<Markup>,
    body: Markup,
) -> Markup {
    shell_layout(
        page_class,
        geometry,
        Some(active),
        mode_menu,
        html! {},
        body,
    )
}

fn bare_layout(page_class: &'static str, body: Markup) -> Markup {
    bare_layout_with_head(page_class, html! {}, body)
}

fn bare_layout_with_head(page_class: &'static str, head: Markup, body: Markup) -> Markup {
    shell_layout(page_class, PageGeometry::Bare, None, None, head, body)
}

fn shell_layout(
    page_class: &'static str,
    geometry: PageGeometry,
    active: Option<NavPage>,
    mode_menu: Option<Markup>,
    head: Markup,
    body: Markup,
) -> Markup {
    let geometry_class = format!("app-shell app-shell--{}", geometry.as_str());
    let body_class = format!("page-body page-body--{}", geometry.as_str());
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                meta name="description" content="such pretty pictures ~ doki doki";
                link rel="icon" type="image/svg+xml" href="/favicon.svg";
                link rel="stylesheet" href="/__frontend/picmash-client.css";
                title { "Picmash" }
                (head)
            }
            body class=(page_class) data-page-geometry=(geometry.as_str()) {
                div class=(geometry_class) {
                    @if let Some(active) = active {
                        (rail(active, mode_menu))
                    }
                    main class=(body_class) {
                        (body)
                    }
                }
                (reconnect_overlay())
                (shared_runtime_script_block())
            }
        }
    }
}

fn reconnect_overlay() -> Markup {
    html! {
        section.reconnect-overlay hidden="" aria-hidden="true" data-reconnect-overlay="" {
            div.reconnect-panel.swarm-surface-elevated {
                div.reconnect-kicker { "picmash" }
                p.reconnect-text data-reconnect-message="" { "reconnecting" }
            }
        }
    }
}

fn shared_runtime_script_block() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"
                (() => {
                  const overlay = document.querySelector("[data-reconnect-overlay]");
                  const overlayMessage = overlay?.querySelector("[data-reconnect-message]");
                  const loadingPage = document.body.classList.contains("loading-page");
                  const loadingMessage = document.querySelector("[data-loading-message]")?.textContent?.trim() || "warming the machine.";
                  let shouldReload = loadingPage;
                  let pollDelay = loadingPage ? 900 : 2600;

                  const setOverlay = (visible, message) => {
                    if (!overlay || !overlayMessage || loadingPage) return;
                    overlay.hidden = !visible;
                    overlay.setAttribute("aria-hidden", visible ? "false" : "true");
                    overlayMessage.textContent = message;
                    document.body.classList.toggle("reconnect-open", visible);
                  };

                  const pollRuntime = async () => {
                    try {
                      const response = await fetch("/__status", {
                        method: "GET",
                        credentials: "same-origin",
                        cache: "no-store",
                      });
                      if (!response.ok) throw new Error(`runtime status ${response.status}`);
                      const status = await response.json();
                      const phase = String(status.phase || "").toLowerCase();
                      if (phase === "ready") {
                        if (shouldReload) {
                          window.location.reload();
                          return;
                        }
                        setOverlay(false, "");
                        pollDelay = 2600;
                      } else {
                        shouldReload = true;
                        const message = phase === "failed"
                          ? (status.message || "picmash boot failed")
                          : "reconnecting";
                        setOverlay(true, message);
                        pollDelay = 900;
                      }
                    } catch (_error) {
                      shouldReload = true;
                      setOverlay(true, "reconnecting");
                      pollDelay = 900;
                    } finally {
                      window.setTimeout(pollRuntime, pollDelay);
                    }
                  };

                  if (loadingPage) {
                    const detail = document.querySelector("[data-loading-message]");
                    if (detail && !detail.textContent?.trim()) {
                      detail.textContent = loadingMessage;
                    }
                  } else {
                    window.setTimeout(pollRuntime, pollDelay);
                  }

                  if (loadingPage) {
                    window.setTimeout(pollRuntime, 120);
                  }
                })();
                "#,
            ))
        }
    }
}

fn script_block() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"
                (() => {
                  const arenaShell = document.querySelector(".arena-stage-shell");
                  if (!arenaShell) return;

                  const TRANSITION_MS = 300;
                  const normalizeTurns = (value) => ((Number(value) || 0) % 4 + 4) % 4;
                  let currentLayer = arenaShell.querySelector(".arena-stage-layer.is-current");
                  let lookaheadLayer = arenaShell.querySelector(".arena-stage-layer.is-lookahead");
                  let preservedLookaheadLayer = arenaShell.querySelector(".arena-stage-layer.is-preserved-lookahead");
                  let lookaheadReady = false;
                  let preservedLookaheadReady = false;
                  let preparedEpoch = 0;
                  let queuedLookaheadPayload = null;
                  let queuedPreservedPayload = null;
                  let queuedLookaheadRequest = null;
                  let queuedPreservedRequest = null;
                  let isTransitioning = false;
                  let isActionPending = false;

                  const layerStage = (layer) => layer?.querySelector(".arena-stage") ?? null;
                  const layerImages = (layer) =>
                    Array.from(layer?.querySelectorAll(".vote-surface .arena-image") ?? []);
                  const eventElement = (event) =>
                    event.target instanceof Element ? event.target : event.target?.parentElement ?? null;
                  const currentVoteForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".vote-form") ?? []);
                  const currentHideForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".arena-hide-form") ?? []);
                  const currentVetoThreadForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".arena-veto-thread-form") ?? []);
                  const arenaAdvancePolicy = (form) => form.dataset.arenaAdvance || "";

                  const arenaGeometry = (image) => {
                    if (!image.naturalWidth || !image.naturalHeight) return null;
                    const turns = normalizeTurns(image.dataset.rotation);
                    const rotated = turns % 2 !== 0;
                    const naturalWidth = image.naturalWidth;
                    const naturalHeight = image.naturalHeight;
                    const effectiveWidth = rotated ? naturalHeight : naturalWidth;
                    const effectiveHeight = rotated ? naturalWidth : naturalHeight;
                    return {
                      turns,
                      naturalWidth,
                      naturalHeight,
                      effectiveWidth,
                      effectiveHeight,
                      aspect: effectiveWidth / effectiveHeight,
                    };
                  };

                  const applyArenaSplit = (layer) => {
                    const stage = layerStage(layer);
                    const images = layerImages(layer);
                    if (!stage || images.length !== 2) return;
                    const left = arenaGeometry(images[0]);
                    const right = arenaGeometry(images[1]);
                    if (!left || !right) {
                      stage.style.gridTemplateColumns = "";
                      return;
                    }
                    const stageStyle = getComputedStyle(stage);
                    const gap = Number.parseFloat(stageStyle.columnGap || stageStyle.gap || "0") || 0;
                    const availableWidth = stage.clientWidth - gap;
                    const availableHeight = stage.clientHeight;
                    if (availableWidth <= 0 || availableHeight <= 0) return;
                    const leftFullWidth = left.aspect * availableHeight;
                    const rightFullWidth = right.aspect * availableHeight;
                    const totalFullWidth = leftFullWidth + rightFullWidth;
                    let leftWidth;
                    if (availableWidth >= totalFullWidth) {
                      const slack = (availableWidth - totalFullWidth) / 2;
                      leftWidth = leftFullWidth + slack;
                    } else {
                      leftWidth = availableWidth * (left.aspect / (left.aspect + right.aspect));
                    }
                    const minWidth = 1;
                    leftWidth = Math.max(minWidth, Math.min(availableWidth - minWidth, leftWidth));
                    const rightWidth = Math.max(minWidth, availableWidth - leftWidth);
                    stage.style.gridTemplateColumns = `${leftWidth}px ${rightWidth}px`;
                  };

                  const fitArenaImage = (image) => {
                    const surface = image.closest(".vote-surface");
                    const geometry = arenaGeometry(image);
                    if (!surface || !geometry) return;
                    const boundWidth = surface.clientWidth;
                    const boundHeight = surface.clientHeight;
                    if (!boundWidth || !boundHeight) return;
                    const scale = Math.min(
                      boundWidth / geometry.effectiveWidth,
                      boundHeight / geometry.effectiveHeight,
                    );
                    image.style.width = `${geometry.naturalWidth * scale}px`;
                    image.style.height = `${geometry.naturalHeight * scale}px`;
                    image.style.transform = `translate(-50%, -50%) rotate(${geometry.turns * 90}deg)`;
                  };

                  const revealLayer = (layer) => {
                    const images = layerImages(layer);
                    if (!images.length || !images.every((image) => image.dataset.settled === "1")) return;
                    applyArenaSplit(layer);
                    for (const image of images) {
                      fitArenaImage(image);
                      image.classList.add("is-ready");
                    }
                  };

                  const primeLayer = (layer) => {
                    const images = layerImages(layer);
                    if (!layer || !images.length) return Promise.resolve();
                    const pending = [];
                    for (const image of images) {
                      if (
                        image.dataset.settled === "1" &&
                        image.classList.contains("is-ready") &&
                        image.complete &&
                        image.naturalWidth > 0
                      ) {
                        continue;
                      }
                      image.dataset.settled = "";
                      image.classList.remove("is-ready");
                      pending.push(image);
                    }
                    if (!pending.length) {
                      revealLayer(layer);
                      return Promise.resolve();
                    }
                    return new Promise((resolve) => {
                      let remaining = pending.length;
                      const onSettle = (image) => {
                        if (image.dataset.settled === "1") return;
                        image.dataset.settled = "1";
                        remaining -= 1;
                        revealLayer(layer);
                        if (remaining <= 0) resolve();
                      };
                      for (const image of pending) {
                        if (image.complete && image.naturalWidth > 0) {
                          onSettle(image);
                          continue;
                        }
                        image.addEventListener("load", () => onSettle(image), { once: true });
                        image.addEventListener("error", () => onSettle(image), { once: true });
                      }
                    });
                  };

                  const parseStageMarkup = (html) => {
                    const template = document.createElement("template");
                    template.innerHTML = html.trim();
                    return template.content.firstElementChild;
                  };

                  const sameArenaImageIdentity = (left, right) => {
                    if (!(left instanceof HTMLImageElement) || !(right instanceof HTMLImageElement)) {
                      return false;
                    }
                    return (
                      (left.getAttribute("src") || left.currentSrc || "") ===
                        (right.getAttribute("src") || right.currentSrc || "") &&
                      normalizeTurns(left.dataset.rotation) === normalizeTurns(right.dataset.rotation)
                    );
                  };

                  const transplantSettledArenaImages = (sourceLayer, stage) => {
                    const sourceImages = layerImages(sourceLayer);
                    const stageImages = Array.from(
                      stage?.querySelectorAll(".vote-surface .arena-image") ?? [],
                    );
                    if (!sourceImages.length || sourceImages.length !== stageImages.length) return;
                    for (let index = 0; index < stageImages.length; index += 1) {
                      const sourceImage = sourceImages[index];
                      const stageImage = stageImages[index];
                      if (!sameArenaImageIdentity(sourceImage, stageImage)) continue;
                      if (!sourceImage.complete || sourceImage.naturalWidth <= 0) continue;
                      sourceImage.dataset.settled = "1";
                      sourceImage.classList.add("is-ready");
                      stageImage.replaceWith(sourceImage);
                    }
                  };

                  const seedPreparedLayer = async (layer, payload, options = {}) => {
                    const stage = parseStageMarkup(payload.html);
                    if (!layer || !stage) return false;
                    transplantSettledArenaImages(options.preserveImagesFromLayer, stage);
                    layer.replaceChildren(stage);
                    layer.dataset.href = payload.href || "";
                    layer.dataset.localAnchor = payload.localAnchor || "";
                    await primeLayer(layer);
                    return true;
                  };

                  const refreshCurrentLayer = async (payload) => {
                    if (!currentLayer || !payload?.href || !payload?.html) return false;
                    const refreshed = await seedPreparedLayer(currentLayer, payload, {
                      preserveImagesFromLayer: currentLayer,
                    });
                    if (!refreshed) return false;
                    history.replaceState(
                      null,
                      "",
                      currentLayer.dataset.href || window.location.pathname,
                    );
                    void refillLookaheadFromReserve();
                    void refillPreservedLookaheadFromReserve();
                    return true;
                  };

                  const clearPreparedLayer = (layer) => {
                    if (!layer) return;
                    layer.dataset.href = "";
                    layer.dataset.localAnchor = "";
                    layer.replaceChildren();
                  };

                  const invalidatePreparedArena = () => {
                    preparedEpoch += 1;
                    lookaheadReady = false;
                    preservedLookaheadReady = false;
                    queuedLookaheadPayload = null;
                    queuedPreservedPayload = null;
                    queuedLookaheadRequest = null;
                    queuedPreservedRequest = null;
                    clearPreparedLayer(lookaheadLayer);
                    clearPreparedLayer(preservedLookaheadLayer);
                    return preparedEpoch;
                  };

                  const fetchArenaPayload = async (url) => {
                    const response = await fetch(url, {
                      credentials: "same-origin",
                      cache: "no-store",
                    });
                    if (!response.ok) throw new Error(`arena prefetch failed: ${response.status}`);
                    const payload = await response.json();
                    return payload.empty ? null : payload;
                  };

                  const prefetchLookahead = async () => {
                    const epoch = preparedEpoch;
                    try {
                      lookaheadReady = false;
                      const payload = await fetchArenaPayload("/api/arena/next");
                      if (epoch !== preparedEpoch) return;
                      if (!payload) {
                        clearPreparedLayer(lookaheadLayer);
                        return;
                      }
                      const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                      if (epoch !== preparedEpoch) return;
                      lookaheadReady = seeded;
                    } catch (error) {
                      if (epoch !== preparedEpoch) return;
                      console.error(error);
                      clearPreparedLayer(lookaheadLayer);
                    }
                  };

                  const prefetchPreservedLookahead = async () => {
                    const epoch = preparedEpoch;
                    const anchor = currentLayer?.dataset.localAnchor || "";
                    preservedLookaheadReady = false;
                    queuedPreservedPayload = null;
                    if (!anchor) {
                      clearPreparedLayer(preservedLookaheadLayer);
                      return;
                    }
                    try {
                      const payload = await fetchArenaPayload(
                        `/api/arena/next?anchor=${encodeURIComponent(anchor)}`,
                      );
                      if (epoch !== preparedEpoch) return;
                      if (!payload) {
                        clearPreparedLayer(preservedLookaheadLayer);
                        return;
                      }
                      const seeded = await seedPreparedLayer(
                        preservedLookaheadLayer,
                        payload,
                      );
                      if (epoch !== preparedEpoch) return;
                      preservedLookaheadReady = seeded;
                    } catch (error) {
                      if (epoch !== preparedEpoch) return;
                      console.error(error);
                      clearPreparedLayer(preservedLookaheadLayer);
                    }
                  };

                  const queueLookaheadReserve = async () => {
                    const epoch = preparedEpoch;
                    if (queuedLookaheadPayload?.epoch === epoch || queuedLookaheadRequest?.epoch === epoch) return;
                    const promise = fetchArenaPayload("/api/arena/next")
                      .then((payload) => {
                        if (epoch !== preparedEpoch) return;
                        queuedLookaheadPayload = payload ? { epoch, payload } : null;
                      })
                      .catch((error) => {
                        if (epoch !== preparedEpoch) return;
                        console.error(error);
                        queuedLookaheadPayload = null;
                      })
                      .finally(() => {
                        if (queuedLookaheadRequest?.epoch === epoch) {
                          queuedLookaheadRequest = null;
                        }
                      });
                    queuedLookaheadRequest = { epoch, promise };
                    await promise;
                  };

                  const queuePreservedReserve = async () => {
                    const epoch = preparedEpoch;
                    const anchor = currentLayer?.dataset.localAnchor || "";
                    if (!anchor) {
                      queuedPreservedPayload = null;
                      return;
                    }
                    if (queuedPreservedPayload?.epoch === epoch && queuedPreservedPayload?.anchor === anchor) return;
                    if (queuedPreservedRequest?.epoch === epoch && queuedPreservedRequest?.anchor === anchor) {
                      await queuedPreservedRequest.promise;
                      return;
                    }
                    queuedPreservedPayload = null;
                    const promise = fetchArenaPayload(`/api/arena/next?anchor=${encodeURIComponent(anchor)}`)
                      .then((payload) => {
                        if (epoch !== preparedEpoch) return;
                        queuedPreservedPayload = payload ? { epoch, anchor, payload } : null;
                      })
                      .catch((error) => {
                        if (epoch !== preparedEpoch) return;
                        console.error(error);
                        queuedPreservedPayload = null;
                      })
                      .finally(() => {
                        if (
                          queuedPreservedRequest?.epoch === epoch &&
                          queuedPreservedRequest?.anchor === anchor
                        ) {
                          queuedPreservedRequest = null;
                        }
                      });
                    queuedPreservedRequest = { epoch, anchor, promise };
                    await promise;
                  };

                  const refillLookaheadFromReserve = async () => {
                    const epoch = preparedEpoch;
                    const payload = queuedLookaheadPayload?.epoch === epoch ? queuedLookaheadPayload.payload : null;
                    queuedLookaheadPayload = null;
                    if (payload) {
                      const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                      if (epoch !== preparedEpoch) return;
                      lookaheadReady = seeded;
                    } else {
                      await prefetchLookahead();
                    }
                    if (epoch !== preparedEpoch) return;
                    void queueLookaheadReserve();
                  };

                  const refillPreservedLookaheadFromReserve = async () => {
                    const epoch = preparedEpoch;
                    const anchor = currentLayer?.dataset.localAnchor || "";
                    const payload =
                      queuedPreservedPayload?.epoch === epoch && queuedPreservedPayload?.anchor === anchor
                        ? queuedPreservedPayload.payload
                        : null;
                    queuedPreservedPayload = null;
                    if (!anchor) {
                      clearPreparedLayer(preservedLookaheadLayer);
                      preservedLookaheadReady = false;
                      return;
                    }
                    if (payload) {
                      const seeded = await seedPreparedLayer(
                        preservedLookaheadLayer,
                        payload,
                      );
                      if (epoch !== preparedEpoch) return;
                      preservedLookaheadReady = seeded;
                    } else {
                      await prefetchPreservedLookahead();
                    }
                    if (epoch !== preparedEpoch) return;
                    void queuePreservedReserve();
                  };

                  const ensurePreparedLookahead = async (usePreservedLookahead) => {
                    if (usePreservedLookahead) {
                      if (preservedLookaheadReady && preservedLookaheadLayer?.dataset.href) return true;
                      await refillPreservedLookaheadFromReserve();
                      return !!(preservedLookaheadReady && preservedLookaheadLayer?.dataset.href);
                    }
                    if (lookaheadReady && lookaheadLayer?.dataset.href) return true;
                    await refillLookaheadFromReserve();
                    return !!(lookaheadReady && lookaheadLayer?.dataset.href);
                  };

                  const bootPreparedLayer = async (layer, markReady, prefetch) => {
                    if (layerImages(layer).length) {
                      await primeLayer(layer);
                      markReady();
                      return;
                    }
                    await prefetch();
                  };

                  const bootLookahead = async () => {
                    await bootPreparedLayer(lookaheadLayer, () => {
                      lookaheadReady = true;
                    }, prefetchLookahead);
                    void queueLookaheadReserve();
                  };

                  const bootPreservedLookahead = async () => {
                    await bootPreparedLayer(preservedLookaheadLayer, () => {
                      preservedLookaheadReady = true;
                    }, prefetchPreservedLookahead);
                    void queuePreservedReserve();
                  };

                  const rotateArenaImage = async (form) => {
                    const frame = form.closest(".frame");
                    const image = frame?.querySelector(".arena-image");
                    if (!image) return;
                    const body = new URLSearchParams(new FormData(form));
                    try {
                      const response = await fetch(form.action, {
                        method: "POST",
                        body,
                        credentials: "same-origin",
                        cache: "no-store",
                      });
                      if (!response.ok) throw new Error(`rotate failed: ${response.status}`);
                      const nextRotation = body.get("next_rotation");
                      if (nextRotation === null) {
                        window.location.reload();
                        return;
                      }
                      image.dataset.rotation = nextRotation;
                      applyArenaSplit(form.closest(".arena-stage-layer"));
                      fitArenaImage(image);
                    } catch (error) {
                      console.error(error);
                      window.location.reload();
                    }
                  };

                  const promotePreparedLayer = async (incoming, bufferClass) => {
                    const outgoing = currentLayer;
                    if (!outgoing || !incoming || !incoming.dataset.href) return false;
                    isTransitioning = true;
                    arenaShell.classList.add("is-transitioning");
                    incoming.classList.remove("is-hidden", "is-lookahead", "is-preserved-lookahead");
                    incoming.setAttribute("aria-hidden", "false");
                    requestAnimationFrame(() => {
                      outgoing.classList.add("is-exiting");
                      incoming.classList.add("is-entering");
                    });
                    await new Promise((resolve) => window.setTimeout(resolve, TRANSITION_MS));
                    outgoing.classList.remove(
                      "is-current",
                      "is-exiting",
                      "is-lookahead",
                      "is-preserved-lookahead",
                    );
                    outgoing.classList.add(bufferClass, "is-hidden");
                    outgoing.setAttribute("aria-hidden", "true");
                    clearPreparedLayer(outgoing);
                    incoming.classList.remove("is-entering");
                    incoming.classList.add("is-current");
                    currentLayer = incoming;
                    if (bufferClass === "is-lookahead") {
                      lookaheadLayer = outgoing;
                      lookaheadReady = false;
                      void refillLookaheadFromReserve();
                    } else {
                      preservedLookaheadLayer = outgoing;
                      preservedLookaheadReady = false;
                      void refillPreservedLookaheadFromReserve();
                    }
                    history.replaceState(null, "", currentLayer.dataset.href || window.location.pathname);
                    arenaShell.classList.remove("is-transitioning");
                    isTransitioning = false;
                    void queueLookaheadReserve();
                    void queuePreservedReserve();
                    return true;
                  };

                  const promoteLookahead = async () => {
                    if (!lookaheadReady || !lookaheadLayer?.dataset.href) return false;
                    return promotePreparedLayer(lookaheadLayer, "is-lookahead");
                  };

                  const promotePreservedLookahead = async () => {
                    if (!preservedLookaheadReady || !preservedLookaheadLayer?.dataset.href) return false;
                    return promotePreparedLayer(preservedLookaheadLayer, "is-preserved-lookahead");
                  };

                  arenaShell.addEventListener("pointerdown", (event) => {
                    const target = eventElement(event);
                    if (target?.closest(".frame-tools")) {
                      event.stopPropagation();
                    }
                  });

                  arenaShell.addEventListener("click", (event) => {
                    const target = eventElement(event);
                    if (!target) return;
                    if (target.closest(".frame-tools")) {
                      event.stopPropagation();
                      return;
                    }
                    const surface = target.closest(".vote-surface");
                    if (!surface || isTransitioning || isActionPending) return;
                    event.preventDefault();
                    surface.closest(".frame")?.querySelector(".vote-form")?.requestSubmit();
                  });

                  arenaShell.addEventListener("submit", async (event) => {
                    const form = event.target;
                    if (!(form instanceof HTMLFormElement)) return;
                    const advancePolicy = arenaAdvancePolicy(form);
                    if (advancePolicy) {
                      const formData = new FormData(form);
                      event.preventDefault();
                      if (isTransitioning || isActionPending) {
                        return;
                      }
                      const body = new URLSearchParams(formData);
                      if (advancePolicy === "buffered") {
                        const ready = await ensurePreparedLookahead(false);
                        if (!ready) {
                          form.submit();
                          return;
                        }
                        const sendAction = fetch(form.action, {
                          method: "POST",
                          body,
                          credentials: "same-origin",
                          cache: "no-store",
                          headers: { Accept: "application/json" },
                        });
                        const promoted = await promoteLookahead();
                        if (!promoted) {
                          window.location.reload();
                          return;
                        }
                        sendAction
                          .then((response) => {
                            if (!response.ok) throw new Error(`arena action failed: ${response.status}`);
                          })
                          .catch((error) => {
                            console.error(error);
                            window.location.reload();
                          });
                        return;
                      }
                      const epoch = invalidatePreparedArena();
                      isActionPending = true;
                      try {
                        const response = await fetch(form.action, {
                          method: "POST",
                          body,
                          credentials: "same-origin",
                          cache: "no-store",
                          headers: { Accept: "application/json" },
                        });
                        if (!response.ok) {
                          throw new Error(`arena action failed: ${response.status}`);
                        }
                        const payload = await response.json();
                        if (epoch !== preparedEpoch) return;
                        if (payload?.empty || !payload?.href || !payload?.html) {
                          window.location.assign(payload?.href || form.action);
                          return;
                        }
                        if (advancePolicy === "refresh-current") {
                          const refreshed = await refreshCurrentLayer(payload);
                          if (!refreshed) {
                            window.location.assign(payload.href);
                          }
                          return;
                        }
                        const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                        if (epoch !== preparedEpoch) return;
                        lookaheadReady = seeded;
                        if (!seeded) {
                          window.location.assign(payload.href);
                          return;
                        }
                        const promoted = await promoteLookahead();
                        if (!promoted) {
                          window.location.assign(payload.href);
                        }
                      } catch (error) {
                        console.error(error);
                        window.location.reload();
                      } finally {
                        isActionPending = false;
                      }
                      return;
                    }
                    if (form.closest(".frame-tools") && form.querySelector(".rotate-tool")) {
                      event.preventDefault();
                      event.stopPropagation();
                      await rotateArenaImage(form);
                    }
                  });

                  document.addEventListener("keydown", (event) => {
                    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
                    if (isTransitioning || isActionPending) return;
                    const voteForms = currentVoteForms();
                    const hideForms = currentHideForms();
                    const vetoThreadForms = currentVetoThreadForms();
                    if (event.altKey) {
                      if (event.key === "1") {
                        event.preventDefault();
                        hideForms[0]?.requestSubmit();
                        return;
                      }
                      if (event.key === "2") {
                        event.preventDefault();
                        hideForms[1]?.requestSubmit();
                        return;
                      }
                      if (event.key === "3") {
                        event.preventDefault();
                        vetoThreadForms[0]?.requestSubmit();
                        return;
                      }
                    }
                    if (event.key === "1") {
                      event.preventDefault();
                      voteForms[0]?.requestSubmit();
                    }
                    if (event.key === "2") {
                      event.preventDefault();
                      voteForms[1]?.requestSubmit();
                    }
                  });

                  window.addEventListener("resize", () => {
                    for (const layer of [currentLayer, lookaheadLayer, preservedLookaheadLayer]) {
                      applyArenaSplit(layer);
                      for (const image of layerImages(layer)) fitArenaImage(image);
                    }
                  });

                  primeLayer(currentLayer).then(() => {
                    void bootLookahead();
                    void bootPreservedLookahead();
                  });
                })();
                "#,
            ))
        }
    }
}

fn design_language_font_path(font_name: &str) -> anyhow::Result<PathBuf> {
    let file_name = match font_name {
        "oxanium-400.woff2" | "oxanium-600.woff2" | "share-tech-mono-400.woff2" => font_name,
        _ => bail!("unknown font asset {font_name}"),
    };
    Ok(FsPath::new(SWARM_DESIGN_LANGUAGE_ROOT)
        .join("assets")
        .join("fonts")
        .join(file_name))
}

const SWARM_DESIGN_LANGUAGE_ROOT: &str =
    "/home/main/programming/projects/swarm.moe/design-language";
const RENDITION_CACHE_VERSION: u32 = 5;

#[derive(Debug, Clone, Copy)]
enum NavPage {
    Arena,
    Board,
    Facemash,
    Identities,
    Explore,
    Triad,
    Vocab,
}

#[derive(Debug, Deserialize)]
struct VoteForm {
    left_id: String,
    right_id: String,
    winner_id: String,
}

#[derive(Debug, Deserialize)]
struct RotateForm {
    asset_id: String,
    direction: i32,
    #[serde(rename = "next_rotation")]
    _next_rotation: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct NudgeForm {
    asset_id: String,
    delta: i32,
}

#[derive(Debug, Deserialize)]
struct HeartAssetForm {
    asset_id: String,
    active: bool,
}

#[derive(Debug, Deserialize)]
struct HideAssetForm {
    asset_id: String,
}

#[derive(Debug, Deserialize)]
struct AssetDomainForm {
    asset_id: String,
    label: AssetDomainLabel,
}

#[derive(Debug, Deserialize)]
struct FacemashVoteForm {
    winner_face_id: String,
    loser_face_id: String,
}

#[derive(Debug, Deserialize)]
struct FacemashHideForm {
    face_id: String,
}

#[derive(Debug, Deserialize)]
struct IdentityThresholdForm {
    percent: u8,
}

#[derive(Debug, Deserialize)]
struct IdentityNameForm {
    anchor_handle: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct IdentityPairForm {
    pair_handle: String,
}

#[derive(Debug, Deserialize, Default)]
struct IdentitiesQuery {
    anchor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HideForm {
    asset_id: String,
    hide: bool,
    #[serde(default, deserialize_with = "deserialize_comma_ids")]
    cluster_ids: Vec<i64>,
}

fn deserialize_comma_ids<'de, D>(deserializer: D) -> Result<Vec<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|s| s.trim().parse::<i64>().map_err(serde::de::Error::custom))
        .collect()
}

#[derive(Debug, Deserialize)]
struct HandleForm {
    asset_id: String,
}

#[derive(Debug, Deserialize)]
struct ThreadLockForm {
    asset_id: String,
    active: bool,
}

#[derive(Debug, Deserialize)]
struct ExternalProbabilityForm {
    percent: u8,
}

#[derive(Debug, Deserialize)]
struct ArenaExploreForm {
    percent: u8,
}

#[derive(Debug, Deserialize)]
struct DedupRadiusForm {
    percent: u8,
}

#[derive(Debug, Deserialize)]
struct FacemashMinFaceSideForm {
    px: u16,
}

#[derive(Debug, Deserialize, Default)]
struct ClientRouteQuery {
    focus: Option<String>,
    mode: Option<String>,
    triad: Option<String>,
}

impl ClientRouteQuery {
    fn map_mode(&self) -> ExploreMapMode {
        self.mode
            .as_deref()
            .and_then(|value| ExploreMapMode::from_str(value).ok())
            .unwrap_or_default()
    }

    fn focus_asset_id(&self) -> Option<AssetId> {
        self.focus
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(|value| AssetId(value.to_owned()))
    }

    fn triad_assets(&self) -> anyhow::Result<Option<[AssetId; 3]>> {
        self.triad
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(parse_triad_query)
            .transpose()
    }
}

fn parse_triad_query(raw: &str) -> anyhow::Result<[AssetId; 3]> {
    let parts = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| AssetId(part.to_owned()))
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("triad query must contain exactly three asset ids");
    }
    if parts[0] == parts[1] || parts[0] == parts[2] || parts[1] == parts[2] {
        bail!("triad query must contain distinct asset ids");
    }
    Ok([parts[0].clone(), parts[1].clone(), parts[2].clone()])
}

type WebResult<T> = Result<T, AppError>;

struct AppError(anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let markup = layout(
            "empty-page",
            html! {
                section.empty-state {
                    div.error-shell {
                        p { (self.0.to_string()) }
                        a href="/" { "return home" }
                    }
                }
            },
        );
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(markup.into_string()),
        )
            .into_response()
    }
}

fn no_store_response(mut response: Response) -> Response {
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response
}

fn render_markup(markup: Markup) -> Response {
    no_store_response(Html(markup.into_string()).into_response())
}

fn cached_text_response(body: &str, mime: HeaderValue) -> Response {
    let mut response = Response::new(body.to_owned().into());
    response.headers_mut().insert(CONTENT_TYPE, mime);
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}

fn cached_image_response(body: Vec<u8>, mime: HeaderValue) -> Response {
    let mut response = Response::new(body.into());
    response.headers_mut().insert(CONTENT_TYPE, mime);
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}

const EXPLORE_EMPTY_NOTE: &str =
    "Explore needs embedded images. Let DINO finish, then reload or rescan.";
const TRIAD_EMPTY_NOTE: &str = "Need at least three embedded images to train the similarity field.";
