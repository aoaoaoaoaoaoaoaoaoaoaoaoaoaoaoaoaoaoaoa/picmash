use std::path::{Path as FsPath, PathBuf};

use anyhow::bail;
use axum::http::{HeaderMap, header::REFERER};
use maud::{DOCTYPE, Markup, PreEscaped, html};

#[derive(Debug, Clone, Copy)]
pub(super) enum ImageToolKind {
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
    pub(super) const fn label(self) -> &'static str {
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

    pub(super) const fn title(self) -> &'static str {
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

    pub(super) const fn class(self, mini: bool) -> &'static str {
        match (self, mini) {
            (Self::RotateLeft | Self::RotateRight, true) => "tool mini rotate-tool",
            (Self::RotateLeft | Self::RotateRight, false) => "tool rotate-tool",
            (Self::Heart, true) => "tool mini heart",
            (Self::Heart, false) => "tool heart",
            (Self::Hide | Self::VetoThread, true) => "tool mini danger",
            (Self::Hide | Self::VetoThread, false) => "tool danger",
            (_, true) => "tool mini",
            (_, false) => "tool",
        }
    }
}

pub(super) fn tool_button(kind: ImageToolKind, mini: bool, active: bool) -> Markup {
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
pub(super) enum PageGeometry {
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

pub(super) fn referer_redirect_target(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(REFERER)?.to_str().ok()?;
    if raw.starts_with('/') {
        return Some(raw.to_owned());
    }

    let (_, rest) = raw.split_once("://")?;
    let slash = rest.find('/')?;
    Some(rest[slash..].to_owned())
}

pub(super) fn frontend_host_markup(
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

pub(super) fn layout(page_class: &'static str, body: Markup) -> Markup {
    bare_layout(page_class, body)
}

pub(super) fn routed_layout(
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
                  const freshTelemetryId = (prefix) => {
                    const uuid = globalThis.crypto?.randomUUID?.();
                    if (uuid) return `${prefix}_${uuid}`;
                    return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
                  };
                  const pageId = freshTelemetryId("page");
                  const telemetry = {
                    pageId,
                    nextInteractionId(kind) {
                      return freshTelemetryId(kind || "interaction");
                    },
                    headers({ headers, interactionId, arenaEpoch } = {}) {
                      const values = new Headers(headers || {});
                      values.set("x-picmash-page-id", pageId);
                      if (interactionId) values.set("x-picmash-interaction-id", interactionId);
                      if (arenaEpoch !== undefined && arenaEpoch !== null && `${arenaEpoch}` !== "") {
                        values.set("x-picmash-arena-epoch", `${arenaEpoch}`);
                      }
                      return values;
                    },
                    reportClientEvent(kind, payload = {}) {
                      const body = JSON.stringify({
                        kind,
                        page_id: pageId,
                        ...payload,
                      });
                      try {
                        const beacon = new Blob([body], { type: "application/json" });
                        if (globalThis.navigator?.sendBeacon?.("/api/client-event", beacon)) return;
                      } catch (_error) {}
                      fetch("/api/client-event", {
                        method: "POST",
                        body,
                        credentials: "same-origin",
                        cache: "no-store",
                        keepalive: true,
                        headers: { "content-type": "application/json" },
                      }).catch(() => {});
                    },
                  };
                  window.__picmashTelemetry = telemetry;

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
                        headers: telemetry.headers(),
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
                        if (loadingPage) {
                          shouldReload = true;
                        }
                        const message = phase === "failed"
                          ? (status.message || "picmash boot failed")
                          : "reconnecting";
                        setOverlay(true, message);
                        pollDelay = 900;
                      }
                    } catch (_error) {
                      if (loadingPage) {
                        shouldReload = true;
                      }
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

pub(super) fn script_block() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"
                (() => {
                  const arenaShell = document.querySelector(".arena-stage-shell");
                  if (!arenaShell) return;

                  const PREFETCH_DEPTH = 5;
                  const PREFETCH_RESERVE_DEPTH = Math.max(0, PREFETCH_DEPTH - 1);
                  const TRANSITION_MS = 300;
                  const normalizeTurns = (value) => ((Number(value) || 0) % 4 + 4) % 4;
                  let currentLayer = arenaShell.querySelector(".arena-stage-layer.is-current");
                  let lookaheadLayer = arenaShell.querySelector(".arena-stage-layer.is-lookahead");
                  let lookaheadReady = false;
                  let preparedEpoch = 0;
                  let queuedLookaheadPayloads = [];
                  let queuedLookaheadRequest = null;
                  let isTransitioning = false;
                  let isActionPending = false;
                  const telemetry = window.__picmashTelemetry || null;

                  const layerStage = (layer) => layer?.querySelector(".arena-stage") ?? null;
                  const layerImages = (layer) =>
                    Array.from(layer?.querySelectorAll(".vote-surface .arena-image") ?? []);
                  const layerRole = (layer) => {
                    if (!layer) return "";
                    if (layer === currentLayer) return "current";
                    if (layer === lookaheadLayer) return "lookahead";
                    return "unknown";
                  };
                  const eventElement = (event) =>
                    event.target instanceof Element ? event.target : event.target?.parentElement ?? null;
                  const currentVoteForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".vote-form") ?? []);
                  const currentHideForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".arena-hide-form") ?? []);
                  const currentVetoThreadForms = () =>
                    Array.from(currentLayer?.querySelectorAll(".arena-veto-thread-form") ?? []);
                  const arenaAdvancePolicy = (form) => form.dataset.arenaAdvance || "";
                  const arenaRootHref = () => "/arena";
                  const currentPairHref = () => currentLayer?.dataset.href || window.location.pathname;
                  const telemetryInteractionId = (prefix) =>
                    telemetry?.nextInteractionId?.(prefix || "arena") || "";
                  const telemetryHeaders = (interactionId, headers) =>
                    telemetry?.headers?.({
                      headers,
                      interactionId,
                      arenaEpoch: preparedEpoch,
                    }) || new Headers(headers || {});
                  const reportArenaAnomaly = (kind, payload = {}) => {
                    telemetry?.reportClientEvent?.(kind, {
                      pair_href: currentPairHref(),
                      current_href: currentLayer?.dataset.href || "",
                      lookahead_href: lookaheadLayer?.dataset.href || "",
                      arena_epoch: preparedEpoch,
                      ...payload,
                    });
                  };
                  const parseVisualKeys = (value) =>
                    (value || "")
                      .split(",")
                      .map((entry) => entry.trim())
                      .filter((entry) => entry.length > 0);
                  const layerVisualKeys = (layer) => parseVisualKeys(layer?.dataset.visualKeys || "");
                  const payloadVisualKeys = (payload) =>
                    Array.isArray(payload?.visualKeys)
                      ? payload.visualKeys.filter((entry) => typeof entry === "string" && entry.length > 0)
                      : [];
                  const payloadTurnId = (payload) =>
                    typeof payload?.turnId === "string" ? payload.turnId : "";
                  const layerTurnId = (layer) => layer?.dataset.turnId || "";
                  const collectKnownTurnIds = () => {
                    const known = new Set();
                    const lookaheadTurnId = layerTurnId(lookaheadLayer);
                    if (lookaheadTurnId) known.add(lookaheadTurnId);
                    for (const payload of queuedLookaheadPayloads) {
                      const turnId = payloadTurnId(payload);
                      if (turnId) known.add(turnId);
                    }
                    return Array.from(known);
                  };
                  const collectExcludedVisualKeys = () => {
                    const excluded = new Set();
                    for (const key of layerVisualKeys(currentLayer)) excluded.add(key);
                    for (const key of layerVisualKeys(lookaheadLayer)) excluded.add(key);
                    for (const payload of queuedLookaheadPayloads) {
                      for (const key of payloadVisualKeys(payload)) excluded.add(key);
                    }
                    return Array.from(excluded);
                  };
                  const arenaPrefetchUrl = (base, excludeVisualKeys, knownTurnIds) => {
                    if (!excludeVisualKeys.length && !knownTurnIds.length) return base;
                    const url = new URL(base, window.location.origin);
                    if (excludeVisualKeys.length) {
                      url.searchParams.set("exclude_visual_keys", excludeVisualKeys.join(","));
                    }
                    if (knownTurnIds.length) {
                      url.searchParams.set("known_turn_ids", knownTurnIds.join(","));
                    }
                    return `${url.pathname}${url.search}`;
                  };

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

                  const revealLayer = (layer, options = {}) => {
                    const progressive = options.progressive === true;
                    const images = layerImages(layer);
                    if (!images.length) {
                      return false;
                    }
                    const settledImages = images.filter(
                      (image) =>
                        image.dataset.settled === "1" && image.dataset.failed !== "1",
                    );
                    if (!settledImages.length) {
                      return false;
                    }
                    const complete = settledImages.length === images.length;
                    if (!progressive && !complete) {
                      return false;
                    }
                    applyArenaSplit(layer);
                    for (const image of settledImages) {
                      fitArenaImage(image);
                      image.classList.add("is-ready");
                    }
                    return complete || progressive;
                  };

                  const primeLayer = (layer, options = {}) => {
                    const progressive = options.progressive === true;
                    const images = layerImages(layer);
                    if (!layer || !images.length) return Promise.resolve(false);
                    const pending = [];
                    for (const image of images) {
                      if (
                        image.dataset.settled === "1" &&
                        image.dataset.failed !== "1" &&
                        image.classList.contains("is-ready") &&
                        image.complete &&
                        image.naturalWidth > 0
                      ) {
                        continue;
                      }
                      image.dataset.settled = "";
                      image.dataset.failed = "";
                      image.classList.remove("is-ready");
                      pending.push(image);
                    }
                    if (!pending.length) {
                      return Promise.resolve(revealLayer(layer, { progressive }));
                    }
                    return new Promise((resolve) => {
                      let remaining = pending.length;
                      let failed = false;
                      const onLoad = (image) => {
                        if (image.dataset.settled === "1") return;
                        image.dataset.failed = "";
                        image.dataset.settled = "1";
                        remaining -= 1;
                        const revealed = revealLayer(layer, { progressive });
                        if (remaining <= 0) resolve(progressive ? revealed : revealed && !failed);
                      };
                      const onError = (image) => {
                        if (image.dataset.settled === "1") return;
                        image.dataset.failed = "1";
                        image.dataset.settled = "1";
                        image.classList.remove("is-ready");
                        revealLayer(layer, { progressive });
                        reportArenaAnomaly("arena_image_error", {
                          layer_role: layerRole(layer),
                          image_src: image.currentSrc || image.getAttribute("src") || "",
                        });
                        failed = true;
                        remaining -= 1;
                        if (remaining <= 0) resolve(false);
                      };
                      for (const image of pending) {
                        if (image.complete) {
                          if (image.naturalWidth > 0) {
                            onLoad(image);
                          } else {
                            onError(image);
                          }
                          continue;
                        }
                        image.addEventListener("load", () => onLoad(image), { once: true });
                        image.addEventListener("error", () => onError(image), { once: true });
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

                  const setLayerMetadata = (layer, payload) => {
                    if (!layer || !payload) return;
                    layer.dataset.href = payload.href || "";
                    layer.dataset.turnId = payload.turnId || "";
                    layer.dataset.actionToken = payload.actionToken || "";
                    layer.dataset.arenaRevision = `${payload.revision ?? ""}`;
                    layer.dataset.arenaSamplerEpoch = `${payload.samplerEpoch ?? ""}`;
                    layer.dataset.visualKeys = payloadVisualKeys(payload).join(",");
                  };

                  const stampLayerForms = (layer) => {
                    if (!layer) return;
                    const fields = {
                      turn_id: layer.dataset.turnId || "",
                      action_token: layer.dataset.actionToken || "",
                      arena_revision: layer.dataset.arenaRevision || "",
                      arena_sampler_epoch: layer.dataset.arenaSamplerEpoch || "",
                    };
                    for (const form of layer.querySelectorAll("form")) {
                      for (const [name, value] of Object.entries(fields)) {
                        const input = form.querySelector(`input[name="${name}"]`);
                        if (input) input.value = value;
                      }
                    }
                  };

                  const syncCurrentLayer = (payload) => {
                    if (!payload || payload.empty || !currentLayer) return false;
                    const currentTurnId = currentLayer.dataset.turnId || "";
                    if (currentTurnId && payload.turnId && currentTurnId !== payload.turnId) {
                      return false;
                    }
                    setLayerMetadata(currentLayer, payload);
                    stampLayerForms(currentLayer);
                    history.replaceState(null, "", currentLayer.dataset.href || window.location.pathname);
                    return true;
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
                    if (!layer || !stage) {
                      reportArenaAnomaly("arena_prefetch_payload_rejected", {
                        interaction_id: options.interactionId || "",
                        layer_role: layerRole(layer),
                        pair_href: payload?.href || currentPairHref(),
                      });
                      return false;
                    }
                    transplantSettledArenaImages(options.preserveImagesFromLayer, stage);
                    layer.replaceChildren(stage);
                    setLayerMetadata(layer, payload);
                    stampLayerForms(layer);
                    return primeLayer(layer, {
                      progressive: options.progressive === true,
                    });
                  };

                  const refreshCurrentLayer = async (payload) => {
                    if (!currentLayer || !payload?.href || !payload?.html) return false;
                    const refreshed = await seedPreparedLayer(currentLayer, payload, {
                      preserveImagesFromLayer: currentLayer,
                      progressive: true,
                    });
                    if (!refreshed) {
                      reportArenaAnomaly("arena_refresh_current_failed", {
                        pair_href: payload.href,
                        layer_role: "current",
                      });
                      return false;
                    }
                    history.replaceState(
                      null,
                      "",
                      currentLayer.dataset.href || window.location.pathname,
                    );
                    void refillLookaheadFromReserve();
                    return true;
                  };

                  const clearPreparedLayer = (layer) => {
                    if (!layer) return;
                    layer.dataset.href = "";
                    layer.dataset.turnId = "";
                    layer.dataset.actionToken = "";
                    layer.dataset.arenaRevision = "";
                    layer.dataset.arenaSamplerEpoch = "";
                    layer.dataset.visualKeys = "";
                    layer.replaceChildren();
                  };

                  const invalidatePreparedArena = () => {
                    preparedEpoch += 1;
                    lookaheadReady = false;
                    queuedLookaheadPayloads = [];
                    queuedLookaheadRequest = null;
                    clearPreparedLayer(lookaheadLayer);
                    return preparedEpoch;
                  };

                  const rewarmPreparedArena = async () => {
                    invalidatePreparedArena();
                    await refillLookaheadFromReserve();
                  };

                  const fetchArenaPayload = async (url, interactionId) => {
                    const response = await fetch(url, {
                      credentials: "same-origin",
                      cache: "no-store",
                      headers: telemetryHeaders(interactionId),
                    });
                    if (!response.ok) throw new Error(`arena prefetch failed: ${response.status}`);
                    const payload = await response.json();
                    if (payload.empty) return null;
                    if (
                      !payload?.href ||
                      !payload?.html ||
                      !payload?.turnId ||
                      !payload?.actionToken ||
                      payload?.revision === undefined ||
                      payload?.samplerEpoch === undefined
                    ) {
                      reportArenaAnomaly("arena_prefetch_payload_rejected", {
                        interaction_id: interactionId || "",
                        pair_href: payload?.href || url,
                      });
                      return null;
                    }
                    return payload;
                  };

                  const prefetchLookahead = async () => {
                    const epoch = preparedEpoch;
                    try {
                      lookaheadReady = false;
                      const interactionId = telemetryInteractionId("arena-prefetch");
                      const payload = await fetchArenaPayload(
                        arenaPrefetchUrl(
                          "/api/arena/next",
                          collectExcludedVisualKeys(),
                          collectKnownTurnIds(),
                        ),
                        interactionId,
                      );
                      if (epoch !== preparedEpoch) return;
                      if (!payload) {
                        clearPreparedLayer(lookaheadLayer);
                        return;
                      }
                      const seeded = await seedPreparedLayer(lookaheadLayer, payload, {
                        interactionId,
                      });
                      if (epoch !== preparedEpoch) return;
                      lookaheadReady = seeded;
                    } catch (error) {
                      if (epoch !== preparedEpoch) return;
                      console.error(error);
                      clearPreparedLayer(lookaheadLayer);
                    }
                  };

                  const queueLookaheadReserve = async () => {
                    const epoch = preparedEpoch;
                    if (queuedLookaheadRequest?.epoch === epoch) {
                      await queuedLookaheadRequest.promise;
                      return;
                    }
                    const promise = (async () => {
                      while (
                        epoch === preparedEpoch &&
                        queuedLookaheadPayloads.length < PREFETCH_RESERVE_DEPTH
                      ) {
                        const interactionId = telemetryInteractionId("arena-prefetch");
                        const payload = await fetchArenaPayload(
                          arenaPrefetchUrl(
                            "/api/arena/next",
                            collectExcludedVisualKeys(),
                            collectKnownTurnIds(),
                          ),
                          interactionId,
                        );
                        if (epoch !== preparedEpoch) return;
                        if (!payload) return;
                        queuedLookaheadPayloads.push(payload);
                      }
                    })()
                      .catch((error) => {
                        if (epoch !== preparedEpoch) return;
                        console.error(error);
                        queuedLookaheadPayloads = [];
                      })
                      .finally(() => {
                        if (queuedLookaheadRequest?.epoch === epoch) {
                          queuedLookaheadRequest = null;
                        }
                      });
                    queuedLookaheadRequest = { epoch, promise };
                    await promise;
                  };

                  const refillLookaheadFromReserve = async () => {
                    const epoch = preparedEpoch;
                    while (queuedLookaheadPayloads.length) {
                      const payload = queuedLookaheadPayloads.shift();
                      if (!payload) break;
                      const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                      if (epoch !== preparedEpoch) return;
                      if (seeded) {
                        lookaheadReady = true;
                        void queueLookaheadReserve();
                        return;
                      }
                      clearPreparedLayer(lookaheadLayer);
                    }
                    await prefetchLookahead();
                    if (epoch !== preparedEpoch) return;
                    void queueLookaheadReserve();
                  };

                  const ensurePreparedLookahead = async () => {
                    if (lookaheadReady && lookaheadLayer?.dataset.href) return true;
                    await refillLookaheadFromReserve();
                    return !!(lookaheadReady && lookaheadLayer?.dataset.href);
                  };

                  const bootPreparedLayer = async (layer, markReady, prefetch) => {
                    if (layerImages(layer).length) {
                      const primed = await primeLayer(layer);
                      if (primed) {
                        markReady();
                        return;
                      }
                      clearPreparedLayer(layer);
                    }
                    await prefetch();
                  };

                  const bootLookahead = async () => {
                    await bootPreparedLayer(lookaheadLayer, () => {
                      lookaheadReady = true;
                    }, prefetchLookahead);
                    void queueLookaheadReserve();
                  };

                  const rotateArenaImage = async (form) => {
                    const frame = form.closest(".frame");
                    const image = frame?.querySelector(".arena-image");
                    if (!image) return;
                    const body = new URLSearchParams(new FormData(form));
                    const interactionId = telemetryInteractionId("arena-rotate");
                    try {
                      const response = await fetch(form.action, {
                        method: "POST",
                        body,
                        credentials: "same-origin",
                        cache: "no-store",
                        headers: telemetryHeaders(interactionId),
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

                  const promoteLookahead = async () => {
                    const outgoing = currentLayer;
                    if (!lookaheadReady || !lookaheadLayer?.dataset.href) return false;
                    if (!outgoing) {
                      reportArenaAnomaly("arena_promotion_failed", {
                        layer_role: "lookahead",
                      });
                      return false;
                    }
                    isTransitioning = true;
                    arenaShell.classList.add("is-transitioning");
                    const incoming = lookaheadLayer;
                    incoming.classList.remove("is-hidden", "is-lookahead");
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
                    );
                    outgoing.classList.add("is-lookahead", "is-hidden");
                    outgoing.setAttribute("aria-hidden", "true");
                    clearPreparedLayer(outgoing);
                    incoming.classList.remove("is-entering");
                    incoming.classList.add("is-current");
                    currentLayer = incoming;
                    lookaheadLayer = outgoing;
                    lookaheadReady = false;
                    if (!isActionPending) void refillLookaheadFromReserve();
                    history.replaceState(null, "", currentLayer.dataset.href || window.location.pathname);
                    arenaShell.classList.remove("is-transitioning");
                    isTransitioning = false;
                    if (!isActionPending) void queueLookaheadReserve();
                    return true;
                  };

                  const seedAuthoritativeLayer = async (payload) => {
                    if (!lookaheadLayer) return false;
                    const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                    lookaheadReady = seeded;
                    return seeded;
                  };

                  const salvagePreparedCurrent = () => {
                    if (layerImages(currentLayer).length) return false;
                    if (!lookaheadLayer?.dataset.href || !layerImages(lookaheadLayer).length) {
                      return false;
                    }
                    const outgoing = currentLayer;
                    outgoing.classList.remove(
                      "is-current",
                      "is-entering",
                      "is-exiting",
                      "is-lookahead",
                    );
                    outgoing.classList.add("is-lookahead", "is-hidden");
                    outgoing.setAttribute("aria-hidden", "true");
                    lookaheadLayer.classList.remove(
                      "is-hidden",
                      "is-lookahead",
                      "is-entering",
                      "is-exiting",
                    );
                    lookaheadLayer.classList.add("is-current");
                    lookaheadLayer.setAttribute("aria-hidden", "false");
                    currentLayer = lookaheadLayer;
                    lookaheadLayer = outgoing;
                    lookaheadReady = false;
                    history.replaceState(
                      null,
                      "",
                      currentLayer.dataset.href || window.location.pathname,
                    );
                    reportArenaAnomaly("arena_salvage_current_from_hidden", {
                      layer_role: "lookahead",
                    });
                    return true;
                  };

                  const resolveArenaActionPayload = async (response, fallbackHref) => {
                    const contentType = response.headers.get("content-type") || "";
                    const payload = contentType.includes("application/json")
                      ? await response.json().catch(() => null)
                      : null;
                    if (response.status === 409) {
                      window.location.assign(payload?.href || fallbackHref || arenaRootHref());
                      return null;
                    }
                    if (!response.ok) {
                      throw new Error(`arena action failed: ${response.status}`);
                    }
                    if (payload?.empty || !payload?.href || !payload?.html) {
                      window.location.assign(payload?.href || fallbackHref);
                      return null;
                    }
                    return payload;
                  };

                  const stampArenaCommandBody = (form, body) => {
                    const layer = form.closest(".arena-stage-layer") || currentLayer;
                    if (!body.get("command_id")) {
                      body.set("command_id", telemetryInteractionId("arena-command"));
                    }
                    body.set("turn_id", layer?.dataset.turnId || body.get("turn_id") || "");
                    body.set("action_token", layer?.dataset.actionToken || body.get("action_token") || "");
                    body.set("arena_revision", layer?.dataset.arenaRevision || body.get("arena_revision") || "");
                    body.set("arena_sampler_epoch", layer?.dataset.arenaSamplerEpoch || body.get("arena_sampler_epoch") || "");
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
                      stampArenaCommandBody(form, body);
                      if (advancePolicy === "buffered") {
                        const interactionId = telemetryInteractionId("arena-action");
                        isActionPending = true;
                        const sendAction = fetch(form.action, {
                          method: "POST",
                          body,
                          credentials: "same-origin",
                          cache: "no-store",
                          headers: telemetryHeaders(interactionId, { Accept: "application/json" }),
                        });
                        sendAction.catch(() => {});
                        try {
                          const ready = await ensurePreparedLookahead();
                          if (ready) {
                            const promoted = await promoteLookahead();
                            if (!promoted) {
                              window.location.reload();
                              return;
                            }
                            const payload = await resolveArenaActionPayload(
                              await sendAction,
                              form.action,
                            );
                            if (!payload) {
                              return;
                            }
                            if (!syncCurrentLayer(payload)) {
                              const refreshed = await refreshCurrentLayer(payload);
                              if (!refreshed) {
                                window.location.assign(payload.href);
                                return;
                              }
                            }
                            await rewarmPreparedArena();
                            return;
                          }
                          const payload = await resolveArenaActionPayload(
                            await sendAction,
                            form.action,
                          );
                          if (!payload) {
                            return;
                          }
                          const seeded = await seedAuthoritativeLayer(payload);
                          if (!seeded) {
                            reportArenaAnomaly("arena_promotion_failed", {
                              interaction_id: interactionId,
                              pair_href: payload.href,
                              layer_role: "lookahead",
                            });
                            window.location.assign(payload.href);
                            return;
                          }
                          const promoted = await promoteLookahead();
                          if (!promoted) {
                            reportArenaAnomaly("arena_promotion_failed", {
                              interaction_id: interactionId,
                              pair_href: payload.href,
                              layer_role: "lookahead",
                            });
                            window.location.assign(payload.href);
                            return;
                          }
                          await rewarmPreparedArena();
                        } catch (error) {
                          console.error(error);
                          window.location.reload();
                        } finally {
                          isActionPending = false;
                        }
                        return;
                      }
                      const epoch = invalidatePreparedArena();
                      const interactionId = telemetryInteractionId("arena-action");
                      isActionPending = true;
                      try {
                        const response = await fetch(form.action, {
                          method: "POST",
                          body,
                          credentials: "same-origin",
                          cache: "no-store",
                          headers: telemetryHeaders(interactionId, { Accept: "application/json" }),
                        });
                        const payload = await resolveArenaActionPayload(response, form.action);
                        if (epoch !== preparedEpoch) return;
                        if (!payload) {
                          return;
                        }
                        if (advancePolicy === "refresh-current") {
                          const refreshed = await refreshCurrentLayer(payload);
                          if (!refreshed) {
                            reportArenaAnomaly("arena_refresh_current_failed", {
                              interaction_id: interactionId,
                              pair_href: payload.href,
                            });
                            window.location.assign(payload.href);
                          }
                          return;
                        }
                        const seeded = await seedPreparedLayer(lookaheadLayer, payload);
                        if (epoch !== preparedEpoch) return;
                        lookaheadReady = seeded;
                        if (!seeded) {
                          reportArenaAnomaly("arena_promotion_failed", {
                            interaction_id: interactionId,
                            pair_href: payload.href,
                            layer_role: "lookahead",
                          });
                          window.location.assign(payload.href);
                          return;
                        }
                        const promoted = await promoteLookahead();
                        if (!promoted) {
                          reportArenaAnomaly("arena_promotion_failed", {
                            interaction_id: interactionId,
                            pair_href: payload.href,
                            layer_role: "lookahead",
                          });
                          window.location.assign(payload.href);
                          return;
                        }
                        await rewarmPreparedArena();
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
                    let submitShortcut = null;
                    if (event.altKey) {
                      if (event.key === "1") {
                        submitShortcut = () => currentHideForms()[0]?.requestSubmit();
                      }
                      if (event.key === "2") {
                        submitShortcut = () => currentHideForms()[1]?.requestSubmit();
                      }
                      if (event.key === "3") {
                        submitShortcut = () => currentVetoThreadForms()[0]?.requestSubmit();
                      }
                    }
                    if (!event.altKey && event.key === "1") {
                      submitShortcut = () => currentVoteForms()[0]?.requestSubmit();
                    }
                    if (!event.altKey && event.key === "2") {
                      submitShortcut = () => currentVoteForms()[1]?.requestSubmit();
                    }
                    if (!submitShortcut) return;
                    event.preventDefault();
                    event.stopPropagation();
                    if (isTransitioning || isActionPending) return;
                    submitShortcut();
                  });

                  window.addEventListener("resize", () => {
                    for (const layer of [currentLayer, lookaheadLayer]) {
                      applyArenaSplit(layer);
                      for (const image of layerImages(layer)) fitArenaImage(image);
                    }
                  });

                  if (salvagePreparedCurrent()) {
                    void refillLookaheadFromReserve();
                  }

                  if (!layerImages(currentLayer).length) {
                    reportArenaAnomaly("arena_current_layer_empty", {
                      layer_role: "current",
                    });
                  }

                  primeLayer(currentLayer, { progressive: true }).then((primed) => {
                    if (!primed) {
                      reportArenaAnomaly("arena_current_layer_empty", {
                        layer_role: "current",
                      });
                      window.location.assign(arenaRootHref());
                      return;
                    }
                    void bootLookahead();
                  });
                })();
                "#,
            ))
        }
    }
}

pub(super) fn design_language_font_path(font_name: &str) -> anyhow::Result<PathBuf> {
    const SWARM_DESIGN_LANGUAGE_ROOT: &str =
        "/home/main/programming/projects/swarm.moe/design-language";

    let file_name = match font_name {
        "oxanium-400.woff2" | "oxanium-600.woff2" | "share-tech-mono-400.woff2" => font_name,
        _ => bail!("unknown font asset {font_name}"),
    };
    Ok(FsPath::new(SWARM_DESIGN_LANGUAGE_ROOT)
        .join("assets")
        .join("fonts")
        .join(file_name))
}

#[derive(Debug, Clone, Copy)]
pub(super) enum NavPage {
    Arena,
    Board,
    Facemash,
    Identities,
    Explore,
    Triad,
    Vocab,
}
