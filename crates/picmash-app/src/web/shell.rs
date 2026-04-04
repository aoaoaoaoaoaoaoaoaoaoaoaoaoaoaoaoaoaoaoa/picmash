use super::*;

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
            (Self::LockThread, true) => "tool mini",
            (Self::LockThread, false) => "tool",
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

pub(super) fn script_block() -> Markup {
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
