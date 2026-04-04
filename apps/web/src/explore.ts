import {
  fetchExploreBootstrap,
  fetchExploreSelection,
  heartExploreAsset,
  hideExploreAsset,
  readExploreRouteState,
  rotateExploreAsset,
  setExploreAssetDomain,
  writeExploreRoute,
  type ExploreRouteState,
} from "./api";
import { overlayChipRail, overlayDomainButtons, overlayMeta, overlayToolRail } from "./media-frame";
import type {
  AssetQualitySummaryDto,
  ExploreBootstrapDto,
  ExplorePointDto,
  ExploreSelectionDto,
  FocusAssetDto,
  TriadHandleDto,
} from "./contracts";

const TILE_SIZE = 60;
const IMAGE_CONCURRENCY = 8;
const ZOOM_MIN = 0.04;
const ZOOM_MAX = 36;
const DEFAULT_SCALE = 1.18;
const OVERSCAN = 96;

type CameraState = {
  x: number;
  y: number;
  scale: number;
};

type Rect = {
  width: number;
  height: number;
};

type VisiblePoint = {
  point: ExplorePointDto;
  screenX: number;
  screenY: number;
};

type BitmapRecord =
  | { state: "loading" }
  | { state: "error" }
  | { state: "ready"; bitmap: ImageBitmap };

class BitmapCache {
  private readonly records = new Map<string, BitmapRecord>();
  private readonly queue: string[] = [];
  private readonly pointMap: Map<string, ExplorePointDto>;
  private active = 0;

  constructor(
    points: ExplorePointDto[],
    private readonly onReady: () => void,
  ) {
    this.pointMap = new Map(points.map((point) => [point.assetId, point]));
  }

  rewrite(points: ExplorePointDto[]): void {
    this.pointMap.clear();
    for (const point of points) {
      this.pointMap.set(point.assetId, point);
    }
  }

  enqueue(assetIds: string[]): void {
    for (const assetId of assetIds) {
      if (this.records.has(assetId) || this.queue.includes(assetId)) {
        continue;
      }
      this.queue.push(assetId);
    }
    this.pump();
  }

  lookup(assetId: string): ImageBitmap | null {
    const record = this.records.get(assetId);
    return record?.state === "ready" ? record.bitmap : null;
  }

  private pump(): void {
    while (this.active < IMAGE_CONCURRENCY && this.queue.length > 0) {
      const assetId = this.queue.shift();
      if (!assetId || this.records.has(assetId)) {
        continue;
      }
      const point = this.pointMap.get(assetId);
      if (!point) {
        continue;
      }
      this.records.set(assetId, { state: "loading" });
      this.active += 1;
      void this.load(point);
    }
  }

  private async load(point: ExplorePointDto): Promise<void> {
    try {
      const response = await fetch(point.thumbSrc, {
        cache: "force-cache",
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob);
      this.records.set(point.assetId, { state: "ready", bitmap });
      this.onReady();
    } catch {
      this.records.set(point.assetId, { state: "error" });
    } finally {
      this.active -= 1;
      this.pump();
    }
  }
}

export async function mountExplore(root: HTMLElement): Promise<void> {
  const app = new ExploreClient(root);
  await app.boot();
}

class ExploreClient {
  private readonly state = readExploreRouteState();
  private readonly canvas = document.createElement("canvas");
  private readonly shell = document.createElement("section");
  private readonly viewport = document.createElement("div");
  private readonly sidebar = document.createElement("aside");
  private readonly modeRail = document.createElement("div");
  private readonly actionRail = document.createElement("div");
  private readonly cameraRail = document.createElement("div");
  private readonly statusLine = document.createElement("div");
  private readonly canvasWrap = document.createElement("div");
  private readonly hud = document.createElement("div");
  private readonly triadButton = document.createElement("a");
  private readonly rerollButton = document.createElement("button");
  private readonly zoomInButton = document.createElement("button");
  private readonly zoomOutButton = document.createElement("button");
  private readonly zoomResetButton = document.createElement("button");
  private readonly lightbox = document.createElement("section");
  private bootstrap: ExploreBootstrapDto | null = null;
  private selection: ExploreSelectionDto | null = null;
  private previewAsset: FocusAssetDto | null = null;
  private readonly camera: CameraState = { x: 0, y: 0, scale: DEFAULT_SCALE };
  private bitmapCache = new BitmapCache([], () => this.invalidate());
  private frame = 0;
  private rect: Rect = { width: 1, height: 1 };
  private hoverId: string | null = null;
  private isDragging = false;
  private dragPointerId: number | null = null;
  private dragOrigin = { x: 0, y: 0, cameraX: 0, cameraY: 0 };
  private visiblePoints: VisiblePoint[] = [];
  private loading = false;
  private selectionRequestToken = 0;

  constructor(private readonly root: HTMLElement) {
    this.root.replaceChildren();
    this.root.classList.add("ts-frontend-root");

    this.shell.className = "ts-shell explore-shell";
    this.viewport.className = "ts-explore-viewport";
    this.canvasWrap.className = "ts-explore-canvas-wrap";
    this.canvas.className = "ts-explore-canvas";
    this.sidebar.className = "ts-explore-sidebar";
    this.hud.className = "ts-explore-hud";
    this.modeRail.className = "ts-tool-row ts-explore-mode-rail";
    this.actionRail.className = "ts-tool-row ts-explore-action-rail";
    this.cameraRail.className = "ts-tool-row ts-explore-camera-rail";
    this.statusLine.className = "ts-explore-status";
    this.lightbox.className = "ts-lightbox";
    this.lightbox.hidden = true;
    this.triadButton.className = "ts-tool";
    this.triadButton.textContent = "triad";
    this.rerollButton.className = "ts-tool";
    this.rerollButton.type = "button";
    this.rerollButton.textContent = "new";
    this.zoomInButton.className = "ts-tool";
    this.zoomInButton.type = "button";
    this.zoomInButton.textContent = "+";
    this.zoomOutButton.className = "ts-tool";
    this.zoomOutButton.type = "button";
    this.zoomOutButton.textContent = "−";
    this.zoomResetButton.className = "ts-tool";
    this.zoomResetButton.type = "button";
    this.zoomResetButton.textContent = "reset";

    this.viewport.append(this.canvasWrap, this.hud);
    this.canvasWrap.append(this.canvas);
    this.actionRail.append(this.triadButton, this.rerollButton);
    this.cameraRail.append(this.zoomOutButton, this.zoomInButton, this.zoomResetButton);
    this.hud.append(this.modeRail, this.actionRail, this.cameraRail);
    this.shell.append(this.viewport, this.sidebar);
    this.root.append(this.shell, this.lightbox, this.statusLine);

    this.installModeButtons();
    this.installCanvasHandlers();
    this.installHudHandlers();
    this.installLightboxHandlers();
  }

  async boot(): Promise<void> {
    window.addEventListener("resize", this.handleResize);
    window.addEventListener("popstate", this.handlePopState);
    this.resizeCanvas();
    await this.reload("replace");
  }

  private readonly handleResize = (): void => {
    this.resizeCanvas();
    this.invalidate();
  };

  private readonly handlePopState = (): void => {
    const fresh = readExploreRouteState();
    this.state.mode = fresh.mode;
    this.state.focusId = fresh.focusId;
    this.state.triad = fresh.triad;
    void this.reload("replace");
  };

  private installModeButtons(): void {
    const raw = document.createElement("button");
    raw.className = "ts-tool";
    raw.type = "button";
    raw.textContent = "raw";
    raw.dataset.mode = "raw";
    const learned = document.createElement("button");
    learned.className = "ts-tool";
    learned.type = "button";
    learned.textContent = "learned";
    learned.dataset.mode = "learned";
    this.modeRail.append(raw, learned);
    this.modeRail.addEventListener("click", (event) => {
      const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>(
        "button[data-mode]",
      );
      if (!button) {
        return;
      }
      const mode = button.dataset.mode === "learned" ? "learned" : "raw";
      if (mode === this.state.mode) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      this.state.mode = mode;
      writeExploreRoute(this.state, "replace");
      this.syncHud();
      void this.reload("replace");
    });
  }

  private installHudHandlers(): void {
    const stopHudPropagation = (event: Event): void => {
      event.stopPropagation();
    };
    for (const type of ["pointerdown", "pointerup", "click", "dblclick"]) {
      this.hud.addEventListener(type, stopHudPropagation);
    }
    this.hud.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        event.stopPropagation();
      },
      { passive: false },
    );
    this.rerollButton.addEventListener("click", () => {
      this.state.triad = null;
      writeExploreRoute(this.state, "replace");
      void this.reload("replace");
    });
    this.zoomInButton.addEventListener("click", () => {
      this.zoomAt(this.rect.width * 0.5, this.rect.height * 0.5, 1.22);
    });
    this.zoomOutButton.addEventListener("click", () => {
      this.zoomAt(this.rect.width * 0.5, this.rect.height * 0.5, 1 / 1.22);
    });
    this.zoomResetButton.addEventListener("click", () => {
      this.resetCamera();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.previewAsset) {
        event.preventDefault();
        this.closePreview();
        return;
      }
      if (event.key === "+" || event.key === "=") {
        this.zoomAt(this.rect.width * 0.5, this.rect.height * 0.5, 1.22);
      } else if (event.key === "-") {
        this.zoomAt(this.rect.width * 0.5, this.rect.height * 0.5, 1 / 1.22);
      } else if (event.key === "0") {
        this.resetCamera();
      }
    });
  }

  private installLightboxHandlers(): void {
    this.lightbox.addEventListener("click", (event) => {
      if (event.target === this.lightbox || (event.target as HTMLElement | null)?.closest(".ts-lightbox-close")) {
        this.closePreview();
      }
    });
  }

  private installCanvasHandlers(): void {
    this.viewport.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
        this.zoomAt(event.offsetX, event.offsetY, factor);
      },
      { passive: false },
    );

    this.viewport.addEventListener("pointerdown", (event) => {
      const focus = this.pickPoint(event.offsetX, event.offsetY);
      if (!focus) {
        this.isDragging = true;
        this.dragPointerId = event.pointerId;
        this.dragOrigin = {
          x: event.clientX,
          y: event.clientY,
          cameraX: this.camera.x,
          cameraY: this.camera.y,
        };
        this.viewport.setPointerCapture(event.pointerId);
        this.viewport.classList.add("is-dragging");
      }
    });

    this.viewport.addEventListener("pointermove", (event) => {
      if (this.isDragging && this.dragPointerId === event.pointerId) {
        this.camera.x = this.dragOrigin.cameraX + (event.clientX - this.dragOrigin.x);
        this.camera.y = this.dragOrigin.cameraY + (event.clientY - this.dragOrigin.y);
        this.invalidate();
        return;
      }
      const nextHover = this.pickPoint(event.offsetX, event.offsetY)?.assetId ?? null;
      if (nextHover !== this.hoverId) {
        this.hoverId = nextHover;
        this.viewport.style.cursor = nextHover ? "pointer" : "grab";
        this.invalidate();
      }
    });

    const endDrag = (event: PointerEvent): void => {
      if (this.dragPointerId === event.pointerId) {
        this.isDragging = false;
        this.dragPointerId = null;
        this.viewport.classList.remove("is-dragging");
        if (this.viewport.hasPointerCapture(event.pointerId)) {
          this.viewport.releasePointerCapture(event.pointerId);
        }
      }
    };
    this.viewport.addEventListener("pointerup", endDrag);
    this.viewport.addEventListener("pointercancel", endDrag);
    this.viewport.addEventListener("dblclick", (event) => {
      if (!this.pickPoint(event.offsetX, event.offsetY)) {
        this.resetCamera();
      }
    });
    this.viewport.addEventListener("click", (event) => {
      if (this.isDragging) {
        return;
      }
      const point = this.pickPoint(event.offsetX, event.offsetY);
      if (point) {
        void this.focus(point.assetId, "push");
      }
    });
  }

  private async reload(historyMode: "push" | "replace"): Promise<void> {
    this.loading = true;
    this.selectionRequestToken += 1;
    this.renderStatus("loading map");
    try {
      this.applyBootstrap(await fetchExploreBootstrap(this.state), historyMode);
    } catch (error) {
      this.renderFailure(error);
    } finally {
      this.loading = false;
    }
  }

  private applyBootstrap(bootstrap: ExploreBootstrapDto, historyMode: "push" | "replace"): void {
    const sanitized = sanitizeBootstrap(bootstrap);
    this.bootstrap = sanitized;
    this.selection = sanitized.selection;
    this.state.mode = sanitized.mode;
    this.state.triad = sanitized.triad;
    this.state.focusId =
      sanitized.selection?.focus.assetId ?? sanitized.focusId ?? this.state.focusId;
    writeExploreRoute(this.state, historyMode);
    this.bitmapCache.rewrite(sanitized.points);
    this.syncHud();
    this.renderSidebar();
    this.reconcilePreview();
    this.renderStatus(this.describeStatus());
    this.invalidate();
  }

  private reconcilePreview(): void {
    if (!this.previewAsset) {
      return;
    }
    const next =
      this.selection?.focus.assetId === this.previewAsset.assetId ? this.selection.focus : null;
    if (next) {
      this.previewAsset = next;
      this.renderPreview();
    } else {
      this.closePreview();
    }
  }

  private syncHud(): void {
    for (const button of this.modeRail.querySelectorAll<HTMLButtonElement>("button[data-mode]")) {
      button.classList.toggle("active", button.dataset.mode === this.state.mode);
    }
    const triad = this.state.triad
      ? [this.state.triad.assetAId, this.state.triad.assetBId, this.state.triad.assetCId].join(",")
      : null;
    this.triadButton.href = triad ? `/triad?triad=${encodeURIComponent(triad)}` : "/triad";
  }

  private renderSidebar(): void {
    this.sidebar.replaceChildren();
    if (!this.bootstrap) {
      return;
    }
    if (this.bootstrap.emptyNote) {
      this.sidebar.append(this.noteBlock(this.bootstrap.emptyNote));
      return;
    }
    if (!this.selection) {
      this.sidebar.append(this.noteBlock("Pick an image on the map."));
      return;
    }

    this.sidebar.append(
      this.sidebarSection("focus", this.selection.focus.name, this.focusCard(this.selection.focus)),
    );

    const neighborGrid = document.createElement("div");
    neighborGrid.className = "ts-neighbor-grid";
    for (const neighbor of this.selection.neighbors) {
      neighborGrid.append(this.neighborCard(neighbor.asset, neighbor.distance));
    }
    this.sidebar.append(
      this.sidebarSection(
        this.state.mode,
        `${this.selection.neighbors.length} neighbors`,
        neighborGrid,
      ),
    );

    const launch = document.createElement("div");
    launch.className = "ts-inline-actions";
    const openTriad = document.createElement("a");
    openTriad.className = "ts-tool";
    openTriad.href = this.triadHref();
    openTriad.textContent = "triad";
    const nextTriad = document.createElement("button");
    nextTriad.className = "ts-tool";
    nextTriad.type = "button";
    nextTriad.textContent = "new";
    nextTriad.addEventListener("click", () => {
      this.state.triad = null;
      writeExploreRoute(this.state, "replace");
      void this.reload("replace");
    });
    launch.append(openTriad, nextTriad);
    this.sidebar.append(this.sidebarSection("contrastive", "closest pair", launch));
  }

  private focusCard(asset: FocusAssetDto): HTMLElement {
    const media = document.createElement("div");
    media.className = "ts-media-frame ts-focus-media";
    const image = document.createElement("img");
    image.className = "ts-focus-image";
    image.alt = asset.name;
    image.src = asset.previewSrc;
    image.draggable = false;
    image.title = "open fullscreen";
    image.addEventListener("click", () => {
      this.openPreview(asset);
    });
    media.append(
      image,
      this.withDomainTools(
        overlayToolRail("ts-overlay-tools", [
        { kind: "rotate-left", action: () => void this.rotateFocused(asset.assetId, -1) },
        { kind: "rotate-right", action: () => void this.rotateFocused(asset.assetId, 1) },
        {
          kind: "heart",
          active: asset.hearted,
          action: () => void this.heartFocused(asset.assetId),
        },
        { kind: "hide", action: () => void this.hideFocused(asset.assetId) },
        ]),
        asset,
      ),
      overlayChipRail("ts-quality-chips", qualityChipSpecs(asset.quality)),
      overlayMeta("ts-overlay-meta", [
        { text: asset.name, emphasized: true },
        { text: `global ${signed(asset.globalScore)}` },
        { text: `${asset.winCount} wins / ${asset.compareCount} duels` },
      ]),
    );
    return media;
  }

  private neighborCard(asset: FocusAssetDto, distance: number): HTMLElement {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "ts-neighbor-card ts-media-frame";
    card.addEventListener("click", () => {
      void this.focus(asset.assetId, "push");
    });
    const image = document.createElement("img");
    image.className = "ts-neighbor-image";
    image.alt = asset.name;
    image.src = asset.previewSrc;
    image.loading = "lazy";
    image.decoding = "async";
    const badge = document.createElement("span");
    badge.className = "ts-distance-badge";
    badge.textContent = distance.toFixed(2);
    card.append(
      image,
      badge,
      overlayChipRail("ts-quality-chips", qualityChipSpecs(asset.quality)),
      overlayMeta("ts-overlay-meta", [
        { text: asset.name, emphasized: true },
        { text: `distance ${distance.toFixed(2)}` },
        { text: `${asset.winCount} wins / ${asset.compareCount} duels` },
      ]),
    );
    return card;
  }

  private async rotateFocused(assetId: string, direction: -1 | 1): Promise<void> {
    try {
      const bootstrap = await rotateExploreAsset(this.state, assetId, direction);
      this.applyBootstrap(bootstrap, "replace");
    } catch (error) {
      this.renderFailure(error);
    }
  }

  private async hideFocused(assetId: string): Promise<void> {
    try {
      const bootstrap = await hideExploreAsset(this.state, assetId);
      if (this.state.focusId === assetId) {
        this.state.focusId = null;
      }
      if (this.previewAsset?.assetId === assetId) {
        this.closePreview();
      }
      this.applyBootstrap(bootstrap, "replace");
    } catch (error) {
      this.renderFailure(error);
    }
  }

  private async heartFocused(assetId: string): Promise<void> {
    try {
      const bootstrap = await heartExploreAsset(this.state, assetId, true);
      this.applyBootstrap(bootstrap, "replace");
    } catch (error) {
      this.renderFailure(error);
    }
  }

  private async setDomain(assetId: string, label: "real" | "anime"): Promise<void> {
    try {
      const bootstrap = await setExploreAssetDomain(this.state, assetId, label);
      this.applyBootstrap(bootstrap, "replace");
    } catch (error) {
      this.renderFailure(error);
    }
  }

  private withDomainTools(rail: HTMLDivElement, asset: FocusAssetDto): HTMLDivElement {
    rail.append(
      ...overlayDomainButtons({
        domain: asset.domain,
        action: (label) => void this.setDomain(asset.assetId, label),
      }),
    );
    return rail;
  }

  private openPreview(asset: FocusAssetDto): void {
    this.previewAsset = asset;
    this.renderPreview();
  }

  private closePreview(): void {
    this.previewAsset = null;
    this.renderPreview();
  }

  private renderPreview(): void {
    if (!this.previewAsset) {
      this.lightbox.hidden = true;
      this.lightbox.replaceChildren();
      document.body.classList.remove("ts-lightbox-open");
      return;
    }

    const asset = this.previewAsset;
    const shell = document.createElement("div");
    shell.className = "ts-lightbox-shell ts-media-frame";

    const image = document.createElement("img");
    image.className = "ts-lightbox-image";
    image.alt = asset.name;
    image.src = asset.fullSrc;
    image.loading = "eager";
    image.decoding = "sync";
    image.draggable = false;

    const close = document.createElement("button");
    close.type = "button";
    close.className = "ts-tool ts-lightbox-close";
    close.textContent = "X";
    close.title = "close preview";
    close.setAttribute("aria-label", "close preview");
    close.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.closePreview();
    });

    shell.append(
      image,
      this.withDomainTools(
        overlayToolRail("ts-overlay-tools", [
        { kind: "rotate-left", action: () => void this.rotateFocused(asset.assetId, -1) },
        { kind: "rotate-right", action: () => void this.rotateFocused(asset.assetId, 1) },
        {
          kind: "heart",
          active: asset.hearted,
          action: () => void this.heartFocused(asset.assetId),
        },
        { kind: "hide", action: () => void this.hideFocused(asset.assetId) },
        ]),
        asset,
      ),
      overlayChipRail("ts-quality-chips", qualityChipSpecs(asset.quality)),
      close,
      overlayMeta("ts-overlay-meta", [
        { text: asset.name, emphasized: true },
        { text: `global ${signed(asset.globalScore)}` },
        { text: `${asset.winCount} wins / ${asset.compareCount} duels` },
      ]),
    );

    this.lightbox.hidden = false;
    this.lightbox.replaceChildren(shell);
    document.body.classList.add("ts-lightbox-open");
  }

  private async focus(assetId: string, historyMode: "push" | "replace"): Promise<void> {
    this.state.focusId = assetId;
    writeExploreRoute(this.state, historyMode);
    this.renderSidebar();
    this.invalidate();
    const token = ++this.selectionRequestToken;
    try {
      const response = await fetchExploreSelection(assetId, this.state);
      if (token !== this.selectionRequestToken) {
        return;
      }
      this.selection = response.selection;
      this.state.focusId = response.focusId;
      writeExploreRoute(this.state, "replace");
      this.renderSidebar();
      this.invalidate();
    } catch (error) {
      this.renderFailure(error);
    }
  }

  private resizeCanvas(): void {
    const rect = this.viewport.getBoundingClientRect();
    this.rect = {
      width: Math.max(1, Math.floor(rect.width)),
      height: Math.max(1, Math.floor(rect.height)),
    };
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.floor(this.rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(this.rect.height * dpr));
    this.canvas.style.width = `${this.rect.width}px`;
    this.canvas.style.height = `${this.rect.height}px`;
  }

  private invalidate(): void {
    if (this.frame !== 0) {
      return;
    }
    this.frame = window.requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  private draw(): void {
    const context = this.canvas.getContext("2d");
    if (!context) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, this.rect.width, this.rect.height);
    context.fillStyle = "#09090b";
    context.fillRect(0, 0, this.rect.width, this.rect.height);

    const points = this.bootstrap?.points ?? [];
    const visible: VisiblePoint[] = [];
    for (const point of points) {
      if (!Number.isFinite(point.plotX) || !Number.isFinite(point.plotY)) {
        continue;
      }
      const screenX = this.rect.width * 0.5 + this.camera.x + ((point.plotX - 0.5) * this.camera.scale * this.rect.width);
      const screenY = this.rect.height * 0.5 + this.camera.y + ((point.plotY - 0.5) * this.camera.scale * this.rect.height);
      if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
        continue;
      }
      if (
        screenX < -OVERSCAN ||
        screenY < -OVERSCAN ||
        screenX > this.rect.width + OVERSCAN ||
        screenY > this.rect.height + OVERSCAN
      ) {
        continue;
      }
      visible.push({ point, screenX, screenY });
    }
    this.visiblePoints = visible;
    this.bitmapCache.enqueue(
      visible
        .slice()
        .sort((left, right) => {
          const leftScore = distanceToCenterSq(left.screenX, left.screenY, this.rect);
          const rightScore = distanceToCenterSq(right.screenX, right.screenY, this.rect);
          return leftScore - rightScore;
        })
        .map(({ point }) => point.assetId),
    );

    for (const entry of visible) {
      this.drawPoint(context, entry);
    }
  }

  private drawPoint(context: CanvasRenderingContext2D, entry: VisiblePoint): void {
    const half = TILE_SIZE * 0.5;
    const left = entry.screenX - half;
    const top = entry.screenY - half;
    const isFocus = entry.point.assetId === this.state.focusId;
    const isHover = entry.point.assetId === this.hoverId;
    context.fillStyle = "#000000";
    context.fillRect(left, top, TILE_SIZE, TILE_SIZE);
    const bitmap = this.bitmapCache.lookup(entry.point.assetId);
    if (bitmap) {
      context.drawImage(bitmap, left + 1, top + 1, TILE_SIZE - 2, TILE_SIZE - 2);
    }
    context.lineWidth = isFocus ? 2 : 1;
    context.strokeStyle = isFocus ? "#f6f6f7" : isHover ? "#8fe6ff" : "#41424a";
    context.strokeRect(left + 0.5, top + 0.5, TILE_SIZE - 1, TILE_SIZE - 1);
  }

  private pickPoint(screenX: number, screenY: number): ExplorePointDto | null {
    for (let index = this.visiblePoints.length - 1; index >= 0; index -= 1) {
      const entry = this.visiblePoints[index];
      const half = TILE_SIZE * 0.5;
      if (
        Math.abs(entry.screenX - screenX) <= half &&
        Math.abs(entry.screenY - screenY) <= half
      ) {
        return entry.point;
      }
    }
    return null;
  }

  private zoomAt(screenX: number, screenY: number, factor: number): void {
    const previous = this.screenToWorld(screenX, screenY);
    this.camera.scale = clamp(this.camera.scale * factor, ZOOM_MIN, ZOOM_MAX);
    this.camera.x = screenX - this.rect.width * 0.5 - ((previous.x - 0.5) * this.camera.scale * this.rect.width);
    this.camera.y = screenY - this.rect.height * 0.5 - ((previous.y - 0.5) * this.camera.scale * this.rect.height);
    this.invalidate();
  }

  private screenToWorld(screenX: number, screenY: number): { x: number; y: number } {
    return {
      x:
        0.5 +
        (screenX - this.rect.width * 0.5 - this.camera.x) /
          Math.max(1e-6, this.camera.scale * this.rect.width),
      y:
        0.5 +
        (screenY - this.rect.height * 0.5 - this.camera.y) /
          Math.max(1e-6, this.camera.scale * this.rect.height),
    };
  }

  private resetCamera(): void {
    this.camera.x = 0;
    this.camera.y = 0;
    this.camera.scale = DEFAULT_SCALE;
    this.invalidate();
  }

  private renderFailure(error: unknown): void {
    const message = error instanceof Error ? error.message : "frontend fault";
    this.sidebar.replaceChildren(this.noteBlock(message));
    this.renderStatus(message);
  }

  private renderStatus(message: string): void {
    this.statusLine.textContent = message;
  }

  private describeStatus(): string {
    if (!this.bootstrap) {
      return "loading";
    }
    const focus = this.selection?.focus.name ?? "none";
    return `${this.bootstrap.points.length} points · ${this.state.mode} · focus ${focus}`;
  }

  private sectionHead(kicker: string, title: string): HTMLElement {
    const header = document.createElement("header");
    header.className = "ts-section-head";
    const left = document.createElement("span");
    left.className = "ts-kicker";
    left.textContent = kicker;
    const right = document.createElement("span");
    right.className = "ts-title";
    right.textContent = title;
    header.append(left, right);
    return header;
  }

  private sidebarSection(kicker: string, title: string, ...children: HTMLElement[]): HTMLElement {
    const section = document.createElement("section");
    section.className = "ts-explore-section";
    section.append(this.sectionHead(kicker, title), ...children);
    return section;
  }

  private noteBlock(message: string): HTMLElement {
    const note = document.createElement("section");
    note.className = "ts-explore-note";
    note.append(this.sectionHead("status", "note"));
    const text = document.createElement("p");
    text.className = "ts-note";
    text.textContent = message;
    note.append(text);
    return note;
  }

  private triadHref(): string {
    const triad = this.state.triad
      ? [this.state.triad.assetAId, this.state.triad.assetBId, this.state.triad.assetCId].join(",")
      : null;
    return triad ? `/triad?triad=${encodeURIComponent(triad)}` : "/triad";
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function signed(value: number): string {
  return value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function posterior(summary: { mean: number; sigma: number }): string {
  return `${formatScalar(summary.mean)} ± ${formatScalar(summary.sigma)}`;
}

function formatScalar(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 100) {
    return value.toFixed(0);
  }
  if (abs >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function qualityChipSpecs(summary: AssetQualitySummaryDto) {
  return [
    { text: `q ${posterior(summary.asset)}`, tone: "asset" as const },
    { text: `b ${posterior(summary.baseline)}`, tone: "baseline" as const },
    ...(summary.semantic
      ? [{ text: `m ${posterior(summary.semantic)}`, tone: "semantic" as const }]
      : []),
    ...(summary.vibe ? [{ text: `s ${posterior(summary.vibe)}`, tone: "vibe" as const }] : []),
    ...(summary.technical
      ? [{ text: `k ${posterior(summary.technical)}`, tone: "technical" as const }]
      : []),
    ...(summary.face ? [{ text: `f ${posterior(summary.face)}`, tone: "face" as const }] : []),
  ];
}

function sanitizeBootstrap(bootstrap: ExploreBootstrapDto): ExploreBootstrapDto {
  return {
    ...bootstrap,
    points: bootstrap.points
      .filter(
        (point) =>
          Number.isFinite(point.plotX) &&
          Number.isFinite(point.plotY) &&
          point.latent.every((axis) => Number.isFinite(axis)),
      )
      .map((point) => ({
        ...point,
        plotX: clamp(point.plotX, 0, 1),
        plotY: clamp(point.plotY, 0, 1),
      })),
  };
}

function distanceToCenterSq(x: number, y: number, rect: Rect): number {
  const dx = x - rect.width * 0.5;
  const dy = y - rect.height * 0.5;
  return dx * dx + dy * dy;
}
