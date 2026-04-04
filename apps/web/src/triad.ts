import {
  fetchTriadBootstrap,
  heartTriadAsset,
  hideTriadAsset,
  readTriadRoute,
  rotateTriadAsset,
  setTriadAssetDomain,
  trainTriad,
  writeTriadRoute,
} from "./api";
import { overlayChipRail, overlayDomainButtons, overlayMeta, overlayToolRail } from "./media-frame";
import type {
  AssetQualitySummaryDto,
  SimilarityChoice,
  TriadAssetDto,
  TriadBootstrapDto,
  TriadHandleDto,
} from "./contracts";

export async function mountTriad(root: HTMLElement): Promise<void> {
  const app = new TriadClient(root);
  await app.boot();
}

class TriadClient {
  private readonly shell = document.createElement("section");
  private readonly stage = document.createElement("div");
  private readonly rail = document.createElement("div");
  private readonly prompt = document.createElement("div");
  private readonly choiceRow = document.createElement("div");
  private bootstrap: TriadBootstrapDto | null = null;
  private triad: TriadHandleDto | null = readTriadRoute();
  private requestToken = 0;

  constructor(private readonly root: HTMLElement) {
    this.root.replaceChildren();
    this.root.classList.add("ts-frontend-root");
    this.shell.className = "ts-shell triad-shell";
    this.stage.className = "ts-triad-stage";
    this.rail.className = "ts-triad-rail";
    this.prompt.className = "ts-triad-prompt";
    this.choiceRow.className = "ts-triad-choice-row";
    this.rail.append(this.prompt, this.choiceRow);
    this.shell.append(this.stage, this.rail);
    this.root.append(this.shell);
  }

  async boot(): Promise<void> {
    window.addEventListener("popstate", this.handlePopState);
    window.addEventListener("keydown", this.handleKeydown);
    await this.reload("replace");
  }

  private readonly handlePopState = (): void => {
    this.triad = readTriadRoute();
    void this.reload("replace");
  };

  private readonly handleKeydown = (event: KeyboardEvent): void => {
    if (event.repeat) {
      return;
    }
    if (event.key === "1") {
      void this.choose("ab");
    } else if (event.key === "2") {
      void this.choose("ac");
    } else if (event.key === "3") {
      void this.choose("bc");
    }
  };

  private async reload(historyMode: "push" | "replace"): Promise<void> {
    await this.transact("loading triad", historyMode, () => fetchTriadBootstrap(this.triad));
  }

  private applyBootstrap(bootstrap: TriadBootstrapDto, historyMode: "push" | "replace"): void {
    this.bootstrap = bootstrap;
    this.triad = bootstrap.triad;
    writeTriadRoute(this.triad, historyMode);
    this.render();
  }

  private render(): void {
    this.stage.replaceChildren();
    this.choiceRow.replaceChildren();
    this.prompt.textContent = "";
    if (!this.bootstrap) {
      return;
    }
    if (this.bootstrap.emptyNote || !this.bootstrap.triad || this.bootstrap.assets.length !== 3) {
      const note = document.createElement("div");
      note.className = "ts-triad-empty";
      note.textContent = this.bootstrap.emptyNote ?? "Need three embedded images.";
      this.stage.append(note);
      return;
    }

    const labels = ["A", "B", "C"] as const;
    for (const [index, asset] of this.bootstrap.assets.entries()) {
      this.stage.append(this.panel(labels[index] ?? "?", asset));
    }

    this.prompt.textContent = "pick the closest pair";
    this.choiceRow.append(
      this.choiceButton("1", "A/B", "ab"),
      this.choiceButton("2", "A/C", "ac"),
      this.choiceButton("3", "B/C", "bc"),
    );
  }

  private panel(label: string, asset: TriadAssetDto): HTMLElement {
    const section = document.createElement("article");
    section.className = "ts-media-frame ts-triad-panel";
    const badge = document.createElement("span");
    badge.className = "ts-triad-label";
    badge.textContent = label;
    const image = document.createElement("img");
    image.className = "ts-triad-image";
    image.alt = asset.name;
    image.src = asset.fullSrc;
    image.loading = "eager";
    image.decoding = "async";
    image.draggable = false;
    section.append(
      image,
      badge,
      this.withDomainTools(
        overlayToolRail("ts-overlay-tools", [
        { kind: "rotate-left", action: () => void this.rotate(asset.assetId, -1) },
        { kind: "rotate-right", action: () => void this.rotate(asset.assetId, 1) },
        {
          kind: "heart",
          active: asset.hearted,
          action: () => void this.heart(asset.assetId),
        },
        { kind: "hide", action: () => void this.hide(asset.assetId) },
        ]),
        asset,
      ),
      overlayChipRail("ts-quality-chips", qualityChipSpecs(asset.quality)),
      overlayMeta("ts-overlay-meta", [
        { text: asset.name, emphasized: true },
        { text: `${asset.winCount} wins / ${asset.compareCount} duels` },
      ]),
    );
    return section;
  }

  private choiceButton(
    shortcut: string,
    label: string,
    choice: SimilarityChoice,
  ): HTMLButtonElement {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-triad-choice";
    button.textContent = `${shortcut} · ${label}`;
    button.title = label;
    button.addEventListener("click", () => {
      void this.choose(choice);
    });
    return button;
  }

  private async choose(choice: SimilarityChoice): Promise<void> {
    if (!this.triad) {
      return;
    }
    await this.transact("forging next triad", "push", () => trainTriad(this.triad!, choice));
  }

  private async rotate(assetId: string, direction: -1 | 1): Promise<void> {
    await this.transact("rotating image", "replace", () =>
      rotateTriadAsset(this.triad, assetId, direction),
    );
  }

  private async hide(assetId: string): Promise<void> {
    await this.transact("hiding image", "replace", () => hideTriadAsset(this.triad, assetId));
  }

  private async heart(assetId: string): Promise<void> {
    await this.transact("marking favorite", "replace", () =>
      heartTriadAsset(this.triad, assetId, true),
    );
  }

  private async setDomain(assetId: string, label: "real" | "anime"): Promise<void> {
    await this.transact("updating domain label", "replace", () =>
      setTriadAssetDomain(this.triad, assetId, label),
    );
  }

  private withDomainTools(rail: HTMLDivElement, asset: TriadAssetDto): HTMLDivElement {
    rail.append(
      ...overlayDomainButtons({
        domain: asset.domain,
        action: (label) => void this.setDomain(asset.assetId, label),
      }),
    );
    return rail;
  }

  private renderFailure(error: unknown): void {
    const message = error instanceof Error ? error.message : "triad fault";
    this.stage.replaceChildren();
    const note = document.createElement("div");
    note.className = "ts-triad-empty";
    note.textContent = message;
    this.stage.append(note);
  }

  private async transact(
    status: string,
    historyMode: "push" | "replace",
    task: () => Promise<TriadBootstrapDto>,
  ): Promise<void> {
    const token = ++this.requestToken;
    try {
      const bootstrap = await task();
      await preloadTriadAssets(bootstrap);
      if (token !== this.requestToken) {
        return;
      }
      this.applyBootstrap(bootstrap, historyMode);
    } catch (error) {
      if (token === this.requestToken) {
        this.renderFailure(error);
      }
    }
  }
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

async function preloadTriadAssets(bootstrap: TriadBootstrapDto): Promise<void> {
  if (!bootstrap.triad || bootstrap.assets.length === 0) {
    return;
  }
  await Promise.all(bootstrap.assets.map(({ fullSrc }) => preloadImage(fullSrc)));
}

function preloadImage(src: string): Promise<void> {
  return new Promise((resolve) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve();
    image.onerror = () => resolve();
    image.src = src;
    if (image.complete) {
      resolve();
    }
  });
}
