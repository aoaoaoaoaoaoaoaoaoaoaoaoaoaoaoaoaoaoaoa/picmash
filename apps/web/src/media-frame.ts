import type { AssetDomainLabel, AssetDomainSummaryDto } from "./contracts";

export type OverlayToolKind = "rotate-left" | "rotate-right" | "heart" | "hide";

export type OverlayToolSpec = {
  kind: OverlayToolKind;
  active?: boolean;
  action: () => void;
};

export type OverlayMetaLine = {
  text: string;
  emphasized?: boolean;
};

export type OverlayChipSpec = {
  text: string;
  tone?: "asset" | "baseline" | "semantic" | "vibe" | "technical" | "face";
};

export type OverlayDomainSpec = {
  domain: AssetDomainSummaryDto;
  action: (label: AssetDomainLabel) => void;
};

const TOOL_COPY: Record<
  OverlayToolKind,
  { label: string; title: string; extraClass: string }
> = {
  "rotate-left": { label: "L", title: "rotate left", extraClass: "" },
  "rotate-right": { label: "R", title: "rotate right", extraClass: "" },
  heart: { label: "♥", title: "bless image", extraClass: " ts-tool-heart" },
  hide: { label: "X", title: "hide image", extraClass: " ts-tool-danger" },
};

export function overlayTool(
  kind: OverlayToolKind,
  action: () => void,
  active = false,
): HTMLButtonElement {
  const button = document.createElement("button");
  const copy = TOOL_COPY[kind];
  button.type = "button";
  button.className = `ts-tool${copy.extraClass}`;
  if (active) {
    button.classList.add("active");
  }
  button.textContent = copy.label;
  button.title = copy.title;
  button.setAttribute("aria-label", copy.title);
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    action();
  });
  return button;
}

export function overlayToolRail(
  className: string,
  specs: readonly OverlayToolSpec[],
): HTMLDivElement {
  const rail = document.createElement("div");
  rail.className = className;
  rail.append(...specs.map(({ kind, action, active }) => overlayTool(kind, action, active)));
  return rail;
}

export function overlayMeta(
  className: string,
  lines: readonly OverlayMetaLine[],
): HTMLDivElement {
  const meta = document.createElement("div");
  meta.className = className;
  meta.append(
    ...lines.map(({ text, emphasized }) => {
      const line = document.createElement("div");
      line.className = emphasized ? "ts-overlay-meta-line emphasized" : "ts-overlay-meta-line";
      line.textContent = text;
      return line;
    }),
  );
  return meta;
}

export function overlayChipRail(
  className: string,
  chips: readonly OverlayChipSpec[],
): HTMLDivElement {
  const rail = document.createElement("div");
  rail.className = className;
  rail.append(
    ...chips.map(({ text, tone }) => {
      const chip = document.createElement("span");
      chip.className = tone ? `ts-quality-chip ts-quality-chip-${tone}` : "ts-quality-chip";
      chip.textContent = text;
      return chip;
    }),
  );
  return rail;
}

function domainDisplay(label: AssetDomainLabel): string {
  return label === "real" ? "3D" : "2D";
}

function domainTitle(label: AssetDomainLabel, domain: AssetDomainSummaryDto): string {
  const cues: string[] = [];
  if (domain.manualLabel === label) {
    cues.push(`manual ${domainDisplay(label)}`);
  }
  if (domain.predictedLabel === label && domain.predictedPercent != null) {
    cues.push(`model ${domainDisplay(label)} ${domain.predictedPercent}%`);
  }
  return cues.length === 0 ? domainDisplay(label) : `${domainDisplay(label)} · ${cues.join(" · ")}`;
}

export function overlayDomainButtons(spec: OverlayDomainSpec): HTMLButtonElement[] {
  return (["real", "anime"] as const).map((label) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `ts-tool ts-domain-tool ${label === "real" ? "ts-domain-tool-3d" : "ts-domain-tool-2d"}`;
    if (spec.domain.predictedLabel === label) {
      button.classList.add("predicted");
    }
    if (spec.domain.manualLabel === label) {
      button.classList.add("manual", "active");
    }
    button.textContent = domainDisplay(label);
    button.title = domainTitle(label, spec.domain);
    button.setAttribute("aria-label", domainTitle(label, spec.domain));
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      spec.action(label);
    });
    return button;
  });
}
