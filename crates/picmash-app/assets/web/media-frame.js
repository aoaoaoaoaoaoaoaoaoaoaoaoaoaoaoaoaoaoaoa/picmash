const TOOL_COPY = {
    "rotate-left": { label: "L", title: "rotate left", extraClass: "" },
    "rotate-right": { label: "R", title: "rotate right", extraClass: "" },
    heart: { label: "♥", title: "bless image", extraClass: " ts-tool-heart" },
    hide: { label: "X", title: "hide image", extraClass: " ts-tool-danger" },
};
export function overlayTool(kind, action, active = false) {
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
export function overlayToolRail(className, specs) {
    const rail = document.createElement("div");
    rail.className = className;
    rail.append(...specs.map(({ kind, action, active }) => overlayTool(kind, action, active)));
    return rail;
}
export function overlayMeta(className, lines) {
    const meta = document.createElement("div");
    meta.className = className;
    meta.append(...lines.map(({ text, emphasized }) => {
        const line = document.createElement("div");
        line.className = emphasized ? "ts-overlay-meta-line emphasized" : "ts-overlay-meta-line";
        line.textContent = text;
        return line;
    }));
    return meta;
}
export function overlayChipRail(className, chips) {
    const rail = document.createElement("div");
    rail.className = className;
    rail.append(...chips.map(({ text, tone }) => {
        const chip = document.createElement("span");
        chip.className = tone ? `ts-quality-chip ts-quality-chip-${tone}` : "ts-quality-chip";
        chip.textContent = text;
        return chip;
    }));
    return rail;
}
function domainDisplay(label) {
    return label === "real" ? "3D" : "2D";
}
function domainTitle(label, domain) {
    const cues = [];
    if (domain.manualLabel === label) {
        cues.push(`manual ${domainDisplay(label)}`);
    }
    if (domain.predictedLabel === label && domain.predictedPercent != null) {
        cues.push(`model ${domainDisplay(label)} ${domain.predictedPercent}%`);
    }
    return cues.length === 0 ? domainDisplay(label) : `${domainDisplay(label)} · ${cues.join(" · ")}`;
}
export function overlayDomainButtons(spec) {
    return ["real", "anime"].map((label) => {
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
