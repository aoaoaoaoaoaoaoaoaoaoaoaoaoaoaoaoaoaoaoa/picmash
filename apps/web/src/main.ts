import "./styles.css";

import { mountExplore } from "./explore";
import { mountTriad } from "./triad";

async function main(): Promise<void> {
  const root = ensureRoot();
  const pathname = window.location.pathname;
  if (pathname.startsWith("/triad")) {
    await mountTriad(root);
    return;
  }
  if (pathname.startsWith("/explore")) {
    await mountExplore(root);
    return;
  }
  root.textContent = "unsupported frontend route";
}

function ensureRoot(): HTMLElement {
  const existing = document.getElementById("picmash-app");
  if (existing) {
    return existing;
  }
  const root = document.createElement("main");
  root.id = "picmash-app";
  document.body.append(root);
  return root;
}

void main();
