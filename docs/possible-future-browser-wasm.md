# Possible Future Idea: Fully Client-Side Browser Build

## Status

This document is speculative only.

It is **not** part of any active plan, roadmap, milestone, or implementation
queue. No agent should treat it as scheduled work, implied direction, or hidden
priority. It exists only to preserve a plausible future idea so it is not lost.

## Idea

One possible future direction is a fully client-side browser build of picmash:

- ranking and session logic compiled to WebAssembly
- corpus metadata, embeddings, thumbnails, and model state persisted locally in
  the browser
- image access granted by the user through browser file-system permissions
- DINO-style embeddings computed in-browser and cached by content hash

The intent would be a local-first artifact with no required server process.

## Feasibility Read

There does not appear to be a hard blocker.

The main constraints are practical rather than conceptual:

- browser file access is permission-gated and does not behave like stable native
  filesystem paths
- embedding inference in-browser is feasible, but would likely rely on ONNX
  Runtime Web / WebGPU or a similar JS-adjacent runtime rather than a purely
  Rust-native stack
- first-run setup would be slower because the browser would need to download
  model weights and compute embeddings locally
- background indexing would be weaker than the current native app model because
  the browser only works while the page is open and within browser permission
  rules

So the idea seems viable, but not trivial.

## Plausible Shape

If this were ever pursued, the austere architecture would probably be:

- Rust core compiled to `wasm32` for:
  - ranking
  - session state
  - similarity learning
  - event-log mutation
- browser shell for:
  - UI
  - file-handle acquisition
  - storage plumbing
  - model runtime integration
- local persistence via:
  - SQLite WASM with OPFS if possible, or
  - IndexedDB if that path proves cleaner
- embedding runtime via:
  - ONNX-exported DINOv2
  - WebGPU when available
  - WASM fallback when not
- aggressive caching keyed by content hash for:
  - embeddings
  - thumbnails
  - normalized renditions
  - learned local model state

## Important Consequences

Several invariants would change:

- asset identity should be content-hash-first, not path-first
- the app would remember directory handles, not absolute filesystem paths
- any heavy inference must be treated as lazy, resumable, and cached
- a thin JS interop layer would probably be unavoidable even if the core
  remained Rust

## Why This Is Not Active

This path would be a separate product shape, not a small incremental extension
of the current app.

It would force real architectural choices about:

- browser-only storage semantics
- permissions UX
- model packaging and runtime
- offline behavior
- cross-browser support policy

Those are large enough that they should be taken on only by explicit choice,
not by drift.

## If It Is Ever Revived

The correct first question would not be "how do we port the current app?"

It would be:

- what is the smallest browser-native local-first product worth building?

Only after that should implementation questions be reopened.
