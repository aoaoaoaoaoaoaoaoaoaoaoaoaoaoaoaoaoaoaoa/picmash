# Possible Future Idea: TypeScript Frontend Without WebAssembly

## Status

This document is a design sketch only.

It is not an implementation request, not an active milestone, and not an
implied commitment. It exists to preserve one clean architectural option:
replace the heavy browser interaction surfaces with a TypeScript frontend while
keeping the Rust server as the local backend.

WebAssembly is explicitly out of scope for this note.

## Thesis

The present frontend bottleneck is architectural rather than linguistic.

`explore` in particular wants:

- a client-owned camera
- a client-owned render loop
- client-side hit-testing
- asynchronous thumbnail fill-in
- near-zero synchronous layout work during drag and zoom

That does not require WebAssembly. It requires abandoning the DOM-heavy
interaction model for the hot surfaces.

## Division of Labor

The austere split is:

- Rust remains the local application and data plane.
- TypeScript becomes the browser interaction and rendering plane.

Rust should continue to own:

- corpus ingest and rescan
- content-addressed asset identity
- SQLite persistence
- ranking and session learning
- DINO embedding extraction
- thumbnail / rendition generation
- stable routing for deep links

TypeScript should own:

- `explore` rendering
- `triad` rendering
- camera state
- pointer / keyboard interaction
- local hover, selection, and hit-testing
- image loading priority
- optimistic client-side state transitions where latency matters

## Why Not Rewrite Everything

`arena` and `board` are not the pathological surfaces.

They can remain server-rendered until there is a concrete reason to replace
them. The first frontend extraction should be narrow and violent:

- keep `arena` and `board` as they are
- replace only `explore` and `triad` with a TS client

That keeps the blast radius small and preserves the useful local-Rust model.

## Runtime Shape

One plausible layout:

- `crates/picmash-app`
  - continues to serve assets, JSON endpoints, and the existing pages
- `apps/web`
  - Vite-based TypeScript frontend
  - mounted only for `explore` and `triad`

The browser shell can still be served by the Rust app:

- `GET /explore` returns a minimal HTML host page
- that host page loads a compiled JS bundle
- the bundle fetches JSON and image renditions from the same local server

No separate dev server is required in production.

## Backend Contract

The Rust server should stop emitting big HTML blobs for `explore`.

Instead it should expose a narrow JSON contract. The TS side needs only a few
surfaces.

### Explore bootstrap

`GET /api/explore/bootstrap?mode=raw|learned`

Returns:

- corpus/session metadata
- current map mode
- current triad ids
- current focus id if any
- point records:
  - `asset_id`
  - `plot_x`
  - `plot_y`
  - optional `latent_x[5]` for learned mode
  - cheap summary stats
  - precomputed tooltip strings only if genuinely useful
- URLs for thumbnail and preview renditions

### Explore selection

`GET /api/explore/focus/{asset_id}?mode=...&triad=...`

Returns:

- focus card data
- nearest neighbors
- distances in the true metric space
- current triad summary

This is the current sidebar payload, but formalized as JSON instead of injected
HTML.

### Triad bootstrap

`GET /api/triad/current`

Returns:

- current triad asset ids
- image URLs
- any lightweight triad metadata needed for display

### Triad train

`POST /api/triad/train`

Body:

- triad ids
- selected closest pair

Returns:

- updated triad
- any revised learned-mode layout token if necessary

### Rescan and status

Keep:

- `POST /rescan`
- `GET /__status`

Those already fit the local-app model.

## Rendering Model

The `explore` surface should become a single client-owned scene:

- one canvas or WebGL scene for the map
- one overlay layer for HUD chrome
- one sidebar DOM column for focus details

The key invariant is:

- camera motion must not cause per-node layout work

The map is just points in a 2D field. The client should hold those points in a
compact array and render them in one pass.

For the first TS implementation, the sensible options are:

- `canvas` 2D if the point count remains modest
- `PixiJS` if we want a stronger 2D scene abstraction immediately

The important thing is not framework fashion. It is escaping the DOM as the
camera substrate.

## Thumbnail Strategy

There are two different image surfaces:

- map thumbnails
- sidebar / triad images

Map thumbnails should be treated as opportunistic adornment:

- render point shells immediately
- load only visible or near-visible thumbs
- keep a tiny client LRU for decoded image bitmaps
- never let image decode stall camera motion

Sidebar and triad images can be loaded at higher priority because they are few
and semantically central.

The Rust backend already has the right underlying concept: stable rendition
URLs keyed by asset and kind.

## Interaction Model

The client should own all hot interactions locally:

- pan
- zoom
- hover
- focus selection
- double-click reset
- keyboard navigation within the map

Server round-trips should only happen for semantic mutations:

- train triad
- rotate
- hide
- board nudges if exposed in TS later
- rescan

Even focus changes should feel local. The TS app should:

- compute nearest hit locally
- mark focus locally
- then fetch sidebar detail asynchronously

That way a click never feels like a page navigation.

## Routing Shape

There is no need for a SPA empire.

The minimal route shape is:

- `/arena` and `/board`
  - keep server-rendered
- `/explore`
  - TS host page
- `/triad`
  - TS host page

Deep links should remain stable and literal:

- `/explore?mode=raw&focus=...&triad=a,b,c`
- `/triad?a=...&b=...&c=...`

The client reads those parameters on boot, and mutations update the URL via the
history API without a full page reload.

## State Model

The TypeScript app should keep a strict split between:

- immutable dataset state
  - points
  - rendition URLs
  - embedding-space coordinates
- camera state
  - pan
  - zoom
- UI state
  - hover
  - focus
  - sidebar load phase
- mutation state
  - in-flight train / rotate / hide actions

Do not let server payload shape leak straight into mutable UI mush.

## Performance Target

The correct standard is:

- drag and zoom remain smooth even if every image is still blank

That means:

- interaction loop never depends on network
- interaction loop never depends on image decode
- interaction loop never depends on DOM relayout of every point

If those are not true, the architecture has already failed.

## Migration Shape

The clean migration is four passes.

### 1. Formalize JSON

Add JSON endpoints for:

- explore bootstrap
- focus sidebar
- triad bootstrap
- triad training

Do not change behavior yet.

### 2. Replace explore only

Ship a TS `explore` host page using the existing backend data and asset routes.

Keep triad server-rendered for the moment.

### 3. Replace triad

Move triad to the same TS frontend once the asset and mutation plumbing are
proven.

### 4. Reevaluate arena and board

Only after living with the split should we decide whether those pages deserve
the same treatment.

## What Must Not Happen

Several bad futures should be rejected early:

- no duplicated ranking logic in TS
- no separate browser-side source of truth for sessions
- no second asset identity scheme
- no HTML-over-the-wire pseudo-API for the new frontend
- no premature WebAssembly cargo cult

The TS side is a rendering and interaction engine, not a rival application.

## Executive Summary

The TS-only solution is:

- keep Rust as the local data, model, and rendition server
- replace only `explore` and `triad` with a client-owned TS renderer
- formalize the server contract as JSON plus image URLs
- move pan/zoom/hit-testing/image prioritization entirely into the browser
- use canvas or a 2D rendering library to eliminate DOM camera thrash

That is the smallest serious move that addresses the current frontend pain
without dragging in WebAssembly for ideological reasons.
