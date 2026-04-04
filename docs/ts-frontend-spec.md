# TS Frontend Spec

This document describes the current frontend architecture. It is not speculative.

## Scope

The TypeScript frontend owns the hot interactive surfaces:

- `/explore`
- `/triad`

The Rust backend remains authoritative for:

- corpus ingest
- SQLite persistence
- DINO embedding management
- ranking and similarity models
- image rendition serving
- route-level boot/runtime status

`/arena` and `/board` remain server-rendered.

## Boundary

Rust types are the single source of truth for the browser contract.

- request/response DTOs live in `crates/picmash-app/src/api.rs`
- TypeScript declarations are generated via `ts-rs`
- the export binary is `crates/picmash-app/src/bin/export-ts.rs`

The browser speaks plain HTTP + JSON to Rust. Images remain ordinary URL-addressed bytes.

## Backend Shape

Rust serves:

- static frontend assets from `crates/picmash-app/assets/web`
- host shells for `/explore` and `/triad`
- JSON APIs for bootstrap and mutations

Important rule:

- map layout and panel/triad resolution are separate paths
- panel and triad endpoints must not compute explore layout

## Frontend Shape

The frontend is plain TypeScript + Vite, no framework.

- `apps/web/src/main.ts` mounts by route
- `apps/web/src/explore.ts` owns the canvas map, camera, hit-testing, sidebar state, and visible-first thumbnail loading
- `apps/web/src/triad.ts` owns the full-screen three-image contrastive trainer
- `apps/web/src/api.ts` is the only transport layer

The explore map is client-owned:

- camera state is local
- hit-testing is local
- rendering is canvas-based
- thumbnail loads are demand-driven

## Build

Frontend build is mandatory for `check` and `install`.

- `check.py` runs `npm ci` and `npm run build` before Rust verification/install
- Vite outputs directly into the Rust asset directory

## Non-Goals

This architecture intentionally does not do the following:

- no wasm layer
- no frontend framework
- no OpenAPI client generation
- no server-rendered explore/triad markup
- no backward-compat path aliases for the removed explore URL shape
