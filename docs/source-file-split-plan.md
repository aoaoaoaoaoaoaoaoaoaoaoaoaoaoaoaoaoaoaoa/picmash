# Source File Split Status

The oversized Rust walls are gone, and the workspace now enforces the ratchet.

## Shipped Structure

`store.rs` is split into:

- `src/store.rs`
- `src/store/external.rs`
- `src/store/faces.rs`
- `src/store/schema.rs`
- `src/store/tests.rs`

`app.rs` is split into:

- `src/app.rs`
- `src/app/external.rs`
- `src/app/explore.rs`
- `src/app/facemash.rs`

`web.rs` is split into:

- `src/web.rs`
- `src/web/arena.rs`
- `src/web/board.rs`
- `src/web/explore.rs`
- `src/web/facemash.rs`
- `src/web/media.rs`

## Ratchet

The workspace now uses `workspace.metadata.rust-starter`, and
`workspace.metadata.rust-starter.source_files.max_lines = 2500` is enforced by
`check.py` before any cargo work runs.

The repo also now pins the toolchain in `rust-toolchain.toml`.

## Current Line Counts

Largest files after the split:

- `crates/picmash-app/src/app.rs`: `2485`
- `crates/picmash-app/src/model.rs`: `1924`
- `crates/picmash-app/src/store.rs`: `1878`
- `crates/picmash-app/src/web.rs`: `1525`

Everything is under the `2500`-line ceiling.

## Intent

This is no longer a plan. It is the current baseline. Any future growth that
breaks the cap should be answered with another split, not an exclusion.
