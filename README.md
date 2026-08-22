# Picmash Engine

Picmash is now a headless Rust library for learning a person's preferences over
a local image collection. This repository contains the engine to be grafted
into The Poolrooms; it deliberately contains no server, browser client,
installer, background service, or model downloader.

The engine owns four things:

- exact image and occurrence identity;
- immutable, globally ordered observations with their presentation context;
- reversible collection state such as favorites and visibility;
- explicit Bradley-Terry preference snapshots with chronological holdout
  metrics.

It does not own a universal image-quality score. Similarity triads remain raw
observations until a representation and learner earn authority through an
evaluation. Face detection, attractiveness, external discovery, and the old
unified-quality models were rejected rather than carried into the graft.

## Boundary

`picmash-engine` exposes one synchronous `Engine` handle. The host chooses
the database location, scans a collection, opens a judgment session, requests a
comparison, records a command, and rebuilds preferences explicitly:

```rust
use picmash_engine::{CommandId, Engine};

let engine = Engine::open(state_database)?;
let scan = engine.scan(image_directory)?;
let session = engine.start_session(scan.collection_id, "poolrooms-comparison-v1")?;
let prompt = engine.propose_comparison(&session.id)?;
engine.record_comparison(
    &prompt.id,
    &prompt.left.asset_id,
    &CommandId::fresh(),
    None,
)?;
let snapshot = engine.rebuild_preferences(scan.collection_id)?;
```

The future application must place mutable state under the platform state
directory, not in its source checkout. On Linux that means an XDG state path
such as `$XDG_STATE_HOME/the-poolrooms/picmash.db`. Corpus scans are read-only.

## Legacy Data

`Engine::import_legacy` reads a former web-app database in SQLite read-only
mode. It imports exact local assets, occurrences, sessions, local pairwise
comparisons, threshold judgments, reversible favorite events, and raw
similarity triads. It refuses to promote learned scores, unversioned
embeddings, face state, or comparisons contaminated by the old external-import
path. Ambiguous observations are counted in the import report.

## Verification

Run `./check.py`. The canonical gate checks formatting, denies every Clippy
warning, and runs the sparse engine-law suite.
