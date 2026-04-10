# Single-Writer SQLite Design

## Objective

The north star is ruthless UX responsiveness.

No click-path action should ever be delayed behind background maintenance because
some broad sweep happened to hold an in-process write lease. SQLite already
wants a single writer. The application should stop fighting that truth with
ad-hoc lock discipline and instead make the single writer explicit.

## Current Failure Mode

Today the app still exposes a closure-based write gate:

- [with_db_write_gate()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L131)
- [with_locked_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L140)
- [with_fresh_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L151)

That abstraction is too weak. It permits arbitrary code to squat on the write
lease. The canonical live offender is
[devour_bootstrap_maintenance()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/lifecycle.rs#L203),
which still performs a whole bootstrap sweep under
[lifecycle.rs:206](/home/main/programming/projects/picmash/crates/picmash-app/src/app/lifecycle.rs#L206).

This is not a type-level guarantee against lag. It is merely a runtime mutex
with better logging.

There is a second, equally important failure mode on the read side. Foreground
paths still hydrate session state through a shared store mutex, and
[session_field()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/arena.rs#L1143)
currently rebuilds a large in-memory view from many queries on demand. A
perfect writer actor would still leave visible click latency if this stays as
is. The design therefore has two inseparable planks:

- single-writer serialization for all mutations
- cached, incrementally maintained read-side session state with no shared
  read mutex

## Design

Replace the write gate with a true single-writer actor.

- one dedicated writer task
- one write-capable SQLite connection owned only by that task
- all other code uses independent read-only or read-mostly connections under
  WAL
- application code submits typed write commands, never arbitrary closures
- heavy work happens before submission
- writer commands are small, deterministic, SQL-only commits

The writer is the sole place where mutating transactions happen.

At the same time, replace the shared `self.store: Mutex<Store>` read choke
with either:

- per-call read connections, or
- a small read pool

and treat `SessionField` as a cached projection updated by committed write
deltas, not a structure rehydrated from scratch on every click.

## Core Types

The app should revolve around three layers:

1. `Prepared*`

Prepared values are pure data produced outside the writer.

Examples:

- `PreparedRemoteIdentity`
- `PreparedRemoteEmbedding`
- `PreparedImportedOutcome`
- `PreparedFaceScanBatch`
- `PreparedQualityFeatureBatch`

These may contain:

- ids
- timestamps
- already-computed vectors
- serialized payloads
- decoded identity facts

These must not contain:

- closures
- filesystem scans
- ONNX inference
- network fetches
- mutable stores

2. `WriteCmd`

The only mutating surface.

Examples:

- `WriteCmd::PersistRemoteIdentity(PreparedRemoteIdentity)`
- `WriteCmd::PersistExternalEmbedding(PreparedExternalEmbedding)`
- `WriteCmd::SealImportedOutcome(PreparedImportedOutcome)`
- `WriteCmd::AppendExternalEvent(PreparedExternalEvent)`
- `WriteCmd::ApplyMaintenanceBatch(MaintenanceBatchCmd)`
- `WriteCmd::PersistHarvestBatch(PreparedHarvestBatch)`
- `WriteCmd::AdvanceReplay(PreparedReplayAdvance)`
- `WriteCmd::IngestCorpusBatch(PreparedCorpusBatch)`

3. `WriteResult`

Conflict handling is typed, not generic retry sludge.

Examples:

- `Applied(T)`
- `NoOp`
- `Stale`
- `NeedsReprepare`

4. `ReadModelDelta`

Every committed command may also yield a delta for the hot in-memory read
model.

Examples:

- `ReadModelDelta::HeartBlessed { asset_id }`
- `ReadModelDelta::SubsourceLockSet { lock }`
- `ReadModelDelta::SelectionNoted { asset_id, at }`
- `ReadModelDelta::ReplayAdvanced { cache_patch }`

The app should update its cached read projection from these deltas rather than
reloading the world.

## Read-Side Doctrine

The user feels end-to-end action latency, not merely write-lock latency.

Therefore the design must explicitly kill three read-side epicycles:

### No shared read mutex

The app should not serialize all reads through one long-lived `Store` behind a
Rust mutex. SQLite WAL already allows concurrent readers. The process should
use separate read connections and let SQLite do the right thing.

### Hot `SessionField`

`SessionField` should be a cached in-memory projection.

It may be:

- eagerly built once at session boot
- patched by `ReadModelDelta` after each successful write
- invalidated and lazily rebuilt only on rare structural transitions

It must not be rebuilt from many queries on ordinary arena clicks.

### No inference or file I/O on render-critical paths

Foreground render should not synchronously:

- compute missing face embeddings
- compute missing quality features
- decode large files just to answer a view request

Those belong to warming and background maintenance. A missing derivative should
degrade the card, not stall the click.

## Command Semantics

Commands should fall into a few explicit classes.

### Idempotent upserts

Examples:

- persist identity
- persist embedding
- persist quality features
- set resolved asset id

These should generally return `Applied` or `NoOp`.

### Monotone appends

Examples:

- append duel event
- append unary external event
- append heart event

These should not need retry logic; they are append-only.

### Compare-and-set / cursor advancement

Examples:

- replay cursor advance
- maintenance claim/complete transitions
- any commit that depends on an expected prior revision

These may return `Stale` and require reprepare.

### Resumable maintenance batches

Examples:

- backfill next identity batch
- resolve next remote alias batch
- persist next quality batch

These should carry cursors explicitly and return:

- `advanced_to`
- `applied_count`
- `more`

That is not retry logic. It is chunked progress.

### Atomic multi-row commits

One `WriteCmd` may update many related rows, but the command boundary is the
transaction boundary. The writer should not silently bundle unrelated commands
together.

Examples:

- one duel vote command may append the event, update session memory, and commit
  cache deltas atomically
- one harvest batch command may upsert a bounded set of streams/items in one
  transaction

But the command must be semantically single and small. The writer is not a
license for broad opportunistic batching.

## Conflict Model

With a true single writer, in-process write/write contention disappears.

What remains:

### Stale preparation

A command may be prepared from slightly old read state, then submitted after
something else changed.

This should be handled by typed outcomes:

- `NoOp` if the work is already reflected
- `Stale` if the command depended on a version/cursor that no longer matches
- `NeedsReprepare` if the final lightweight validation failed

Do not build a generic optimistic retry engine.

### External SQLite contention

If some outside process writes the DB and SQLite returns `BUSY`, the writer task
may perform a tiny bounded retry with jitter. This belongs in exactly one place:
the writer loop, not spread across call sites.

### Read/modify/write invariants

If a command needs a fresh check, the writer should do the final lightweight
read+check+commit atomically. The heavy payload still gets prepared outside.

## Channel Semantics

The writer is a command processor, not an unbounded junk drawer.

- the submission channel should be bounded
- foreground critical commands may await capacity briefly
- background maintenance must yield or reschedule instead of backpressuring the
  UI indefinitely
- command ordering is FIFO at the writer boundary
- commands that require an immediate result should await their specific
  `WriteResult`
- non-critical bookkeeping may be fire-and-forget only if the caller has an
  explicit reconciliation path

The app should make these classes explicit instead of informally treating all
writes as alike.

## Maintenance Doctrine

Maintenance must stop being “devour a whole category under one lease.”

Every maintenance job should be expressed as:

1. read candidate ids outside the writer
2. compute expensive payloads outside the writer
3. submit one small `WriteCmd`
4. repeat if `more`

This applies especially to:

- bootstrap maintenance
- corpus face scan backfill
- corpus face recognition backfill
- corpus quality feature backfill
- remote face/identity/quality warming
- quality replay persistence
- corpus ingest persistence
- external harvest persistence

No maintenance command should hold the writer for more than a small bounded
transaction. The goal is not “usually fast”; it is structural impossibility of
minute-long click starvation.

Harvest work in particular should converge toward:

1. fetch and decode source payloads outside the writer
2. derive identity / embedding / warm-state facts outside the writer
3. submit one bounded harvest batch command

not hundreds of tiny gated upserts nor one monolithic sweep.

## UI Policy

Foreground UX should classify mutations into two buckets.

### Synchronous tiny commits

Examples:

- record arena vote
- append reject/keep/heart event
- set or clear subsource lock
- note selection

These can await the writer directly if the command is tiny.

### Optimistic commits

Examples:

- actions already visually hidden by lookahead buffers
- non-critical bookkeeping

These may update UI immediately and let the writer trail, provided the UI model
has an explicit reconciliation path.

The writer architecture does not force optimism; it merely makes synchronous
writes predictably cheap enough that optimism becomes a product choice instead
of a defensive necessity.

The same doctrine applies to expensive model maintenance:

- never retrain an oracle under the writer
- never run ONNX inference under the writer
- never let a view request trigger heavyweight derivative computation as a
  hidden side effect

## Migration Staircase

The safest path is staged.

### Stage 0

Normalize the existing surface before migration.

- collapse the false distinction between
  [with_locked_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L140)
  and
  [with_fresh_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L151)
- inventory every current write-site by semantic command family
- inventory every render-critical path that still performs file I/O or
  inference

### Stage 1

Introduce the writer task and `WriteCmd` algebra without changing behavior.

- add writer runtime
- route a few tiny writes through it
- keep the old gate only as a temporary adapter

### Stage 1.5

Kill the read-side choke points.

- remove the shared `Mutex<Store>` read bottleneck
- introduce cached `SessionField`
- patch the cache from committed write deltas
- allow rare cold rebuilds only behind explicit invalidation

### Stage 2

Delete closure-based public write APIs from `AppState`.

- remove or quarantine
  [with_db_write_gate()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L131),
  [with_locked_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L140),
  and
  [with_fresh_store_write()](/home/main/programming/projects/picmash/crates/picmash-app/src/app/gate.rs#L151)
- leave only typed writer submission

### Stage 3

Convert all click-path writes first.

This gives immediate UX wins and proves the model.

### Stage 4

Convert maintenance and bootstrap work into chunked resumable commands.

This is where the long-holder class dies for good.

### Stage 4.5

Move heavyweight derivative completion out of foreground render paths.

- face embedding completion
- quality feature completion
- any remaining decode/probe side effects on card load

At this point click latency should reflect only view assembly plus a tiny
writer round-trip.

### Stage 5

Delete the old gate implementation entirely.

At that point, write serialization is not an app-wide convention. It is an
architectural fact.

## Success Criteria

- no arbitrary closure can hold the write lease
- no background sweep can block foreground clicks for seconds
- no shared read mutex serializes foreground reads
- no ordinary click rehydrates the full `SessionField`
- write transactions become observable, typed, and auditable
- “retry” is rare and typed, not ambient
- maintenance is resumable and chunked
- responsiveness improves because the system stops permitting monolithic writes
  and monolithic read hydration

## Anti-Goals

- no generic “retry any failed write” framework
- no hidden fallback path that reintroduces broad in-gate work
- no dual world where some writes use commands and others still use raw closures
- no dual world where writes are disciplined but reads still funnel through a
  global mutex
- no claim that WAL alone solves in-process writer starvation

## Bottom Line

The current write gate is a mutex with better telemetry.

The correct design is a typed single-writer effect algebra:

- prepare outside
- commit inside
- tiny commands only
- typed stale/no-op outcomes
- resumable maintenance

Anything weaker is another local patch on the same structural flaw.
