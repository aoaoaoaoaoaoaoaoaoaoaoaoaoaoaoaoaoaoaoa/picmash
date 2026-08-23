# Remote Acquisition

## Names

A **discovery** is bounded source metadata plus an immutable origin snapshot. It
owns no downloaded media.

A **prepared candidate** is a validated payload in the XDG cache. An **offer**
is the sole prepared candidate currently shown against a local asset. Fetching,
prepared candidates, and the offer constitute the **reservoir**.

A **promotion commitment** is an accepted offer plus its exact judgment,
including the anchor's occurrence, render, and rotation, persisted before the
chamber advances. The bounded **archive queue** turns that commitment into a
collection occurrence by decoding the payload, applying its presentation
rotation, losslessly re-encoding the canonical raster as JPEG XL, verifying
render identity, and atomically writing beneath `.picmash-imported/`.

`not_x` promotes either winner of a remote comparison: choosing the local image
still means the challenger was not rejected. `hearted` promotes only a remote
winner or an explicit favorite. Reject (`X`) retires one candidate. Reject
Stream (`Alt+J`) retires its thread.

## Configuration

F2 exposes the acquisition switch, sampling chance, reservoir cap, and the
configuration-file path. Source topology and filters remain explicit TOML:

```toml
[remote]
enabled = true
sample_probability = 0.25
reservoir_capacity = 4
metadata_capacity = 64

[[remote.sources]]
weight = 1.0
import_policy = "not_x"
scan_interval_seconds = 120

[remote.sources.upstream]
type = "4chan_board"

[remote.sources.upstream.settings]
board = "s"

[remote.sources.upstream.settings.content]
allow_nsfw = true
allow_video = false

[remote.sources.upstream.settings.harvest]
catalog_threads = 24
thread_fetches_per_scan = 2

[remote.sources.upstream.settings.filters]
min_shortest_edge = 800
max_download_bytes = 8388608
max_pixels = 40000000
```

`local_directory` replaces the upstream settings with `root`, `recurse`, and
the same `filters` table. Configuration admits at most 32 sources. Zero weight
disables one source without discarding its definition.

## Machine

Each positive-weight source occupies one catalog phase:

```text
Due -> Cataloging -> Waiting(deadline)
                   -> Backoff(finite deadline, strikes)
```

Catalog success replaces only that source's metadata partition. Failure enters
exponential backoff capped at five minutes. Every successful scan uses the
configured interval. Reconfiguration increments an epoch; completions release
their unique permits before the epoch is examined. Stale completions cannot
introduce metadata or prepared media into the current epoch. An in-flight fetch
remains a counted reservoir member until that completion retires it.

Each candidate occupies one persistent phase:

```text
Discovered -> Fetching -> Prepared -> Offered
     ^            |          |          |
     +------------+----------+          +-> Rejected
                                      \----> Promoting -> Promoted
```

`Fetching -> Discovered` is the recoverable failure transition. Process startup
performs the same recovery for interrupted fetches and removes cache files not
owned by a prepared row. `Offered -> Promoting` first commits the judgment and
archive target in SQLite and reserves the judgment's engine observation, then
releases the comparison chamber. The reservation fixes global evidence order;
archive completion fills it rather than recording a later observation. Graceful
exit cancels the encoder but preserves both durable facts; startup reconciles
and resubmits them. A stream veto cannot revoke an earlier promotion commitment.
Terminal candidates are never resurrected by a later catalog scan.

Catalog, fetch, and offer selection use weighted least-service scheduling. For
source `i`, the scheduler minimizes `servicesᵢ / weightᵢ`, comparing ratios by
exact cross multiplication. Deterministic source order breaks ties. Before a
counter can approach overflow, the scheduler subtracts the same whole number
of weighted service rounds from every currently eligible source; all compared
ratios shift by one constant, so ordering is unchanged.

## Bounds

Let:

- `R ∈ [1, 8]` be reservoir capacity;
- `M ∈ [8, 256]` be metadata capacity;
- `S ≤ 32` be the positive-weight source count, with configuration requiring
  `S ≤ M`;
- `B ≤ 512 MiB` be the largest configured payload limit;
- `P ≤ 64,000,000` be the largest configured pixel limit;
- `A = 8` be the fixed archive-queue capacity.

Define `Q = |Fetching| + |Prepared| + |Offered|`. The only transition that adds
to `Q` is `Discovered -> Fetching`, whose guard is `Q < R`. Fetch completion
replaces one fetching member with at most one prepared member. Offering replaces
one prepared member with the offer; rejection removes it, while archival
commitment transfers it out of `Q`. Therefore `Q ≤ R` is inductive.

Let `J = |Promoting|`. Only `Offered -> Promoting` adds to `J`, guarded by
`J < A`; completion removes one member. Thus `J ≤ A`. The archive mailbox and
completion channel each have capacity `A`, and exactly one encoder process may
run. The combined cache frontier is at most `R + A` payloads plus one atomic
staging output.

One unique fetch permit exists, so at most one media request and one validating
decode run concurrently. One unique catalog permit exists. The permits survive
epoch changes and are consumed exactly once by their typed completions; a stale
effect therefore cannot overlap a replacement effect in the same lane.

Metadata is partitioned at `floor(M / S)` discoveries per source. Because
`S ≤ M`, every enabled source owns at least one slot, and total retained
metadata is at most `S floor(M / S) ≤ M`. 4chan scans admit at most 64
discoveries, inspect at most 64 catalog threads, and fetch at most four thread
documents. Each JSON body is capped at 8 MiB. Local scans inspect at most
100,000 directory entries and retain only a 64-element selection heap.
Each successful catalog transaction retires and deletes superseded unjudged
metadata for that source. Persistent frontier size is therefore bounded by 64
items per configured source plus the reservoir and archive queue. Rejected,
promoted, and duel records grow only with user judgments.

Payload length and encoded dimensions are checked before full decode. Thus one
fetch retains at most `B` payload bytes and decodes at most `P` pixels. The
catalog/fetch result channel holds at most the two extant lane completions, and
the UI event channel holds 64 events.

Promotion invokes the lossless JPEG XL encoder at maximum effort 10 on its own
lane. No archival deadline sacrifices compression density. The engine worker
continues serving comparisons, browsing, persistence, and remote acquisition.
Application retirement checks cancellation every 250 ms, kills the child, and
leaves the durable commitment and reserved evidence order for restart.

## Liveness

Assume an enabled source remains reachable, its configured interval expires,
the application remains running, and the user eventually retires displayed
offers. Catalog failure backoff is finite. Once due, weighted least-service
scheduling selects every continuously eligible positive-weight source after a
finite number of selections: any source left unserved retains a fixed service
ratio while each selected competitor's ratio strictly increases.

Every source owns metadata capacity, so another source cannot evict its entire
frontier. The same scheduler chooses fetches and offers. When offers are
eventually retired, each continuously nonempty source is cataloged, fetched,
and offered infinitely often. No liveness claim attaches to one item whose
origin disappears or whose later catalog replaces it. Unreachable sources do
not obstruct reachable ones: their bounded catalog attempt releases the sole
permit before entering finite backoff.

The liveness claim excludes a user-held offer, a disabled or zero-weight source,
an origin that changes after discovery, permanent external failure, and process
termination. These are explicit environment or user choices, not scheduler
starvation.

Assume each encoder invocation is total. The archive lane is FIFO and contains
at most `A` commitments. Every commitment therefore has finitely many finite
predecessors and eventually reaches verified admission. Later judgments may be
served while it runs but cannot overtake its reserved observation. Encoder or
filesystem failure pauses that durable commitment for explicit restart rather
than losing the user's judgment or blocking the chamber.
