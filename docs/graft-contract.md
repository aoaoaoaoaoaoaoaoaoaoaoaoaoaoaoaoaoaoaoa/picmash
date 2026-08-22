# Graft Contract

## Canonical Names

An **asset** is one exact oriented RGBA rendering. Byte-distinct files with the
same rendering are one asset.

An **occurrence** is one path and byte blob in one collection. An asset may
have several occurrences.

An **observation** is immutable evidence produced by a judgment command. Its
global database ID is its recorded order.

A **snapshot** is a derived, reproducible preference projection over one
catalog revision and one duel frontier. It is not primary evidence.

## Native Boundary

The native application treats the engine as its sole authority for catalog
identity, judgment persistence, favorites, and preference scores. UI state,
navigation, image presentation, and application lifecycle remain host concerns.

The host:

1. Supplies a durable database path under platform data and a corpus path.
2. Gives each judgment surface a stable context revision.
3. Renders the exact occurrence, render digest, and rotation in the issued
   prompt.
4. Mints one command ID per user intent and reuses it only for an exact retry.
5. Rebuilds preferences on the engine worker, then adopts its collection and
   prompt projections in event order.
6. Displays holdout metrics as diagnostics, never as a claim of universal image
   quality.

The graft exposes local collection browsing, pairwise preference, favorites,
hiding, rotation, and explicitly configured remote challengers. Similarity is
stored but has no authorized learner. Face workflows remain excluded.

The native event loop owns no blocking filesystem, database, image-decoding,
or model work. A bounded command channel feeds one engine worker. Two fixed
remote effect lanes perform catalog and media work; every mailbox and retained
frontier is bounded. Image blades cross back as immutable RGBA buffers and
become GPU textures on the UI thread.
The external acceptance executable observes only a small one-way state and
stable target vocabulary from `picmash-contract`.

## Persistence Law

Schema migrations are transactional and namespaced under `pm_`. Foreign
keys, strict tables, WAL, and a busy timeout are enabled at open. Scans retire
unseen occurrences only after a complete traversal; unreadable images become
unavailable and retain a diagnostic. The engine never deletes corpus files.

Judgments capture the assets, occurrences, exact render digests, rotations,
prompt revision, policy or representation revision, response time, session,
and ordering authority. A command ID is idempotent only when its payload digest
matches.

Favorites are current collection state plus immutable set/unset events. Hiding
changes the preference population and therefore the catalog revision. Rotation
changes presentation but not asset identity or the preference population.

Remote payloads are cache, never collection occurrences. Promotion first
decodes the prepared payload, applies the judged rotation, losslessly
re-encodes it as JPEG XL, verifies exact render identity, and atomically writes
it beneath `.picmash-imported/`. Only then may the engine ingest the occurrence
and seal the remote item as promoted. A promoted duel is inserted atomically;
its command ID survives exact retries.

## Preference Law

`bradley-terry-l2-v1` is a regularized pairwise model. It consumes only asset
duels whose endpoints remain visible. Scores are centered and carry observed
duel counts, not fabricated uncertainty. At twenty or more duels, the snapshot
reports chronological 80/20 holdout log loss and accuracy. New non-duel events
do not invalidate a preference snapshot.
