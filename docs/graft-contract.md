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

## Host Obligations

The Poolrooms graft should treat the engine as its sole authority for catalog
identity, judgment persistence, favorites, and preference scores. UI state,
navigation, image presentation, and application lifecycle remain host
concerns.

The host must:

1. Supply a durable state-database path and a corpus path.
2. Give each judgment surface a stable context revision.
3. Render the exact occurrence, render digest, and rotation in the issued
   prompt.
4. Mint one command ID per user intent and reuse it only for an exact retry.
5. Rebuild preferences off the interaction path, then adopt the returned
   snapshot atomically.
6. Display holdout metrics as diagnostics, never as a claim of universal image
   quality.

The first graft should expose only local collection browsing, pairwise
preference, favorites, hiding, and rotation. Similarity is stored but has no
authorized learner. External sources and face workflows require separate
commissions.

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

## Preference Law

`bradley-terry-l2-v1` is a regularized pairwise model. It consumes only asset
duels whose endpoints remain visible. Scores are centered and carry observed
duel counts, not fabricated uncertainty. At twenty or more duels, the snapshot
reports chronological 80/20 holdout log loss and accuracy. New non-duel events
do not invalidate a preference snapshot.
