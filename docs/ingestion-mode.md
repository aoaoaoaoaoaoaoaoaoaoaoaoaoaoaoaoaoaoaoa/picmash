# Ingestion Mode

## Intent

Ingestion should become its own work mode, not a side effect of ordinary pairwise
ranking.

The goal is not to produce a fine ordering. The goal is to triage a new stream of
images into:

- obvious keep
- obvious reject
- uncertain, worth a quick human glance

The machine should learn online from the human thresholding behavior and
progressively skip the easy cases, while remaining conservative about novelty and
uncertainty.

## Human Loop

The core interaction is a binary up/down workflow:

- `up`: this image clears the current threshold and should enter the ranked corpus
- `down`: this image fails the current threshold and should stay out

This is not the same judgment as pairwise ranking. It is a thresholded admission
decision.

The interface should be fast and single-image:

- one image at a time
- almost no chrome
- strong keyboard control
- visible current operating mode and confidence

The human is calibrating a frontier, not sorting the world.

## Decision States

Each candidate should land in one of four states:

- `promoted`: admitted with no more ingest work needed
- `quarantined`: withheld from the main corpus but preserved
- `probe`: not enough confidence; must be shown to the human
- `defer`: temporarily skipped because the model is uncertain or the queue is saturated

There should be no hard deletion from learned ingest decisions alone.

## Learned Signal

The ranking model already gives us the right latent decomposition:

- `α_i`: global cross-mood quality
- `c_i`: mood-sensitive coordinates

Ingestion should primarily learn a predictor for `α_i` from the frozen visual
embedding:

- `x_i`: DINO embedding
- `q(x_i) -> (μ_α, σ_α)`

This makes ingestion a conservative prediction problem over global quality, not a
full reconstruction of the ranking model.

## Online Threshold Calibration

The threshold is not fixed forever. It is a moving operating point chosen by the
human.

Conceptually:

- maintain an admission threshold `τ`
- learn `P(α_i > τ | x_i, history)`
- use human up/down decisions as threshold labels

The model should update online after every manual ingest decision.

The important output is not just a score, but a calibrated confidence that an item
is safely above or below the current threshold.

## Skip Policy

The skip policy should be asymmetric and conservative.

- auto-promote only when `P(α_i > τ)` is very high
- auto-quarantine only when `P(α_i < τ)` is very high and novelty is low
- otherwise route to `probe`

Abstention is a first-class outcome. The model should be rewarded for saying "I do
not know" instead of hallucinating confidence.

## Novelty Guard

A pure quality predictor will eventually become overconfident about familiar
regions and brittle on unfamiliar ones.

So every ingest decision also needs a novelty signal:

- embedding density / nearest-neighbor distance in DINO space
- distance from known promoted and quarantined regions

If an image is globally unusual, the system should bias toward `probe`, even when
the quality head thinks it is probably bad.

This is how we avoid training the system to sand away the interesting edges.

## Anchor-Probe Fallback

When the predictor is uncertain, ingestion should be allowed to escalate into a
very short comparative workflow instead of an immediate hard admit/reject.

For example:

- show the candidate against a small ladder of anchor images around the current
  threshold
- after 2-5 comparisons, estimate whether the candidate plausibly lies above or
  below `τ`

So ingest mode remains threshold-centric, but can borrow pairwise evidence when a
single-image judgment is too noisy.

## Data We Should Preserve

The ingestion subsystem should log:

- raw candidate identity and source
- model prediction at decision time
- uncertainty / novelty values
- human up/down verdicts
- whether the case was auto-skipped, auto-promoted, auto-quarantined, or probed

This gives us a proper calibration dataset for later revisions.

## Operational Shape

The eventual flow should look like this:

1. ingest a batch
2. score each candidate with the current quality head and novelty guard
3. auto-skip only the most obvious cases
4. present the uncertain frontier to the human in a fast up/down loop
5. update the threshold model online
6. periodically tighten the auto-skip band as confidence improves

## Non-Goals

This mode should not:

- replace pairwise ranking
- permanently delete images on model judgment alone
- collapse novelty into noise
- pretend the learned threshold is static

## First Implementation Shape

The first serious implementation should be deliberately modest:

- a separate ingest queue UI
- manual `up` / `down`
- learned `μ_α, σ_α` head from DINO
- conservative abstention
- no auto-quarantine until calibration is empirically trustworthy

Everything richer can grow from there.
