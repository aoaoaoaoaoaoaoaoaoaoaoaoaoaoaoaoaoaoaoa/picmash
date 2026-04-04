# Similarity Geometry MVP

This is the first pass of the de novo similarity architecture. It is active code,
not a speculative future note.

## Shape

- `raw` explore mode remains a frozen-DINO neighborhood browser.
- `learned` explore mode is now split into:
  - a learned 5D geometry
  - a separate 2D layout reducer

The geometry is authoritative. The 2D map is only a shadow.

## Geometry

The learned geometry is represented by `SimilarityModel`:

- `Linear`
  - a linear projection from centered DINO embeddings into `R^5`
  - seeded from PCA
  - used before enough triad supervision exists
- `Ordinal`
  - per-image learned coordinates in `R^5`
  - initialized from the linear DINO prior
  - updated directly from triad judgments
  - carries the DINO prior forward by periodically refitting a linear map from embeddings into the learned coordinates

The bootstrap threshold is currently `24` triads.

## Training

Triad supervision uses the chosen-closest-pair likelihood:

- `P(ab chosen) ∝ exp(-β ‖y_a - y_b‖²)`
- likewise for `ac` and `bc`

In the ordinal regime:

- touched image coordinates are updated directly in `R^5`
- a weak prior pull keeps them near the DINO-induced initializer
- the DINO prior is then refit to the current learned coordinates

This keeps the learned space replayable and gives new images an inductive placement.

## Layout

`raw` mode still uses the existing manifold-style reducer.

`learned` mode now uses non-metric MDS over the learned 5D pairwise distances:

- initialize from PCA
- fit monotone disparities by isotonic regression
- run a SMACOF-style majorization loop in 2D

For very large corpora, learned layout falls back to PCA instead of paying the full quadratic SMACOF bill.

## Persistence

Similarity state is persisted as an enum payload in `similarity_models`, with the old linear fields still written as a compatibility substrate for existing rows and debugging.

Triad events remain append-only in `similarity_triads`.

## Current limits

- Active triad selection is still heuristic, though it now queries the new geometry abstraction.
- The learned 2D map is more principled than the old learned UMAP, but it is not yet an uncertainty-aware active ordinal-query engine.
- The DINO prior refit is global and cheap; it is not yet a richer probabilistic posterior.
