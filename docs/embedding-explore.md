# Embedding Explore Mode

## Intent

The app should expose the encoder's raw visual geometry directly, not only the
learned ranking surface layered on top of it.

The purpose of `explore` is:

- inspect whether the frozen embedding model is grouping images in ways that
  feel semantically or aesthetically sane
- browse neighborhoods and clusters without the pressure of pairwise ranking
- let session-local steering act on regions of embedding space rather than only
  on exact images
- create a clean architectural home for future learned metric adaptation

This mode is not a ranking view. It is a geometry view.

## Recommendation

The global map should be stable. Session effects should be rendered as overlays,
weights, contours, halos, or pull fields on top of that stable map.

Do not physically relocate points in 2D per session as a first design.

Why:

- stable geometry builds user spatial memory
- moving the map per session makes it impossible to tell whether the encoder is
  globally sane
- the 2D map is already a projection artifact; applying session warps to the
  projection itself compounds the distortion

So the recommended split is:

- global embedding space `e_i`
- global reduced coordinates `r_i = project(e_i)`
- session-local steering field `f_s(e)`
- session-local cluster pulls defined in original embedding space, not in 2D
  screen distance

## Data Products

The current system already stores frozen embeddings. `explore` should add:

- `embedding_layouts`
  - keyed by `(model_name, layout_name, layout_version)`
  - stores per-image 2D coordinates
- `embedding_clusters`
  - keyed by `(model_name, clustering_name, clustering_version)`
  - stores per-image cluster membership and confidence
- optional `embedding_neighbors`
  - precomputed KNN graph for fast local browsing

These are derived artifacts and must be versioned and disposable.

The source of truth remains the embedding table plus the comparison/nudge log.

## UI Surfaces

### 1. Map

A full-screen scatter or density map of all visible images.

- zoom and pan
- hover shows thumbnail and summary stats
- click opens inspector / local neighborhood
- lasso or click selects a region

For dense corpora, render density tiles first and reveal image sprites only when
zoomed in enough.

### 2. Neighborhood

Given an anchor image or clicked point:

- nearest neighbors in original embedding space
- local cluster label / density
- optional “why are these near?” diagnostics later

This is the fastest way to answer “is the encoder seeing what I see?”

### 3. Cluster Browser

A side panel or dedicated mode that shows:

- cluster list sorted by size / density / session pull
- representative thumbnails per cluster
- per-cluster `+` / `-` steering
- jump to cluster on map

## Session Steering

Cluster or region nudges should act through a smooth kernel in original
embedding space.

For a clicked anchor embedding `e*`, define a session-local spatial pull:

`k_i = exp(-||e_i - e*||² / (2 σ²))`

Then update a session steering field:

`b_{s,i} += η · sign · k_i`

or, more elegantly, update the session residual head so that nearby items gain
utility through the same latent session machinery already used by nudges.

The second form is preferable, because it keeps one utility model instead of
creating a separate cluster-only side channel.

## Relation To Current Model

The current session utility is:

`u_s(i) = α_i + c_i·z_s + r_s(e_i) + b_{s,i}`

where:

- `α_i` is durable baseline quality
- `c_i·z_s` is canonical mood concordance
- `r_s(e_i)` is the session-local residual head over frozen embeddings
- `b_{s,i}` is exact-image session offset

`explore` should expose:

- the raw embedding geometry `e_i`
- the reduced map `r_i`
- the current session field `r_s(e)` and/or `u_s(i)`

This means cluster `+/-` should ideally update `r_s` first and exact-image
offsets second.

## Projection / Layout

Recommended first pass:

- PCA to 32-64 dims for denoising / speed
- UMAP to 2D for the displayed map

Store the layout coordinates. Do not recompute them on every launch.

The 2D layout is a browsing surface, not the metric used for learning.

All actual similarity-driven pulls should use the original or PCA-compressed
embedding space, never plain 2D screen distance.

## Clustering

Recommended first pass:

- HDBSCAN or density-based clustering over PCA-compressed embeddings

Why:

- cluster count need not be fixed
- outliers remain outliers instead of being forced into nonsense groups
- density structure usually matches aesthetic corpora better than hard K-means

If density clustering proves too brittle at scale, fall back to:

- hierarchical K-means or mini-batch K-means for cheap browse buckets

## Future Metric Adaptation

Do not jump to LoRA first.

The right staircase is:

1. frozen encoder
2. learned linear or low-rank metric on top of embeddings
3. optional shallow nonlinear adapter
4. only then consider parameter-efficient fine-tuning of the encoder

Good intermediate forms:

- low-rank Mahalanobis metric
- learned projection head used for distance
- small MLP adapter on top of frozen embeddings

Those already let the system learn “similarity the way I mean it” without
incurring the instability and retraining burden of LoRA.

If a future LoRA exists, it should be judged by one standard only:

- does local neighborhood quality in `explore` improve?

## MVP Shape

The first usable `explore` mode should provide:

- `/explore` route
- persistent 2D layout
- hover thumbnail
- click-to-neighborhood
- cluster sidebar
- cluster `+/-` steering via the current session residual model

That is enough to make embedding quality inspectable and operationally useful
without building the full cathedral on day one.
