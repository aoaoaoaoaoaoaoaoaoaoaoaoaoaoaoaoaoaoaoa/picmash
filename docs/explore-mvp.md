# Explore MVP

## Executive Summary

The similarity browser is built around a learned 5D aesthetic space over frozen
DINO embeddings.

- `x ∈ R^d`: frozen DINO embedding
- `A : R^d → R^5`: learned global linear projection
- `q(x) = A(x - μ)`: the corpus' current 5D similarity coordinates
- `d(x, y) = ‖q(x) - q(y)‖²`: the operative similarity metric

The on-screen map is not the metric itself. It is a 2D PCA shadow of the
current 5D coordinates. Every operation that matters uses the 5D geometry, not
raw screen distance.

## MVP Shape

- `/explore` redirects to a stable explicit triad URL.
- `/explore/{a}/{b}/{c}` renders:
  - a dense 2D map of tiny square thumbnails
  - a focus panel showing nearest 5D neighbors of the selected image
  - a contrastive triad trainer: pick the closest pair among `A/B/C`
- map clicks update `focus=` in the URL and resolve neighbors in 5D space
- triad training persists events and updates the global projection immediately

## Learning Rule

Each triad produces one event: the user picks the most similar pair among
`(a, b)`, `(a, c)`, `(b, c)`.

The model uses a soft choice rule

- `p(ab) ∝ exp(-β d(a, b))`
- `p(ac) ∝ exp(-β d(a, c))`
- `p(bc) ∝ exp(-β d(b, c))`

and takes a gradient step on `A`, followed by row orthonormalization. This
keeps the learned 5D space crisp and prevents uncontrolled scale/shear drift.

## Deliberate Omissions

- no session-conditioned similarity warp yet
- no nonlinear metric head yet
- no direct sampler coupling from explore interactions yet
- no formal clustering; the browser is neighborhood-first, not bucket-first

Those are all compatible future moves. The durable object is the learned 5D
space, not the 2D projection.
