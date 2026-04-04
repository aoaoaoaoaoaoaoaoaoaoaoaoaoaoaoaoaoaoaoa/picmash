# ArcFace Face Oracle Next

The current facemash oracle is mis-specified.

## Actual Current Machine

- training data comes from `faces.embedding`
- `faces.embedding` is the aligned-face DINO embedding
- ArcFace lives separately in `faces.recognition_embedding`
- the oracle therefore ignores the actual face-recognition representation
- training target is collapsed posterior beauty, not the duel log itself

This is weaker than necessary even before model-class questions.

## First Corrective Cut

1. Finish and harden ArcFace backfill coverage for local and remote faces.
2. Switch oracle training data from `faces.embedding` to `faces.recognition_embedding`.
3. Keep the first upgraded head austere:
   - pooled identity ArcFace embedding
   - linear Bayesian / ridge-style head
   - no nonlinear soup

This should already be materially stronger because ArcFace encodes facial structure
that is much closer to the geometry relevant to attractiveness than generic DINO
face-crop embeddings.

## Next Strengthening After The Switch

Stop regressing posterior beauty means and train on the actual pairwise facemash
evidence.

Preferred formulation:

- latent score `s(z)` over ArcFace embedding `z`
- pairwise Bradley-Terry / Thurstone-probit likelihood on duel outcomes
- strong shrinkage
- replay-fit, not hand-wavy online slop

## If Linear ArcFace Still Underfits

The next sane extension is a low-rank quadratic head, not an MLP:

- `score(z) = aᵀz + Σ λ_j (u_jᵀz)^2`
- whitened ArcFace coordinates
- very small rank, e.g. `4..8`

This buys curvature without descending into dimensionality-cursed nonsense.

## What Not To Do

- no trees
- no raw RBF over the full embedding
- no MLP on current sample counts
- no feature soup mixing DINO and ArcFace before the ArcFace cut is actually in

