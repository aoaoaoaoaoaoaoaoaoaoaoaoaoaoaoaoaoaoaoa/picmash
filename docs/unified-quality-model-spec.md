# Unified Bayesian Quality Model

This note is the formal specification for the shipped `hierarchical_perturbative_v3`
quality model.

It replaces the old overgrown session-conditioned machine in which asset-side
semantic and vibe latents, session-side semantic and vibe latents, residual
heads, and assorted classifier sidecars all competed to explain the same
evidence.

The new doctrine is austere:

- canonical quality should absorb evidence first
- session dependence is a perturbation, not a coequal branch
- asset-side perturbation features are fixed deterministic functions of the image
- session dependence must be earned by a tightly shrunk scalar gate
- immutable event truth is canonical; cached posteriors are disposable

## Non-Negotiable Shape

1. Asset-to-asset comparisons are Gaussian duel observations.
2. Subject beauty remains Gaussian duel inference.
3. Asset quality backflows into subject beauty through the dominant visible face.
4. Session dependence is a single gated perturbation around canonical quality.
5. Reject / keep / heart are unary observations on the same latent utility.
6. Priors come from embeddings and descriptors; they do not replace posterior
   inference.
7. Replay from immutable observations is the source of truth.

## Versioning Doctrine

The formal model is versioned separately from the prior family.

- `formal version`
  - latent-variable meaning
  - likelihood family
  - replay semantics
  - cache payload layout
- `prior family`
  - how frozen embeddings / descriptors seed priors inside a fixed formal model

Therefore:

- moving from `hierarchical_gaussian_v1` to `hierarchical_perturbative_v3` is a
  formal-version bump
- replacing the linear `τ → κ` head with ARNIQA, LoRA, or some other amortized
  prior within the same semantics is not

## Objects

Indices:

- `u`
  - subject
- `f`
  - face instance
- `i`
  - asset
- `s`
  - session

Observed deterministic features:

- `x_i ∈ R^d`
  - frozen asset embedding
- `τ_i ∈ R^7`
  - 3D-only technical nuisance descriptor
  - resolution, blur, noise, blocking, dark clip, bright clip
- `v_i ∈ R^L`
  - 3D-only low-level vibe descriptor, standardized corpus-wide
- `c_i ∈ R^K`
  - fixed semantic basis from the frozen embedding
- `δ_i ∈ {0, 1}`
  - hard domain gate
  - `1` for `3D`, `0` for `2D`

Deterministic structure:

- `subject(f) = u`
  - confirmed subject owning face `f`
- `faces(i)`
  - visible non-tombstoned faces in asset `i`
- `ω_if`
  - face salience weights

In the current shipped cut:

- `ω_if` is degenerate on the dominant visible face

## Latent Variables

### Subject Layer

- `b_u`
  - latent subject beauty
- `ρ_f`
  - deferred face-instance residual; not yet explicit in the shipped runtime

### Asset Layer

- `a_i`
  - non-face, non-technical canonical baseline
- `κ_i`
  - 3D-only technical quality scalar
- `w_face,d`
  - domain-conditioned face weight
- `w_tech,d`
  - domain-conditioned technical weight
- `Q0_i`
  - canonical asset quality shared across sessions

### Session Layer

- `w_s ∈ R^(K+L)`
  - session perturbation weights over the fixed asset basis
- `λ_s ≥ 0`
  - scalar session-importance gate
- `T_s`
  - session unary threshold
- `η_is`
  - explicit remembered asset-session offset

## Canonical Quality

Let the dominant-face contribution be:

`g_i = Σ_{f ∈ faces(i)} ω_if · b_subject(f)`

Then canonical asset quality is:

`Q0_i = a_i + w_face,d(i) · g_i + δ_i · w_tech,d(i) · κ_i`

Interpretation:

- `a_i`
  - cross-session baseline goodness
- `w_face,d(i) · g_i`
  - quality inherited from the depicted subject
- `δ_i · w_tech,d(i) · κ_i`
  - 3D-only technical cleanliness

If an asset has no visible faces, then `g_i = 0`.

## Session Perturbation

The fixed perturbation basis is:

`h_i = [c_i ; δ_i · γ_vibe,d(i) · v_i]`

The session utility is:

`U_is = Q0_i + λ_s · w_sᵀ h_i + η_is`

This is the central simplification of `v3`.

We do not maintain asset-side session-conditioned semantic latents or
asset-side session-conditioned vibe latents anymore. Session dependence lives on
the session side only, through:

- one weight vector `w_s`
- one scalar gate `λ_s`

This is the null-hypothesis doctrine:

- if the data do not demand session dependence, `λ_s` should stay near `0`
- if the data do demand it, `w_s` learns directions and `λ_s` turns them on

## Unary Threshold

Unary events are observations on `U_is - T_s`.

The shipped `v3` keeps `T_s` as its own tightly regularized scalar rather than
gating it through `λ_s`. This is a deliberate austerity choice: the perturbation
gate governs taste variation, while threshold drift is kept scalar and cheap.

## Likelihoods

### Asset Duel

For a duel between assets `i` and `j` in session `s`:

`P(i beats j) = Φ((U_is - U_js) / β_duel)`

with Gaussian ADF / moment-matching updates online.

### Subject Duel

For a beauty duel between subjects `u` and `v`:

`P(u beats v) = Φ((b_u - b_v) / β_face)`

### Unary Evidence

Reject / keep / heart are one-sided observations:

- reject: `U_is < T_s`
- keep: `U_is > T_s`
- heart: `U_is > T_s`, but with a lower observation noise than reject

These are not separate classifiers. They are unary factors on the same latent
utility.

## Priors

### Technical Prior

For `3D` assets only:

`κ_i ~ N(w_τᵀ τ_i + b_τ, σ²_κ(τ_i))`

The shipped prior family uses a replay-fit linear head over the nuisance
descriptor `τ_i`.

For `2D` assets:

- `δ_i = 0`
- `κ_i = 0`

### Perturbation Basis

The basis itself is deterministic:

- `c_i`
  - projection of the frozen embedding onto a tiny semantic basis
- `v_i`
  - standardized 3D-only vibe descriptor

The shipped prior family uses:

- semantic PCA/SVD basis from the frozen embedding corpus
- standardized low-level vibe descriptor from cached image features
- replay-fit domain-conditioned canonical branch weights

The current runtime learns:

- `w_face,2d`
- `w_face,3d`
- `w_tech,3d`
- `γ_vibe,3d`

Baseline remains the gauge anchor. We do not learn a separate baseline
coefficient.

No asset-side perturbation posterior is learned. The asset contributes fixed
coordinates; the session learns how to read them.

### Session Gate Prior

We parameterize the nonnegative session-importance gate with a raw scalar:

- `ρ_s ~ N(μ_ρ, σ²_ρ)` with `μ_ρ << 0`
- `λ_s = softplus(ρ_s)`

The shipped runtime uses a tight prior centered near zero session dependence.

### Session Weight Prior

`w_s ~ N(0, σ²_w I)`

with `σ²_w` small.

This prevents the perturbation branch from freelancing until evidence forces it
to matter.

## 2D / 3D Gate

The technical and vibe branches are active only for `3D`.

When `δ_i = 0`:

- `κ_i = 0`
- `v_i = 0`
- the perturbation basis contains only the semantic slice

This is hard, not soft.

## Replay Truth

The canonical truth set is:

- `comparisons`
- `nudge_events`
- `heart_events`
- `face_comparisons`
- confirmed subject bindings and merges
- asset-domain labels
- cached embeddings and descriptors

The following are rebuildable cache, not truth:

- asset quality caches
- session quality caches
- subject beauty caches
- frontier / threshold cache
- learned linear technical prior head
- learned perturbative hyperparameters

## Gauge Fixing

The `3D` technical branch is gauge-fixed after replay.

The mean `κ_i` over all technical-bearing cached entries is recentered to `0`,
and the absorbed offset is pushed back into the canonical baseline. This keeps
`κ` interpretable as a within-`3D` residual rather than a hidden domain
intercept, while leaving total utility invariant.

## Migration Posture

The move from `hierarchical_gaussian_v1` to `hierarchical_perturbative_v3` is
allowed to be lossy.

The shipped migration strategy is:

1. treat the most recently touched old session as the center session
2. absorb its old session-conditioned contribution into the canonical baseline
3. initialize the new session perturbation at the null
4. replay from immutable event truth under the new semantics

This is the right cut for a replay-first system. No sentimental compatibility is
owed to obsolete cached utilities.
