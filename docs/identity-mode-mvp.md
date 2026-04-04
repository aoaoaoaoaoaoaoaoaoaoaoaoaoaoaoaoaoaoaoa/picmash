# Identity Mode MVP

This note supersedes the earlier half-formed identity plan.

It is the de novo design for proper face identification in the current
subject-clustered world.

## Intent

Identity work should be a dedicated curation surface.

Facemash is for beauty judgment. It should not contain name fields, same-id
buttons, or any other identity-assignment slop.

The machine should instead expose a top-level `identities` mode with one job:

- surface likely same-person matches
- let the operator name clusters
- let the operator confirm or reject candidate merges
- propagate confirmed names everywhere else

No inline facemash naming. No popup nagging. No raw integer identity ids in the
UI.

## Naming The Thing

`identity_id` is ugly and semantically muddled.

Internally, the first-class object should be called a `Subject`.

- `SubjectId`
  - stable internal cluster key
- `Subject`
  - a cluster of one or more confirmed same-person face instances
- `display_name`
  - optional user-facing label for that subject

The route/page can still be called `identities`, because that is the natural
operator-facing term. But the domain model should be `Subject`, not
`FaceIdentity`.

This cleanly spans:

- real people
- anime characters
- stylized fictional persons

without overcommitting to any one ontology.

## Current Reality

The live system is already not the older nullable-identity world.

Every face already belongs to a stable cluster, even if it is only a singleton.

So the correct review frontier is not:

- named subject vs unknown face

but:

- anchor subject vs candidate subject

where most candidate subjects will be unnamed singletons.

That is a better machine anyway.

## Non-Negotiable Product Rules

1. Facemash may display confirmed names, but may not assign or merge
   identities.
2. `identities` is the only page where subject assignment happens.
3. The user never sees raw `SubjectId` values.
4. A name is required before a subject can absorb another subject.
5. Recognition uses a dedicated identity model, not the beauty embedding.

## Recognition Lane

- `SCRFD` remains the detector and landmark source.
- `ArcFace` becomes the recognition embedding model.
- recognition embeddings are computed eagerly for local faces
- imported remotes become eligible only once they are local assets

So the recognition stack is:

1. detect face
2. align tight `112×112` crop
3. run ArcFace ONNX
4. persist normalized recognition embedding

This lives in the existing background face-scan pipeline, not lazily in the UI.

## Subject Model

The first-class machine state is:

- `Face`
  - one detected face instance tied to an asset plus geometry
- `Subject`
  - cluster of confirmed same-person faces
- `SubjectBinding`
  - durable confirmed `(owner, geometry_key) -> SubjectId`
  - this is the canonical truth across rescans and detector refreshes
- `RecognitionEmbedding`
  - ArcFace vector attached to one face
- `SubjectPrototype`
  - normalized centroid of one subject’s member embeddings
- `SubjectPairVeto`
  - durable “these two subjects are not the same person”

The important change from the earlier note is the veto shape.

Because the live system already works in subject clusters, the durable negative
memory should also be subject-shaped, not face-shaped.

That means the first-pass veto is:

- canonical unordered pair of subjects
- keyed to the current recognition model

not:

- `(face_id, identity_id)`

and not:

- `(asset_id, geometry_key, identity_id)`

Those older shapes were appropriate for a nullable-identity world. They are
no longer the best fit.

## Persistence

The clean storage shape is:

- rename conceptual `face_identities` → `subjects`
  - implementation may retain the old table name temporarily, but the code
    should speak in `Subject*` terms
- `faces.subject_id NOT NULL`
- `subject_bindings`
  - exactly one owner plus geometry key maps to one confirmed subject
- `faces.recognition_embedding`
- `faces.recognition_model`
- new `subject_match_vetoes`

Suggested `subject_match_vetoes` shape:

- `subject_lo`
- `subject_hi`
- `recognition_model`
- `created_at`

with canonical ordering so the veto is symmetric.

## User-Facing Handle Policy

Raw subject ids must not leak into URLs, labels, or visible controls.

The clean contract is:

- named subjects are addressed by unique slugified display name
- candidate-review actions use opaque signed handles, not raw ids
- unnamed subjects never get a standalone user-facing route

So:

- `/identities`
  - overview
- `/identities/{subject_slug}`
  - named subject review page
- `/identities/{subject_slug}/{candidate_handle}`
  - optional focused review URL

Unnamed subjects exist in the overview as anonymous rows, but they are not
route-addressable by raw id.

## Subject Prototypes

Each subject gets one prototype.

Prototype construction:

1. collect all non-hidden local faces in the subject with current ArcFace
   embeddings
2. L2-normalize each embedding
3. average them
4. renormalize the centroid

That centroid is the subject’s recognition prototype.

If a subject has no valid recognition embeddings, it is inert and does not
participate in review.

## Representative Selection

Each subject row needs one representative face.

The representative should be the subject medoid, not a random member:

- compute cosine similarity from each member embedding to the subject prototype
- choose the member with the highest similarity

For named subjects, this guarantees the left-hand representative is an actual
confirmed member of that subject.

For unnamed singleton subjects, the representative is trivially that one face.

This is the correct “central or arbitrary” rule:

- central when possible
- arbitrary only in the degenerate singleton case

## Candidate Frontier

This should parallel the duplicate frontier, but over subjects.

For each subject:

1. compute similarity from that subject’s prototype to every other subject’s
   prototype
2. discard self
3. discard vetoed subject pairs
4. keep only candidates above the current threshold

The candidate subject should usually be unnamed in the MVP.

That is an important restriction.

### Why Named-vs-Named Matching Should Wait

If both subjects are already named, a positive match can imply:

- the names are aliases
- one name is wrong
- both are the same person but should be merged under one canonical name

That is a real identity-resolution problem, not merely “assign this unknown to
that known.”

So the clean MVP frontier is:

- left side: any active subject, named or unnamed
- right side: unnamed candidate subjects only

This keeps the operator workflow crisp and avoids immediate alias-merge soup.

Named-vs-named merge can be a second phase.

## Threshold

There should be a persisted `identity_match_threshold` slider.

It acts directly on cosine similarity in ArcFace space.

Semantics:

- lower threshold
  - more candidate boxes
  - more false positives
- higher threshold
  - fewer candidate boxes
  - cleaner rows

The first pass should have one threshold only.

No second-best margin. No quality multiplier. No cascade. One number.

## Identities Overview

The overview page is a vertical stack of subject rows.

Each row has:

- left: one representative face card
- left metadata:
  - current display name or blank field
  - member count
  - top candidate score
  - candidate count
- right: candidate boxes for over-threshold unnamed subjects

The visual shape on the right should be board-square sized, face-centric cards.

Each candidate box is a candidate subject represented by its own medoid face.

## Row Semantics

### Named Row

Left side:

- representative face
- editable name field populated with the current name

Right side candidate controls:

- green confirm
  - merge candidate subject into the anchor subject
- red tombstone
  - write a subject-pair veto
- optional open/full-preview affordance
- optional exclude-face affordance on the candidate representative

### Unnamed Row

Left side:

- representative face
- blank name field
- create-name action

Right side candidate controls:

- green confirm disabled
  - tooltip: assign a name first
- red tombstone enabled
  - pruning a false match does not require naming

This is the key UX rule:

- naming is required before absorb/merge
- naming is not required before rejecting a bad candidate

That gives you disciplined semantics without making anonymous review useless.

## Merge Semantics

Confirming a candidate on a named row means:

1. merge candidate subject into anchor subject
2. invalidate prototype cache
3. invalidate review frontier cache
4. rebuild beauty state by replay under the new subject graph

That last step is non-negotiable because facemash is already subject-native.

Any collapsed self-duels stay excluded exactly as in the existing beauty replay
logic.

## Interaction With Beauty / Facemash

Facemash should become identity-display-only:

- show confirmed subject name when present
- no naming field
- no same-id button
- no direct identity mutation controls

Once identities mode exists, facemash no longer owns identity editing.

That is a deliberate simplification.

## Interaction With Existing Subject Rows

Because every face already belongs to a subject cluster:

- a singleton unnamed subject is the default “unassigned” state
- identities mode is really a subject-fusion workflow

This is good.

It means the machine never has to conjure “unknown” as a null state. It only
has:

- unnamed singleton cluster
- unnamed multi-face cluster
- named cluster

Those are better states.

## Ordering

Rows should be ordered by something like:

- highest candidate similarity first
- then candidate count
- then named rows ahead of unnamed rows

The exact ranking can be tuned, but the page should always feel like:

- the best likely merges rise to the top

not:

- arbitrary cluster dump

## Caching / Invalidation

The review frontier should be cached in the same spirit as the duplicate
frontier.

Invalidate when:

- recognition embedding added or replaced
- subject renamed
- subject merged
- veto written or cleared
- face excluded / hidden
- recognition model revision changes
- threshold changes

Prototype cache and frontier cache should be separate.

## Holes / Objections Resolved

### “What about unnamed rows?”

Show them, but only if they are active:

- named, or
- have at least one over-threshold unnamed candidate

Do not dump every inert singleton on the page.

### “What about named-vs-named duplicate people?”

Not in MVP.

That is a second-phase alias merge problem and deserves its own clean workflow.

### “What about raw ids?”

Never visible.

Use subject slugs and opaque review handles.

### “Why allow tombstoning unnamed rows?”

Because rejecting a false match is queue hygiene, not outward identity
projection.

### “Why require a name before confirm?”

Because confirmation projects one cluster into another subject in a way the
operator should be able to reason about semantically, not as anonymous cluster
math.

## Route Surface

MVP routes:

- `/identities`
  - overview rows
  - threshold control
- `/identities/{subject_slug}`
  - focused review for one named subject

The candidate permalink can wait until a later pass. The MVP only needs the
overview plus named-subject focus.

## MVP Deliverable

The first clean implementation should include:

- ArcFace ONNX lane
- eager recognition embeddings for local faces
- subject prototypes
- subject-pair vetoes
- identities overview page with threshold slider
- unnamed-row naming
- named-row confirm / tombstone
- confirmed names rendered in facemash and elsewhere
- facemash identity editing removed

That is the austere first machine.

It is enough to turn identity from popup sludge into a coherent review system.
