# SAM Vocab Lab

`/vocab-lab` is a workflow-only playground for building an eventual open-vocabulary
segmentation phrasebook. It is intentionally not part of the core inference
machine.

## Doctrine

- no SAM runtime dependency inside `picmash-app`
- no Python dependency inside the shipped product
- the lab may emit prompt specs, but it must not own model execution
- any future SAM integration must remain a sidecar that can be deleted without
  disturbing the rest of the app

## Current Surface

The lab currently provides:

- local asset sampling from the existing corpus
- concept naming plus `presence` / `count` / `absence`
- short text phrases
- positive and negative point prompts
- browser-local saved concept drafts
- exported JSON concept sets
- a request preview showing the exact prompt bundle a future SAM adapter would
  receive

There is no inference path yet. The page is for vocabulary construction and
workflow design only.

## Sidecar Contract

If a SAM adapter is ever added, it should be an external tool consuming the
request-preview payload shape emitted by the lab:

```json
{
  "asset_id": "…",
  "workflow": "sam3_text_then_point_refine",
  "concept": {
    "schema_version": 1,
    "concept_name": "glasses",
    "measurement": "presence",
    "phrases": ["glasses"],
    "notes": "…",
    "prompts": {
      "text": ["glasses"],
      "points": [
        { "x": 0.42, "y": 0.31, "label": "positive" },
        { "x": 0.73, "y": 0.28, "label": "negative" }
      ]
    }
  },
  "sam3_request_preview": {
    "text_prompts": ["glasses"],
    "point_prompts": [
      { "x": 0.42, "y": 0.31, "label": 1 },
      { "x": 0.73, "y": 0.28, "label": 0 }
    ],
    "selection_policy": "refine ambiguous matches"
  }
}
```

The sidecar should:

- accept only this explicit request object plus an image blob
- return explicit masks, boxes, and scores
- cache by `(blob_hash, model_hash, phrase_revision, prompt_revision)`
- remain entirely optional and separately deployable

## Severability

The current lab is intentionally isolated:

- server route + markup live in `crates/picmash-app/src/web/vocab.rs`
- styling lives in the `vocab-lab-*` CSS block
- navigation exposure is a single menu entry

Deleting the experiment should therefore mean deleting one route module, one
style block, one docs note, and one nav entry. That is the intended shape.
