# Picmash

Picmash is a native Poolrooms application for learning one person's preference
over a local image collection. It presents balanced image pairs, records the
chosen rendering and its exact presentation context, and projects those duels
into an explicit Bradley-Terry ordering. The browser shows that learned order
without calling it universal image quality.

The product exposes only the evidence-bearing local workflow:

- compare two images;
- mark favorites;
- hide images without deleting their files;
- correct display rotation;
- browse the visible collection in preference order;
- optionally compare bounded remote challengers and promote accepted images.

Faces, embeddings, triads, and the former unified-quality models were rejected
rather than transplanted from the web application.

## Run

```console
cargo run --release -- /path/to/images
```

The path is optional after the first run. Picmash restores the last successful
collection, and **Open Collection** invokes the platform directory chooser.
Press `A` or `D` to choose the left or right image, `1` or `2` to change
chambers, `F1` for the generated command guide, and `F2` for settings.

The currently proved product coordinate is Linux/X11. Ordinary catalog work is
read-only. When remote acquisition is enabled, promotion writes canonical
lossless JPEG XL files beneath the selected collection's
`.picmash-imported/` directory. Picmash places databases under platform data,
the active-collection pointer under state, disposable remote payloads under
cache, and `picmash.toml` under configuration. No mutable product state belongs
in the checkout. Promotion requires `cjxl` on `PATH`.

## Install

```console
scripts/install-local
```

This installs the executable and desktop launcher under `~/.local`. Remove
them with `scripts/uninstall-local`; configuration, state, and image data are
left intact.

## Architecture

`picmash-engine` is the synchronous authority for exact asset and occurrence
identity, collection state, immutable observations, and preference snapshots.
The native `picmash` crate owns presentation and sends bounded commands to one
engine worker; scanning, SQLite, image decoding, and preference fitting never
run on the event-loop thread. `picmash-contract` contains the dependency-light
UI vocabulary shared with the external `picmash-acceptance` executable.

Remote source definitions, sampling chance, reservoir capacity, filters, and
import policy live under `[remote]` in `picmash.toml`. Existing web-app source
configuration is migrated once as a fallback. The native pipeline has one
catalog lane, one media-fetch lane, bounded metadata, and a reservoir that
counts fetching, ready, and displayed challengers. See
[Remote Acquisition](docs/remote-acquisition.md) for its state and resource
laws.

An asset is one exact, EXIF-oriented RGBA rendering. An occurrence is one path
and byte blob in one collection. Judgments retain both identities, rotation,
prompt policy, response time, session, and global observation order. Preference
snapshots are derived projections and may be rebuilt; observations are the
evidence.

`Engine::import_legacy` remains the explicit, read-only migration seam for a
former web database. It imports exact local evidence and refuses old learned
scores, embeddings, face state, and interactions with assets outside the
judged collection. The native application does not search the filesystem for
legacy databases.

Run a known migration explicitly with `picmash --import-legacy DATABASE`. A
legacy image's provenance does not invalidate judgments after that image has
become a member of the judged local collection; interactions with assets
outside the session's collection remain excluded.

## Verification

```console
./check.py
scripts/test-acceptance /tmp/picmash-acceptance
```

The canonical gate formats, lints, and tests the workspace. The hermetic native
story proves favorite, rotation, voting, hiding, browsing, restart persistence,
a bounded local-source reservoir, and canonical remote promotion without
network access.
