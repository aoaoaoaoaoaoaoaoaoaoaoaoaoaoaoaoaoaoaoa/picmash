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
- browse the visible collection in preference order.

External discovery, faces, embeddings, and the former unified-quality models
were rejected rather than transplanted from the web application.

## Run

```console
cargo run --release -- /path/to/images
```

The path is optional after the first run. Picmash restores the last successful
collection, and **Open Collection** invokes the platform directory chooser.
Press `A` or `D` to choose the left or right image, `1` or `2` to change
chambers, `F1` for the generated command guide, and `F2` for settings.

The currently proved product coordinate is Linux/X11. Corpus access is
read-only. Picmash places its database under the platform data directory, the
active-collection pointer under state, and `picmash.toml` under configuration.
No mutable product state belongs in the checkout.

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

An asset is one exact, EXIF-oriented RGBA rendering. An occurrence is one path
and byte blob in one collection. Judgments retain both identities, rotation,
prompt policy, response time, session, and global observation order. Preference
snapshots are derived projections and may be rebuilt; observations are the
evidence.

`Engine::import_legacy` remains the explicit, read-only migration seam for a
former web database. It imports exact local evidence and refuses old learned
scores, embeddings, face state, and external-source contamination. The native
application does not search the filesystem for legacy databases.

## Verification

```console
./check.py
scripts/test-acceptance /tmp/picmash-acceptance
```

The canonical gate formats, lints, and tests the workspace. The hermetic native
story seeds a real image corpus, then proves favorite, rotation, voting, hiding,
browsing, and restart persistence without network access.
