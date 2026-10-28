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
- browse the visible collection in preference order, narrowed by source and
  stream;
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
run on the event-loop thread. While a pair is judged, two smith threads decode
the images of the pair likely to follow it: the engine's forecast of the next
local pair, or the next remote offer and its anchor. `picmash-contract`
contains the dependency-light UI vocabulary shared with the external
`picmash-acceptance` executable.

Remote source definitions, sampling chance, reservoir capacity, filters, and
import policy live under `[remote]` in `picmash.toml`. Existing web-app source
configuration is migrated once as a fallback. The native pipeline has one
catalog lane, one media-fetch lane, one maximum-effort archive lane, bounded
metadata, and a reservoir that counts fetching, ready, and displayed
challengers. See
[Remote Acquisition](docs/remote-acquisition.md) for its state and resource
laws.

An asset is one exact, EXIF-oriented RGBA rendering. An occurrence is one path
and byte blob in one collection. Judgments retain both identities, rotation,
prompt policy, response time, session, and global observation order. Preference
snapshots are derived projections and may be rebuilt; observations are the
evidence.

Provenance records each item an asset was admitted from: its source (a 4chan
board or local directory), stream (a thread or folder), and origin (URL and
MD5, or file path). Admission inscribes it in the transaction that records the
occurrence. Browse narrows the gallery to one source or stream, or to assets
without provenance.

`Engine::import_legacy` remains the explicit, read-only migration seam for a
former web database. It imports exact local evidence and refuses old learned
scores, embeddings, face state, and interactions with assets outside the
judged collection. The native application does not search the filesystem for
legacy databases.

Run a known migration explicitly with `picmash --import-legacy DATABASE`. A
legacy image's provenance does not invalidate judgments after that image has
become a member of the judged local collection; interactions with assets
outside the session's collection remain excluded.
The import also attests each legacy image's source and stream from the web
app's records. Both steps are idempotent, so rerunning it on a database
imported by an earlier release attests provenance alone.

## Verification

```console
./check.py
scripts/test-acceptance /tmp/picmash-acceptance
```

The canonical gate formats, lints, and tests the workspace. The hermetic native
story proves favorite, rotation, voting, hiding, browsing, restart persistence,
a bounded local-source reservoir, nonblocking durable archive resumption,
canonical remote promotion, and source and stream narrowing without network
access.

## Showcase

The showcase is a presentation take, never a verdict. After the build in
`scripts/test-acceptance`, `picmash-acceptance --stills DIR` performs the
comparison loop and develops its stills, the frontispiece being the fuzzy
Persian against the Sphynx; `--film DIR` also records `picmash-showcase.mp4`.
`PICMASH_ACCEPTANCE_BINARY` names the instrumented Picmash, as for acceptance.
The specimen is the photographs listed in
`crates/picmash-acceptance/specimen/cats.toml`: CC0 or public-domain Wikimedia
Commons files and Pexels-licensed photographs, none asking attribution or
share-alike, fetched into the host cache and admitted only by SHA-256. Every
take publishes `specimen-credits.md`, crediting their authors as a courtesy.
