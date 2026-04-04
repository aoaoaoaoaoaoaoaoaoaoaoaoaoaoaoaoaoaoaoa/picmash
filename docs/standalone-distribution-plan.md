# Standalone Distribution Plan

## Status

This document is an active feature plan.

It describes the work required to turn `picmash` into a clean standalone app
for third-party users, with a first-class install story and no accidental
reliance on the local development environment.

It is not a promise that every phase lands immediately, but it is the intended
direction.

## Thesis

`picmash` should ship as a self-contained local application:

- one Rust binary
- embedded frontend assets
- SQLite in XDG state
- ONNX models bootstrapped on first launch or provided explicitly
- CPU inference as the guaranteed floor
- GPU acceleration as opportunistic enrichment

The end-user contract should not require:

- Python
- npm
- a repo checkout
- systemd
- manual copying of provider libraries

Those are maintainer concerns, not user concerns.

## Target User Experience

The publishable install path should be:

1. download a release archive
2. unpack it
3. run `picmash init IMAGE_ROOT`
4. run `picmash`
5. wait for first-launch model bootstrap if models are not already present
6. use the app at `http://127.0.0.1:8788`

Optional platform integration may exist later:

- Linux user service
- desktop launcher
- native packaging

But the core contract must work without any of that.

## Non-Goals

- shipping ONNX models inside the Rust crate
- making `check.py` the official install story
- requiring GPU support
- requiring source builds
- making systemd mandatory

## Current State

The good:

- runtime bootstrap is already Python-free
- frontend assets are already embedded into the Rust binary
- SQLite is already bundled
- ONNX models already bootstrap lazily into XDG cache

The bad:

- the current `install` workflow is a maintainer script, not a user installer
- Linux/systemd assumptions leak into the install path
- GPU provider placement is ad hoc
- the service is not first-run safe on a fresh machine
- remote import crushing still assumes optional shell tools

## Current Blockers

### 1. Maintainer Script Posing as Installer

`check.py install` currently assumes:

- Python
- npm
- a writable repo checkout
- `cargo install --path`
- `systemctl --user`

That is acceptable for development and unacceptable as the canonical public
install story.

### 2. First-Run Configuration Trap

The service currently starts `picmash` with no corpus argument, while the app
requires a configured `runtime.corpus_root` unless `IMAGE_ROOT` is passed once.

So the current user service cannot be treated as a fresh-install entry point.

### 3. GPU Provider Packaging Is Not Real Distribution

The current flow copies Linux `.so` provider libraries from `target/release`
into `~/.local/bin`.

This is:

- Linux-specific
- tied to the local build tree
- not a valid release-asset story

### 4. Source Builds Still Have System-Lib Assumptions

At least one dependency stack still expects native system support:

- `turbojpeg` with `pkg-config`

This is tolerable for developers and should not define the published user
experience.

### 5. Optional Runtime Tooling Still Leaks Through

Lossless import crushing shells out to:

- `jpegtran`
- `jpegoptim`
- `oxipng`

This is not fatal because admission falls back to original bytes, but it is
still a system assumption and should either be internalized or explicitly
demoted to an optional enhancement.

### 6. Model Bootstrap Depends on External URLs

First launch currently downloads ONNX models from remote URLs.

That is a reasonable default, but it is still part of the public install
contract and should be treated deliberately:

- the URLs should be explicit and versioned
- the app should degrade cleanly offline
- the boot UX should make the bootstrap state obvious

## Desired Architecture

### Product Shape

The application remains one product crate in spirit:

- Rust server
- embedded web assets
- runtime model files on disk

The ONNX weights are runtime assets, not crate contents.

### Runtime Search Order

Model discovery should obey one strict order:

1. explicit env/config override
2. installed model directory beside or under the app
3. XDG cache copy
4. remote bootstrap

This allows:

- explicit operator control
- offline reuse
- clean first-launch bootstrap

### Install Surface

The public release artifact should contain:

- `picmash` binary
- ONNX Runtime provider libraries needed by that platform
- optional helper scripts or service files

It should not require a local build tree.

## Phases

### Phase 1. Split Maintainer and User Install Paths

Create an explicit boundary:

- maintainer path: current repo-local build/test/install tooling
- user path: prebuilt release artifacts only

Concretely:

- stop presenting `check.py install` as the public install contract
- treat it as maintainer glue
- define release artifact layout clearly

Acceptance:

- a third-party user can install without Python or npm

### Phase 2. First-Run Initialization Becomes Explicit

Add a clean initialization entry point, likely:

- `picmash init IMAGE_ROOT`

Responsibilities:

- persist `runtime.corpus_root`
- validate the corpus path
- optionally pre-create config/data dirs
- print the launch URL and bootstrap expectations

Then:

- `picmash` with no args becomes a clean pure run path

Acceptance:

- a fresh user no longer needs to understand config internals just to start

### Phase 3. Release Packaging of ORT Provider Libraries

Move provider library handling out of the local build tree assumptions.

The release artifact should package platform-appropriate ORT runtime pieces so
the binary can discover them without an ad hoc copy step from `target/release`.

Acceptance:

- published binaries run on supported targets without repo-local post-install
  surgery

### Phase 4. Harden First-Launch Model Bootstrap

Formalize the first-launch ONNX fetch path.

Requirements:

- versioned model identities
- atomic download and install
- clear progress logging
- clean offline failure mode
- env/config override for operator-managed models

Nice-to-have:

- model bundle manifest rather than raw hardcoded URLs

Acceptance:

- the app can boot from a clean machine with no local models and produce a
  sane user-facing loading state

### Phase 5. Demote or Internalize Optional Native Tooling

Decide one of two paths for import crushing:

- internalize it fully in Rust
- or keep shell-outs but declare them optional and noncanonical

The published install story must not silently rely on those binaries being
present.

Acceptance:

- absence of `jpegoptim`, `jpegtran`, or `oxipng` is not a surprising feature
  regression for end users

### Phase 6. Optional Platform Integration

After the core install contract is clean, add sugar:

- Linux user service
- desktop launcher
- maybe native packaging later

These should be optional wrappers around the same standalone app, not required
for normal operation.

Acceptance:

- system integration improves convenience without changing the canonical
  runtime contract

## Release Contract

The official support stance should be:

- CPU inference works everywhere we claim support
- GPU acceleration is best-effort
- missing GPU runtime pieces must not make the app unusable

This matters more than theoretical peak performance.

## Repo Implications

The repo can keep its current broad shape:

- Rust application crate
- TypeScript source for the embedded frontend
- maintainer tooling in scripts

But the meaning must become explicit:

- `apps/web` is maintainer source
- embedded frontend assets are the runtime payload
- `check.py` is maintainer automation
- release artifacts are the user install surface

## Licensing Gate

Before public packaging is treated as settled, model licensing must be reviewed
deliberately.

In particular, if bundled or bootstrapped models impose restrictions that are
incompatible with the intended release posture, that is a product constraint,
not a minor footnote.

## Definition of Done

This feature is done when a third-party user can:

1. obtain a release archive
2. unpack it on a supported machine
3. run one explicit init step with their corpus root
4. start `picmash`
5. have models bootstrap without Python or npm
6. use the app without repo-local assumptions

At that point `picmash` is a real standalone application rather than a
developer-operated project that happens to be runnable by others.
