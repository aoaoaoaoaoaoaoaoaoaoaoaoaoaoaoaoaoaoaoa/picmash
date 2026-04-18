# Arena State Machine Redesign

Date: 2026-04-13

## Summary

Picmash arena has accumulated a class of skip, duplicate, stale-pair, and route fallback bugs because the browser currently keeps too much parallel arena state. The redesign goal is to make the backend the single authority for arena turns while preserving rapid, smooth user interactions through backend-issued prefetch.

The chosen direction is arena-first, not a generic framework-first rewrite. We will build a rigorous arena state machine with authoritative turns, idempotent commands, a typed sampler scope, and a backend-reserved prefetch pipeline. If facemash, triads, or another future sampler-driven page later needs the same machinery, we can extract shared pieces then. For now, a cross-system `SampledSystem` trait is deferred.

## Decision

Proceed with an arena-specific implementation of:

- Authoritative finite state machine on the backend.
- Monotonic arena revisions and turn IDs.
- Idempotent command IDs for every mutating action.
- Typed arena sampler scope, initially `Global` and `LockedThread(ThreadKey)`.
- Exhaustive scope transition reducer for lock, unlock, and thread veto.
- Backend-reserved linear prefetch pipeline.
- Frontend that renders backend-issued envelopes and submits commands transactionally.

Explicitly defer:

- A generic `StatefulSampledSystem` framework.
- Full candidate trees.
- Multi-lane prefetch caches.
- Paxos/Raft-style distributed consensus.

SQLite plus the single write actor is the authority/leader. We want the linearizability and fencing-token lessons from consensus systems, not literal multi-replica consensus.

## Implementation Status

Implemented arena-first on 2026-04-17 with a deliberately smaller spine than the aspirational protocol below:

- Arena state is process-local and authoritative while the app is running: `ArenaRevision`, `ArenaTurnId`, `ArenaActionToken`, current turn, reserved FIFO pipeline, and a bounded in-memory command replay log live in `ArenaSessionRuntime`.
- Command IDs are carried by every mutating arena command. They are minted into server-rendered forms and JS fills a missing ID before submit; retries of the same rendered form reuse the same value. The log is not persisted. After restart, old commands become stale by revision/turn mismatch rather than replaying as duplicates.
- The wire response returns the authoritative current turn plus a single lookahead payload as needed by the current frontend, not the full `ArenaEnvelope` / `pipeline: Vec<_>` shape sketched below.
- `SamplerEpoch` is implemented as a hard validity fence on issued turns. Immediate exogenous sampler changes bump the epoch, clear current/pipeline, and make old browser-held commands stale; eventual changes leave the issued pipeline to drain.
- Scope transitions use the reducer for lock/unlock/TX preserve/reset decisions. Vote preserves the issued pipeline; hide and thread-scope mutations reset or preserve according to their typed transition.
- Render, prefetch, payload construction, and mutating command application run through blocking tasks so arena sampling and DB work do not sit on Tokio hot workers.
- Cache pruning landed alongside this work because the arena prefetch/rendition cache had become operationally dangerous; it is not logically part of the state-machine design.

## Terms

- `TX`: Thread veto. This bans/removes the current remote thread from the sampler frontier.
- `Thread lock`: Restrict the arena sampler to one remote thread. This is the practical dual of thread veto: lock keeps only that thread; TX removes that thread.
- `Unlock`: Leave locked-thread scope and return to global scope. This must invalidate/reset the active pipeline because the user explicitly asked to stop looking only at that thread.
- `Turn`: One displayed pair of arena cards.
- `Envelope`: Backend-issued payload containing the current turn, revision data, scope, and optional future turns.
- `Pipeline`: Backend-reserved queue of future turns for smooth rapid interactions.
- `Command`: A mutating user action against a specific issued turn.
- `CommandId`: An idempotency key carried by a command for retry-safe handling. It may be client-generated or pre-minted by the backend into the rendered action form.
- `TurnId`: Backend-generated ID for a displayed turn.
- `Revision`: Backend-generated monotonic revision for a session's arena state.
- `SamplerEpoch`: Backend-generated hard validity epoch for issued turns. It changes only when an exogenous hard eligibility change would make old turns contradict current sampler rules.

## Primary Correctness Goals

- The backend is the only source of truth for the current arena state.
- Every mutating command applies at most once.
- A stale command cannot mutate a later turn.
- A frontend-rendered future turn must have been issued by the backend.
- The frontend may animate and queue backend-issued turns, but must not sample or invent arena truth.
- Scope transitions must be exhaustive and type-checked.
- Thread lock, unlock, and TX interactions must be encoded in one reducer, not scattered across route handlers and sampling logic.
- Visual duplicates should be prevented at sampling/reservation time, not by trusting ad hoc browser exclusion state.

## Non-Goals

- Do not introduce Paxos or Raft. There is one backend authority.
- Do not build a reusable sampler framework until there is a second real consumer.
- Do not require a full prefetch candidate tree for the first redesign.
- Do not make normal rating effects immediately invalidate the pipeline if delayed application gives smoother UX and preserves correctness.

## Backend Protocol Shape

The backend should serve the arena page/API with an authoritative envelope:

```rust
struct ArenaEnvelope {
    session_id: SessionId,
    revision: ArenaRevision,
    sampler_epoch: SamplerEpoch,
    scope: ArenaScope,
    current: ArenaTurnEnvelope,
    pipeline: Vec<ArenaTurnEnvelope>,
}

struct ArenaTurnEnvelope {
    turn_id: ArenaTurnId,
    left: ArenaHandle,
    right: ArenaHandle,
    invalidation_keys: Vec<ArenaInvalidationKey>,
    sampler_epoch: SamplerEpoch,
    action_token: ArenaActionToken,
}
```

Commands are submitted against a specific issued turn:

```rust
struct ArenaCommand {
    command_id: CommandId,
    expected_revision: ArenaRevision,
    expected_sampler_epoch: SamplerEpoch,
    turn_id: ArenaTurnId,
    action_token: ArenaActionToken,
    action: ArenaAction,
}
```

The backend responds with:

```rust
enum ArenaCommitResult {
    Applied { next: ArenaEnvelope },
    Duplicate { next: ArenaEnvelope },
    Stale { current: ArenaEnvelope },
    Invalid { current: ArenaEnvelope, reason: ArenaRejectReason },
}
```

`Duplicate` must return the same logical result as the original command or the current authoritative envelope in a way that is safe for retry. The client should treat `Duplicate` as success.

`Stale` means the client must discard its local pipeline and render/refetch from `current`.

## Idempotent Commands

Every mutating arena action must carry a `command_id`.

Initial command coverage:

- Vote.
- Hide.
- Heart.
- Rotate.
- Thread lock.
- Thread unlock.
- Thread veto/TX.
- Remote keep/enshrine if it uses a separate route.

Backend responsibilities:

- Store recently applied command IDs per session.
- Make command application and idempotency recording part of the same write transaction.
- If a duplicate command ID is seen, do not reapply side effects.
- Return a safe authoritative envelope for duplicates.
- Bound retention with TTL or a fixed-size per-session log.

Frontend responsibilities:

- Generate `crypto.randomUUID()` per user action.
- Preserve the command ID across retries for the same action.
- Do not create a new command ID for a retry caused by ambiguous network failure.

## Arena Scope

Arena scope is part of state, not incidental route behavior:

```rust
enum ArenaScope {
    Global,
    LockedThread(ThreadKey),
}

struct ThreadKey {
    source_key: SourceKey,
    stream_id: StreamId,
}
```

Scope-changing actions:

```rust
enum ArenaScopeAction {
    LockThread(ThreadKey),
    UnlockThread,
    VetoThread(ThreadKey),
}
```

Pipeline dispositions:

```rust
enum PipelineDisposition {
    Preserve,
    ResetAndRefill { reason: PipelineResetReason },
}
```

We may later add partial filtering, but the first implementation should prefer full reset for scope changes. It is simpler and less bug-prone.

Required transition cases:

```rust
match (scope, action) {
    (ArenaScope::Global, ArenaScopeAction::LockThread(thread)) => {
        // Enter locked scope. Reset pipeline under the locked sampler.
    }
    (ArenaScope::LockedThread(current), ArenaScopeAction::UnlockThread) => {
        // Leave locked scope. Reset pipeline under global sampler.
    }
    (ArenaScope::Global, ArenaScopeAction::VetoThread(thread)) => {
        // Stay global. Remove/veto thread. Reset pipeline.
    }
    (ArenaScope::LockedThread(current), ArenaScopeAction::VetoThread(thread))
        if current == thread =>
    {
        // TX the locked thread. Clear lock, veto thread, reset under global sampler.
    }
    (ArenaScope::LockedThread(current), ArenaScopeAction::VetoThread(thread)) => {
        // TX a different thread while locked. Stay locked.
        // Preserve only if impossible for current pipeline to include that thread;
        // otherwise reset. Full reset is acceptable for first implementation.
    }
    (ArenaScope::LockedThread(current), ArenaScopeAction::LockThread(thread))
        if current == thread =>
    {
        // Re-lock same thread. No-op or preserve.
    }
    (ArenaScope::LockedThread(current), ArenaScopeAction::LockThread(thread)) => {
        // Switch locked thread. Reset pipeline under new locked scope.
    }
}
```

This transition table should live in one module and should be unit-tested exhaustively. Avoid re-encoding these semantics in web handlers, frontend JS, or sampling loops.

## Pipeline Policy

Start with a linear pipeline:

```rust
struct ArenaPipeline {
    turns: Vec<ArenaTurnEnvelope>,
    issued_under_revision: ArenaRevision,
    sampler_epoch: SamplerEpoch,
    scope: ArenaScope,
}
```

Pipeline rules:

- The backend reserves all pipeline turns.
- The frontend never samples new turns.
- The sampler excludes visual keys already present in `current + pipeline`.
- Pipeline turns carry `SamplerEpoch` plus invalidation keys, including visual key and thread key.
- Normal vote/reject/accept actions may preserve the pipeline.
- Scope changes reset/refill the pipeline.
- Immediate exogenous changes bump `sampler_epoch` and reset/refill.
- Eventual exogenous changes do not bump `sampler_epoch`; they affect only future refill as the issued pipeline drains.
- Thread unlock resets/refills, even though it is logically a relaxation of scope, because the UX intent is to stop viewing only the locked thread.
- TX of the locked thread clears the lock and resets/refills global-with-thread-vetoed.

## Exogenous Sampler Changes

Arena commands such as vote, hide, lock, unlock, and TX are endogenous: they flow through the turn command state machine. Other controls and maintenance events are exogenous because they change sampler assumptions from outside the current turn.

Classify every exogenous sampler mutation as:

```rust
enum SamplerInvalidation {
    Eventual,
    Immediate,
}
```

`Immediate` means the already-issued current/pipeline could now contain a turn that is illegal or visibly contradictory under the new hard constraints. Immediate changes bump `SamplerEpoch`, clear current/pipeline, and force the browser to refetch authoritative state before it can mutate again.

`Eventual` means the already-issued turns are still legal, merely no longer ideally ranked or weighted. Eventual changes do not invalidate the pipeline.

Default examples:

- Source disabled or made hard ineligible: `Immediate`.
- Hard sampler mode switches such as local-only, remote-only, source-only, or a future filtered mode: `Immediate`.
- Dedup radius / visual-identity threshold changes: `Immediate`, because they alter a hard pair eligibility predicate.
- External/local mix probability inside the mixed regime: `Eventual`.
- Explore/exploit percentage, temperature, source weights, and ranking weights: `Eventual`.
- Model/score changes that alter ordering but not eligibility: `Eventual`.

Endpoint settings that become hard modes should be represented as hard regimes, not as ordinary soft weights. For the current scalar source-mix control this means `0%`, `1..99%`, and `100%` are separate semantic regimes; crossing regime boundaries is `Immediate`, while moving inside the mixed regime is `Eventual`.

Delayed sampler effects are acceptable for smoothness:

- A vote does not need to make the winner/loser immediately eligible/ineligible for already reserved turns.
- A remote accept/import does not need to inject the imported asset into the sampler until the current pipeline drains.
- Rating/model effects may lag until refill.

Immediate effects:

- Dedup: prevent visual duplicates at reservation time across current and pipeline turns.
- Thread lock: switch scope and reset/refill.
- Thread unlock: switch scope and reset/refill.
- Thread veto/TX: remove thread and reset/refill, especially if locked.

## Frontend Contract

The frontend may keep only:

- The current backend envelope.
- A queue of backend-issued future turn envelopes.
- The latest known revision/epoch.
- A FIFO list of pending command IDs if rapid actions are allowed before responses return.

The frontend must not keep:

- Independent sampler state.
- Independent source/thread lock truth.
- Independent dedup truth beyond transient render checks.
- Mutating action results as authoritative until the backend confirms or the command is safely queued.

Smoothness options:

- Conservative option: wait for each command response before visually promoting the next turn. This is simplest and safest.
- Smooth option: allow visual promotion of a backend-issued queued turn immediately, but treat it as pending until the command response confirms. Any subsequent user command must either wait behind the pending command or include an explicit predecessor dependency. If the predecessor returns stale/invalid, discard the pending visual state and render the authoritative envelope.

The smooth option preserves rapid feel while keeping backend authority, but it requires more careful client logic. The first implementation can choose the conservative option if latency is acceptable.

## Server-Side Invariants

Design tests around these invariants:

- Applying the same `command_id` twice does not duplicate side effects.
- A command for a non-current `turn_id` cannot mutate current state unless explicitly accepted as a duplicate of an already applied command.
- A command with the wrong revision/token returns stale or invalid.
- The sampler never reserves a pipeline containing duplicate visual keys.
- A locked-thread pipeline contains only that thread.
- A global pipeline after TX does not contain the vetoed thread.
- Unlocking a thread lock changes scope to global and resets/refills.
- TX on the locked thread changes scope to global and resets/refills without the vetoed thread.
- Re-locking the same thread is a no-op or preserve, depending on chosen semantics.
- Switching from one locked thread to another resets/refills under the new thread.
- Config/sampler-epoch changes invalidate old pipeline turns.
- Multi-tab commands produce deterministic `Applied`, `Duplicate`, or `Stale` results.

## Staged Implementation Plan

### Stage 1: Idempotent Arena Commands

Goal: stop data corruption from duplicate/ambiguous submissions.

Work:

- Add `command_id` to arena mutating forms/API requests.
- Add a bounded command result log to the store.
- Apply command effects and record command ID in one write transaction.
- Return `Applied`, `Duplicate`, `Stale`, or `Invalid`.
- Update frontend to generate and reuse command IDs on retry.

This stage can be done before changing the prefetch pipeline.

### Stage 2: Arena Scope FSM

Goal: make lock/unlock/TX interactions exhaustive and testable.

Work:

- Introduce `ArenaScope`, `ThreadKey`, and `ArenaScopeAction`.
- Move lock/unlock/TX semantics into one reducer.
- Replace scattered conditionals in arena app logic with reducer calls.
- Add unit tests for the full transition table.
- Make pipeline reset/preserve decision an explicit reducer output.

This stage is arena-specific. Do not extract a generic framework yet.

### Stage 3: Backend-Issued Turn Envelope and Pipeline

Goal: make the backend own current turn and future turns.

Work:

- Introduce `ArenaEnvelope` and `ArenaTurnEnvelope`.
- Add revision, turn ID, action token, sampler epoch, and scope to envelope responses.
- Make `/api/arena/next` return backend-reserved turns under the current scope/epoch.
- Store or derive enough server-side pipeline context to stop trusting client-supplied exclusion keys as the only dedup mechanism.
- Ensure the sampler excludes `current + pipeline` visual keys.

### Stage 4: Frontend Simplification

Goal: remove browser-side arena truth.

Work:

- Replace the current large inline arena JS state surface with a smaller typed arena client.
- The client renders envelopes and submits commands.
- The client discards its pipeline on stale/invalid responses.
- The client handles `Duplicate` as success.
- Keep prefetch, but restrict it to backend-issued envelopes.

This can be done in inline JS first or as a TypeScript migration. TypeScript is preferred if we can afford the churn.

### Stage 5: Optional Extraction

Only after arena is correct and a second sampler-driven system needs the same machinery:

- Extract a small `TurnPipeline<T>` or command idempotency helper.
- Avoid extracting a broad `SampledSystem` trait until two systems have similar enough turn/action/scope shapes.

Likely first shared pieces:

- `CommandId` / command log table.
- `Revision` / stale response helpers.
- Maybe a generic `TurnEnvelope<Turn, Scope>`.

Do not force facemash/triples/explore into a generic trait prematurely.

## Opus Review Notes

A read-only Opus review agreed with the overall correctness goals but warned against building the generic framework first.

Main takeaways:

- Arena is the only current system with the full prefetch/scope complexity.
- Facemash, triad, and explore do not yet justify a shared `SampledSystem` trait.
- Idempotent commands are the highest-priority correctness fix.
- The arena scope FSM is a clear win.
- The frontend should not use fire-and-forget mutating actions.
- Extract shared infrastructure only when a second system needs it.

Adopted conclusion:

- Proceed arena-first.
- Preserve the rigorous state-machine design.
- Defer the cross-system abstraction.

## Open Questions

- Should the first frontend pass use conservative promotion or pending optimistic promotion?
- How much command result history should be retained per session?
- Should pipeline reservations be persisted in SQLite, held in memory, or regenerated deterministically from revision/epoch?
- Do we want a hard limit on pipeline depth, e.g. 3 or 5 turns?
- Should TX of a different thread while locked preserve the pipeline or always reset for simplicity?
- Should thread lock/unlock actions be exposed as commands against the current turn, or as independent scope commands with their own action tokens?
