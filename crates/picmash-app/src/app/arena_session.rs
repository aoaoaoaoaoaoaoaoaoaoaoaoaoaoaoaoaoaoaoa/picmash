use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use anyhow::{Context, bail};
use ulid::Ulid;

use crate::{
    identity::VisualKey,
    model::{ArenaHandle, ArenaPair, ArenaView, AssetId, RemoteItemId},
};

use super::{AppState, LockExhaustionPolicy, PipelineDisposition, RedirectTarget};

const ARENA_PIPELINE_LIMIT: usize = 6;
const ARENA_COMMAND_REPLAY_LIMIT: usize = 512;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaRevision(pub u64);

impl ArenaRevision {
    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for ArenaRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaSamplerEpoch(pub u64);

impl ArenaSamplerEpoch {
    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for ArenaSamplerEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArenaTurnId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArenaActionToken(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArenaCommandId(pub String);

impl ArenaCommandId {
    #[must_use]
    pub fn forge() -> Self {
        Self(Ulid::new().to_string())
    }
}

impl fmt::Display for ArenaTurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for ArenaActionToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for ArenaCommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaPairRef {
    pub left: ArenaHandle,
    pub right: ArenaHandle,
}

impl ArenaPairRef {
    #[must_use]
    pub fn forge(left: ArenaHandle, right: ArenaHandle) -> Self {
        Self { left, right }
    }

    #[must_use]
    pub fn href(&self) -> String {
        format!("/arena/{}/{}", self.left.slug(), self.right.slug())
    }

    #[must_use]
    pub fn target(&self) -> RedirectTarget {
        RedirectTarget::ArenaPair {
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }

    #[must_use]
    pub fn contains(&self, handle: &ArenaHandle) -> bool {
        &self.left == handle || &self.right == handle
    }
}

impl From<&ArenaPair> for ArenaPairRef {
    fn from(pair: &ArenaPair) -> Self {
        Self::forge(pair.left.handle(), pair.right.handle())
    }
}

#[derive(Debug, Clone)]
pub struct ArenaTurn {
    id: ArenaTurnId,
    action_token: ArenaActionToken,
    revision: ArenaRevision,
    sampler_epoch: ArenaSamplerEpoch,
    pair: ArenaPairRef,
    visual_keys: HashSet<VisualKey>,
}

impl ArenaTurn {
    fn forge(revision: ArenaRevision, sampler_epoch: ArenaSamplerEpoch, pair: &ArenaPair) -> Self {
        Self {
            id: ArenaTurnId(Ulid::new().to_string()),
            action_token: ArenaActionToken(Ulid::new().to_string()),
            revision,
            sampler_epoch,
            pair: ArenaPairRef::from(pair),
            visual_keys: pair.visual_keys(),
        }
    }

    fn from_pair_ref(
        revision: ArenaRevision,
        sampler_epoch: ArenaSamplerEpoch,
        pair: ArenaPairRef,
        visual_keys: HashSet<VisualKey>,
    ) -> Self {
        Self {
            id: ArenaTurnId(Ulid::new().to_string()),
            action_token: ArenaActionToken(Ulid::new().to_string()),
            revision,
            sampler_epoch,
            pair,
            visual_keys,
        }
    }

    fn activate_at(&mut self, revision: ArenaRevision, sampler_epoch: ArenaSamplerEpoch) {
        self.revision = revision;
        self.sampler_epoch = sampler_epoch;
    }

    #[must_use]
    pub fn id(&self) -> &ArenaTurnId {
        &self.id
    }

    #[must_use]
    pub fn action_token(&self) -> &ArenaActionToken {
        &self.action_token
    }

    #[must_use]
    pub const fn revision(&self) -> ArenaRevision {
        self.revision
    }

    #[must_use]
    pub const fn sampler_epoch(&self) -> ArenaSamplerEpoch {
        self.sampler_epoch
    }

    #[must_use]
    pub fn pair(&self) -> &ArenaPairRef {
        &self.pair
    }

    #[must_use]
    pub fn visual_keys(&self) -> &HashSet<VisualKey> {
        &self.visual_keys
    }

    #[must_use]
    pub fn href(&self) -> String {
        self.pair.href()
    }

    #[must_use]
    pub fn target(&self) -> RedirectTarget {
        self.pair.target()
    }
}

#[derive(Debug, Clone)]
pub struct ArenaPageState {
    pub current: Option<ArenaTurn>,
    pub lookahead: Option<ArenaTurn>,
    pub redirect: Option<RedirectTarget>,
}

#[derive(Debug, Clone)]
pub enum ArenaCommand {
    Vote {
        command_id: ArenaCommandId,
        expected_revision: ArenaRevision,
        expected_sampler_epoch: ArenaSamplerEpoch,
        turn_id: ArenaTurnId,
        action_token: ArenaActionToken,
        winner: ArenaHandle,
    },
    Hide {
        command_id: ArenaCommandId,
        expected_revision: ArenaRevision,
        expected_sampler_epoch: ArenaSamplerEpoch,
        turn_id: ArenaTurnId,
        action_token: ArenaActionToken,
        handle: ArenaHandle,
        hidden: bool,
        cluster_ids: Vec<RemoteItemId>,
    },
    LockThread {
        command_id: ArenaCommandId,
        expected_revision: ArenaRevision,
        expected_sampler_epoch: ArenaSamplerEpoch,
        turn_id: ArenaTurnId,
        action_token: ArenaActionToken,
        handle: ArenaHandle,
        active: bool,
    },
    VetoThread {
        command_id: ArenaCommandId,
        expected_revision: ArenaRevision,
        expected_sampler_epoch: ArenaSamplerEpoch,
        turn_id: ArenaTurnId,
        action_token: ArenaActionToken,
        handle: ArenaHandle,
    },
}

impl ArenaCommand {
    #[must_use]
    fn command_id(&self) -> &ArenaCommandId {
        match self {
            Self::Vote { command_id, .. }
            | Self::Hide { command_id, .. }
            | Self::LockThread { command_id, .. }
            | Self::VetoThread { command_id, .. } => command_id,
        }
    }

    #[must_use]
    fn expected_revision(&self) -> ArenaRevision {
        match self {
            Self::Vote {
                expected_revision, ..
            }
            | Self::Hide {
                expected_revision, ..
            }
            | Self::LockThread {
                expected_revision, ..
            }
            | Self::VetoThread {
                expected_revision, ..
            } => *expected_revision,
        }
    }

    #[must_use]
    fn expected_sampler_epoch(&self) -> ArenaSamplerEpoch {
        match self {
            Self::Vote {
                expected_sampler_epoch,
                ..
            }
            | Self::Hide {
                expected_sampler_epoch,
                ..
            }
            | Self::LockThread {
                expected_sampler_epoch,
                ..
            }
            | Self::VetoThread {
                expected_sampler_epoch,
                ..
            } => *expected_sampler_epoch,
        }
    }

    #[must_use]
    fn turn_id(&self) -> &ArenaTurnId {
        match self {
            Self::Vote { turn_id, .. }
            | Self::Hide { turn_id, .. }
            | Self::LockThread { turn_id, .. }
            | Self::VetoThread { turn_id, .. } => turn_id,
        }
    }

    #[must_use]
    fn action_token(&self) -> &ArenaActionToken {
        match self {
            Self::Vote { action_token, .. }
            | Self::Hide { action_token, .. }
            | Self::LockThread { action_token, .. }
            | Self::VetoThread { action_token, .. } => action_token,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaCommandStatus {
    Applied,
    Replayed,
    Stale,
}

#[derive(Debug, Clone)]
pub struct ArenaCommandOutcome {
    pub status: ArenaCommandStatus,
    pub current: Option<ArenaTurn>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerInvalidation {
    Eventual,
    Immediate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArenaCommandAdvance {
    Preserve,
    Flush,
    Reset { local_anchor: Option<AssetId> },
}

impl ArenaCommandAdvance {
    fn from_disposition(disposition: PipelineDisposition, local_anchor: Option<AssetId>) -> Self {
        match disposition {
            PipelineDisposition::Preserve => Self::Preserve,
            PipelineDisposition::Flush => Self::Flush,
            PipelineDisposition::Reset => Self::Reset { local_anchor },
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct ArenaSessionRuntime {
    revision: ArenaRevision,
    sampler_epoch: ArenaSamplerEpoch,
    current: Option<ArenaTurn>,
    pipeline: VecDeque<ArenaTurn>,
    applied_order: VecDeque<ArenaCommandId>,
    applied: HashSet<ArenaCommandId>,
}

impl ArenaSessionRuntime {
    fn revision(&self) -> ArenaRevision {
        self.revision
    }

    fn sampler_epoch(&self) -> ArenaSamplerEpoch {
        self.sampler_epoch
    }

    fn current(&self) -> Option<ArenaTurn> {
        self.current.clone()
    }

    fn current_target(&self) -> RedirectTarget {
        self.current
            .as_ref()
            .map_or(RedirectTarget::ArenaRoot, ArenaTurn::target)
    }

    fn shatter(&mut self) {
        self.revision = self.revision.next();
        self.current = None;
        self.pipeline.clear();
    }

    fn invalidate_sampler(&mut self) {
        self.sampler_epoch = self.sampler_epoch.next();
        self.shatter();
    }

    fn advance_revision(&mut self) {
        self.revision = self.revision.next();
        if let Some(current) = &mut self.current {
            current.activate_at(self.revision, self.sampler_epoch);
        }
        for turn in &mut self.pipeline {
            turn.activate_at(self.revision, self.sampler_epoch);
        }
    }

    fn install_current(&mut self, mut turn: ArenaTurn) {
        turn.activate_at(self.revision, self.sampler_epoch);
        self.current = Some(turn);
    }

    fn flush_sampler(&mut self) {
        self.sampler_epoch = self.sampler_epoch.next();
        self.revision = self.revision.next();
        if let Some(current) = &mut self.current {
            current.activate_at(self.revision, self.sampler_epoch);
        }
        self.pipeline.clear();
    }

    fn reset_to(&mut self, turn: Option<ArenaTurn>) {
        self.advance_revision();
        self.pipeline.clear();
        self.current = turn.map(|mut turn| {
            turn.activate_at(self.revision, self.sampler_epoch);
            turn
        });
    }

    fn reset_sampler_to(&mut self, turn: Option<ArenaTurn>) {
        self.sampler_epoch = self.sampler_epoch.next();
        self.reset_to(turn);
    }

    fn promote_pipeline(&mut self) -> Option<ArenaTurn> {
        let mut turn = self.pipeline.pop_front()?;
        turn.activate_at(self.revision, self.sampler_epoch);
        self.current = Some(turn.clone());
        Some(turn)
    }

    fn remember(&mut self, command_id: ArenaCommandId) {
        if self.applied.insert(command_id.clone()) {
            self.applied_order.push_back(command_id);
        }
        while self.applied_order.len() > ARENA_COMMAND_REPLAY_LIMIT {
            if let Some(retired) = self.applied_order.pop_front() {
                self.applied.remove(&retired);
            }
        }
    }

    fn already_applied(&self, command_id: &ArenaCommandId) -> bool {
        self.applied.contains(command_id)
    }

    fn reconcile_pipeline(&mut self, known_pipeline_ids: &HashSet<ArenaTurnId>) {
        self.pipeline
            .retain(|turn| known_pipeline_ids.contains(turn.id()));
    }

    fn visual_exclusions(&self) -> HashSet<VisualKey> {
        self.current
            .iter()
            .chain(self.pipeline.iter())
            .flat_map(|turn| turn.visual_keys().iter().cloned())
            .collect()
    }
}

impl AppState {
    pub fn arena_page_state(
        &self,
        requested: Option<ArenaPairRef>,
    ) -> anyhow::Result<ArenaPageState> {
        let mut runtime = self.arena_session.lock();
        self.cull_dead_arena_state(&mut runtime)?;

        if let Some(requested_pair) = requested {
            if let Some(current) = runtime.current() {
                if current.pair() != &requested_pair {
                    return Ok(ArenaPageState {
                        current: None,
                        lookahead: None,
                        redirect: Some(current.target()),
                    });
                }
            } else if let Some(turn) = self.mint_arena_turn_from_pair_ref(
                &requested_pair,
                runtime.revision(),
                runtime.sampler_epoch(),
            )? {
                self.note_arena_turn_selected(&turn)?;
                runtime.install_current(turn);
            } else {
                return Ok(ArenaPageState {
                    current: None,
                    lookahead: None,
                    redirect: Some(RedirectTarget::ArenaRoot),
                });
            }
        }

        if runtime.current.is_none()
            && let Some(turn) = self.sample_arena_turn(
                runtime.revision(),
                runtime.sampler_epoch(),
                &HashSet::new(),
                LockExhaustionPolicy::ClearAndRetry,
            )?
        {
            self.note_arena_turn_selected(&turn)?;
            runtime.install_current(turn);
        }

        let lookahead = self.ensure_arena_pipeline_front(&mut runtime, &HashSet::new())?;
        Ok(ArenaPageState {
            current: runtime.current(),
            lookahead,
            redirect: None,
        })
    }

    pub fn arena_current_target(&self) -> anyhow::Result<RedirectTarget> {
        let mut runtime = self.arena_session.lock();
        self.cull_dead_arena_state(&mut runtime)?;
        if runtime.current.is_none()
            && let Some(turn) = self.sample_arena_turn(
                runtime.revision(),
                runtime.sampler_epoch(),
                &HashSet::new(),
                LockExhaustionPolicy::ClearAndRetry,
            )?
        {
            self.note_arena_turn_selected(&turn)?;
            runtime.install_current(turn);
        }
        Ok(runtime.current_target())
    }

    pub fn arena_prefetch_turn(
        &self,
        known_pipeline_ids: &HashSet<ArenaTurnId>,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaTurn>> {
        let mut runtime = self.arena_session.lock();
        self.cull_dead_arena_state(&mut runtime)?;
        if runtime.current.is_none() {
            return Ok(None);
        }
        runtime.reconcile_pipeline(known_pipeline_ids);
        self.cull_dead_arena_pipeline(&mut runtime)?;
        if runtime.pipeline.len() >= ARENA_PIPELINE_LIMIT {
            return Ok(None);
        }
        let mut effective_excluded = runtime.visual_exclusions();
        effective_excluded.extend(excluded_visual_keys.iter().cloned());
        let Some(turn) = self.sample_arena_turn(
            runtime.revision(),
            runtime.sampler_epoch(),
            &effective_excluded,
            LockExhaustionPolicy::PreserveAndStop,
        )?
        else {
            return Ok(None);
        };
        runtime.pipeline.push_back(turn.clone());
        Ok(Some(turn))
    }

    pub fn apply_arena_command(
        &self,
        command: ArenaCommand,
    ) -> anyhow::Result<ArenaCommandOutcome> {
        let mut runtime = self.arena_session.lock();
        self.cull_dead_arena_state(&mut runtime)?;

        if runtime.already_applied(command.command_id()) {
            return Ok(ArenaCommandOutcome {
                status: ArenaCommandStatus::Replayed,
                current: runtime.current(),
            });
        }

        if !Self::arena_command_matches_current(&runtime, &command) {
            return Ok(ArenaCommandOutcome {
                status: ArenaCommandStatus::Stale,
                current: runtime.current(),
            });
        }

        let current_pair = runtime
            .current()
            .map(|turn| turn.pair().clone())
            .context("arena command accepted without a current turn")?;
        let advance = self.devour_arena_command_effect(&command, &current_pair)?;
        runtime.remember(command.command_id().clone());

        match advance {
            ArenaCommandAdvance::Preserve => {
                self.cull_dead_arena_pipeline(&mut runtime)?;
                runtime.advance_revision();
                if runtime.promote_pipeline().is_none() {
                    let turn = self.sample_arena_turn(
                        runtime.revision(),
                        runtime.sampler_epoch(),
                        &HashSet::new(),
                        LockExhaustionPolicy::ClearAndRetry,
                    )?;
                    runtime.current = turn;
                }
            }
            ArenaCommandAdvance::Flush => {
                runtime.flush_sampler();
            }
            ArenaCommandAdvance::Reset { local_anchor } => {
                let turn = self.sample_arena_turn_preserving_local_anchor(
                    runtime.revision().next(),
                    runtime.sampler_epoch().next(),
                    local_anchor.as_ref(),
                    &HashSet::new(),
                    LockExhaustionPolicy::ClearAndRetry,
                )?;
                runtime.reset_sampler_to(turn);
            }
        }

        if let Some(current) = runtime.current() {
            self.note_arena_turn_selected(&current)?;
        }
        Ok(ArenaCommandOutcome {
            status: ArenaCommandStatus::Applied,
            current: runtime.current(),
        })
    }

    pub fn shatter_arena_session(&self) {
        self.arena_session.lock().shatter();
    }

    pub fn apply_arena_sampler_invalidation(&self, invalidation: SamplerInvalidation) {
        if matches!(invalidation, SamplerInvalidation::Immediate) {
            self.arena_session.lock().invalidate_sampler();
        }
    }

    fn arena_command_matches_current(
        runtime: &ArenaSessionRuntime,
        command: &ArenaCommand,
    ) -> bool {
        let Some(current) = &runtime.current else {
            return false;
        };
        command.expected_revision() == runtime.revision()
            && command.expected_sampler_epoch() == runtime.sampler_epoch()
            && current.sampler_epoch() == runtime.sampler_epoch()
            && command.turn_id() == current.id()
            && command.action_token() == current.action_token()
    }

    fn devour_arena_command_effect(
        &self,
        command: &ArenaCommand,
        current_pair: &ArenaPairRef,
    ) -> anyhow::Result<ArenaCommandAdvance> {
        match command {
            ArenaCommand::Vote { winner, .. } => {
                ensure_pair_member(current_pair, winner, "winner")?;
                self.apply_arena_vote_effect(&current_pair.left, &current_pair.right, winner)?;
                Ok(ArenaCommandAdvance::Preserve)
            }
            ArenaCommand::Hide {
                handle,
                hidden,
                cluster_ids,
                ..
            } => {
                ensure_pair_member(current_pair, handle, "hidden handle")?;
                self.apply_hide_arena_handle_effect(
                    handle,
                    *hidden,
                    cluster_ids,
                    &current_pair.left,
                    &current_pair.right,
                )?;
                Ok(ArenaCommandAdvance::Preserve)
            }
            ArenaCommand::LockThread { handle, active, .. } => {
                ensure_pair_member(current_pair, handle, "locked handle")?;
                let ArenaHandle::Remote(item_id) = handle else {
                    bail!("thread lock requires a remote arena handle");
                };
                self.set_external_subsource_lock(*item_id, *active)
                    .map(|disposition| ArenaCommandAdvance::from_disposition(disposition, None))
            }
            ArenaCommand::VetoThread { handle, .. } => {
                ensure_pair_member(current_pair, handle, "vetoed handle")?;
                let (local_anchor, disposition) = self.apply_veto_external_thread_effect(
                    handle,
                    &current_pair.left,
                    &current_pair.right,
                )?;
                Ok(ArenaCommandAdvance::from_disposition(
                    disposition,
                    local_anchor,
                ))
            }
        }
    }

    fn ensure_arena_pipeline_front(
        &self,
        runtime: &mut ArenaSessionRuntime,
        excluded_visual_keys: &HashSet<VisualKey>,
    ) -> anyhow::Result<Option<ArenaTurn>> {
        self.cull_dead_arena_pipeline(runtime)?;
        if runtime.pipeline.is_empty() {
            let mut effective_excluded = runtime.visual_exclusions();
            effective_excluded.extend(excluded_visual_keys.iter().cloned());
            if let Some(turn) = self.sample_arena_turn(
                runtime.revision(),
                runtime.sampler_epoch(),
                &effective_excluded,
                LockExhaustionPolicy::PreserveAndStop,
            )? {
                runtime.pipeline.push_back(turn);
            }
        }
        Ok(runtime.pipeline.front().cloned())
    }

    fn cull_dead_arena_state(&self, runtime: &mut ArenaSessionRuntime) -> anyhow::Result<()> {
        if let Some(current) = runtime.current.clone()
            && (current.sampler_epoch() != runtime.sampler_epoch()
                || self.arena_turn_view(&current)?.is_none())
        {
            runtime.shatter();
        }
        self.cull_dead_arena_pipeline(runtime)
    }

    fn cull_dead_arena_pipeline(&self, runtime: &mut ArenaSessionRuntime) -> anyhow::Result<()> {
        let live = runtime
            .pipeline
            .iter()
            .map(|turn| {
                Ok((
                    turn.id().clone(),
                    turn.sampler_epoch() == runtime.sampler_epoch()
                        && self.arena_turn_view(turn)?.is_some(),
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        runtime
            .pipeline
            .retain(|turn| live.get(turn.id()).copied().unwrap_or(false));
        Ok(())
    }

    fn sample_arena_turn(
        &self,
        revision: ArenaRevision,
        sampler_epoch: ArenaSamplerEpoch,
        excluded_visual_keys: &HashSet<VisualKey>,
        lock_exhaustion: LockExhaustionPolicy,
    ) -> anyhow::Result<Option<ArenaTurn>> {
        Ok(self
            .choose_next_pair(lock_exhaustion, excluded_visual_keys)?
            .as_ref()
            .map(|pair| ArenaTurn::forge(revision, sampler_epoch, pair)))
    }

    fn sample_arena_turn_preserving_local_anchor(
        &self,
        revision: ArenaRevision,
        sampler_epoch: ArenaSamplerEpoch,
        local_anchor: Option<&AssetId>,
        excluded_visual_keys: &HashSet<VisualKey>,
        lock_exhaustion: LockExhaustionPolicy,
    ) -> anyhow::Result<Option<ArenaTurn>> {
        Ok(self
            .choose_next_pair_preserving_local_anchor(
                local_anchor,
                lock_exhaustion,
                excluded_visual_keys,
            )?
            .as_ref()
            .map(|pair| ArenaTurn::forge(revision, sampler_epoch, pair)))
    }

    fn mint_arena_turn_from_pair_ref(
        &self,
        pair_ref: &ArenaPairRef,
        revision: ArenaRevision,
        sampler_epoch: ArenaSamplerEpoch,
    ) -> anyhow::Result<Option<ArenaTurn>> {
        let Some(view) = self.arena_pair(&pair_ref.left, &pair_ref.right)? else {
            return Ok(None);
        };
        let Some(pair) = view.pair else {
            return Ok(None);
        };
        Ok(Some(ArenaTurn::from_pair_ref(
            revision,
            sampler_epoch,
            pair_ref.clone(),
            pair.visual_keys(),
        )))
    }

    pub fn arena_turn_view(&self, turn: &ArenaTurn) -> anyhow::Result<Option<ArenaView>> {
        self.arena_pair(&turn.pair().left, &turn.pair().right)
    }

    fn note_arena_turn_selected(&self, turn: &ArenaTurn) -> anyhow::Result<()> {
        let Some(view) = self.arena_turn_view(turn)? else {
            return Ok(());
        };
        let Some(pair) = view.pair else {
            return Ok(());
        };
        self.note_remote_pair_selected(&pair)
    }
}

fn ensure_pair_member(
    pair: &ArenaPairRef,
    handle: &ArenaHandle,
    label: &'static str,
) -> anyhow::Result<()> {
    if pair.contains(handle) {
        Ok(())
    } else {
        bail!("{label} is not part of the active arena turn")
    }
}
