use super::*;
use crate::identity::VisualKey;
use std::collections::HashSet;
use tracing::{Instrument, info_span};

enum ArenaRootRender {
    Redirect(String),
    Empty(Box<ArenaRootPage>),
}

struct ArenaRootPage {
    view: ArenaView,
    external_status: ExternalArenaStatus,
}

enum ArenaPairRender {
    Redirect(String),
    Pair(Box<ArenaPairPage>),
}

struct ArenaPairPage {
    current: ArenaTurn,
    view: ArenaView,
    lookahead: Option<ArenaLookaheadMarkup>,
    external_status: ExternalArenaStatus,
}

pub(super) async fn arena_root(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let span = info_span!("arena.render.root");
    async move {
        let state = match ready_app_or_snapshot(&state) {
            Ok(state) => state,
            Err(snapshot) => return Ok(boot_response(snapshot)),
        };
        log_site_loaded("/arena");
        match arena_root_render_blocking(state).await? {
            ArenaRootRender::Redirect(href) => Ok(Redirect::to(&href).into_response()),
            ArenaRootRender::Empty(page) => {
                let ArenaRootPage {
                    view,
                    external_status,
                } = *page;
                Ok(render_markup(routed_layout(
                    "arena-page",
                    PageGeometry::Viewport,
                    NavPage::Arena,
                    Some(arena_mode_menu(external_status.clone())),
                    arena_markup(None, view, None, external_status),
                )))
            }
        }
    }
    .instrument(span)
    .await
}

pub(super) async fn arena_reroll(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    if let Some(state) = state.ready_app() {
        state.shatter_arena_session();
    }
    Ok(Redirect::to("/arena").into_response())
}

pub(super) async fn arena(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let span = info_span!(
        "arena.render.pair",
        pair_href = %pair_href,
        left_handle = %left_id,
        right_handle = %right_id,
    );
    async move {
        let state = match ready_app_or_snapshot(&state) {
            Ok(state) => state,
            Err(snapshot) => return Ok(boot_response(snapshot)),
        };
        let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
        let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
        match arena_pair_render_blocking(state, ArenaPairRef::forge(left, right)).await? {
            ArenaPairRender::Redirect(href) => Ok(Redirect::to(&href).into_response()),
            ArenaPairRender::Pair(page) => {
                let ArenaPairPage {
                    current,
                    view,
                    lookahead,
                    external_status,
                } = *page;
                log_site_loaded("/arena/pair");
                Ok(render_markup(routed_layout(
                    "arena-page",
                    PageGeometry::Viewport,
                    NavPage::Arena,
                    Some(arena_mode_menu(external_status.clone())),
                    arena_markup(Some(current), view, lookahead, external_status),
                )))
            }
        }
    }
    .instrument(span)
    .await
}

pub(super) async fn vote(
    State(state): State<SharedRuntimeState>,
    Path((_left_id, _right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<VoteForm>,
) -> WebResult<Response> {
    let span = info_span!(
        "arena.action.vote",
        left_handle = %form.left_id,
        right_handle = %form.right_id,
        winner_handle = %form.winner_id,
        accept_json = %accepts_json(&headers),
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let winner = ArenaHandle::from_str(&form.winner_id).map_err(anyhow::Error::msg)?;
        let command = ArenaCommand::Vote {
            command_id: ArenaCommandId(form.command.command_id),
            expected_revision: ArenaRevision(form.command.arena_revision),
            expected_sampler_epoch: ArenaSamplerEpoch(form.command.arena_sampler_epoch),
            turn_id: ArenaTurnId(form.command.turn_id),
            action_token: ArenaActionToken(form.command.action_token),
            winner,
        };
        let refresh_state = state.clone();
        let outcome = apply_arena_command_blocking(state, command, "vote").await?;
        if !refresh_state.quality_refresh_is_inline()? {
            refresh_state.schedule_quality_model_refresh();
        }
        arena_command_response(refresh_state, outcome, accepts_json(&headers)).await
    }
    .instrument(span)
    .await
}

pub(super) async fn vote_get(
    Path((left_id, right_id)): Path<(String, String)>,
) -> WebResult<Response> {
    Ok(Redirect::to(&format!("/arena/{left_id}/{right_id}")).into_response())
}

async fn apply_arena_command_blocking(
    state: SharedAppState,
    command: ArenaCommand,
    action: &'static str,
) -> anyhow::Result<ArenaCommandOutcome> {
    tokio::task::spawn_blocking(move || state.apply_arena_command(command))
        .await
        .map_err(|error| anyhow::anyhow!("joining arena {action} task: {error:#}"))?
}

async fn arena_root_render_blocking(state: SharedAppState) -> anyhow::Result<ArenaRootRender> {
    tokio::task::spawn_blocking(move || {
        let target = state.arena_target()?;
        match target {
            RedirectTarget::ArenaRoot => Ok(ArenaRootRender::Empty(Box::new(ArenaRootPage {
                view: state.arena_empty()?,
                external_status: state.external_status()?,
            }))),
            RedirectTarget::ArenaPair { .. } => Ok(ArenaRootRender::Redirect(target.href())),
            RedirectTarget::FacemashRoot
            | RedirectTarget::FacemashPair { .. }
            | RedirectTarget::ExploreRoot { .. }
            | RedirectTarget::ExploreTriad { .. } => {
                Ok(ArenaRootRender::Redirect("/arena".to_owned()))
            }
        }
    })
    .await
    .map_err(|error| anyhow::anyhow!("joining arena root render task: {error:#}"))?
}

async fn arena_pair_render_blocking(
    state: SharedAppState,
    pair: ArenaPairRef,
) -> anyhow::Result<ArenaPairRender> {
    tokio::task::spawn_blocking(move || {
        let page = state.arena_page_state(Some(pair))?;
        if let Some(redirect) = page.redirect {
            return Ok(ArenaPairRender::Redirect(redirect.href()));
        }
        let Some(current) = page.current else {
            return Ok(ArenaPairRender::Redirect("/arena".to_owned()));
        };
        let Some(view) = state.arena_turn_view(&current)? else {
            return Ok(ArenaPairRender::Redirect("/arena".to_owned()));
        };
        let lookahead = page
            .lookahead
            .as_ref()
            .map(|turn| arena_lookahead_markup(&state, turn))
            .transpose()?
            .flatten();
        Ok(ArenaPairRender::Pair(Box::new(ArenaPairPage {
            current,
            view,
            lookahead,
            external_status: state.external_status()?,
        })))
    })
    .await
    .map_err(|error| anyhow::anyhow!("joining arena pair render task: {error:#}"))?
}

async fn arena_prefetch_payload_blocking(
    state: SharedAppState,
    known_turn_ids: HashSet<ArenaTurnId>,
    excluded_visual_keys: HashSet<VisualKey>,
) -> anyhow::Result<serde_json::Value> {
    tokio::task::spawn_blocking(move || {
        let turn = state.arena_prefetch_turn(&known_turn_ids, &excluded_visual_keys)?;
        arena_optional_turn_payload(&state, turn.as_ref())
    })
    .await
    .map_err(|error| anyhow::anyhow!("joining arena prefetch task: {error:#}"))?
}

pub(super) async fn api_arena_next(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ArenaNextQuery>,
) -> WebResult<Response> {
    let excluded_visual_count = query.exclude_visual_keys().len();
    let known_turn_count = query.known_turn_ids().len();
    let span = info_span!(
        "arena.prefetch.next",
        excluded_visual_count,
        known_turn_count,
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let excluded_visual_keys = query
            .exclude_visual_keys()
            .into_iter()
            .map(VisualKey)
            .collect::<HashSet<_>>();
        let known_turn_ids = query
            .known_turn_ids()
            .into_iter()
            .map(ArenaTurnId)
            .collect::<HashSet<_>>();
        let payload =
            arena_prefetch_payload_blocking(state, known_turn_ids, excluded_visual_keys).await?;
        Ok(no_store_response(axum::Json(payload).into_response()))
    }
    .instrument(span)
    .await
}

pub(super) async fn rotate(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    Form(form): Form<RotateForm>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let _span = info_span!(
        "arena.action.rotate",
        pair_href = %pair_href,
        asset_handle = %form.asset_id,
        direction = form.direction,
        next_rotation = form.next_rotation.unwrap_or_default(),
    )
    .entered();
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &handle])? {
        return Ok(response);
    }
    state.rotate_arena_handle(&handle, form.direction)?;
    let target = RedirectTarget::ArenaPair { left, right };
    Ok(Redirect::to(&target.href()).into_response())
}

pub(super) async fn heart(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    Form(form): Form<HeartAssetForm>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let _span = info_span!(
        "arena.action.heart",
        pair_href = %pair_href,
        asset_handle = %form.asset_id,
        active = form.active,
    )
    .entered();
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &handle])? {
        return Ok(response);
    }
    let target = match &handle {
        ArenaHandle::Local(asset_id) => {
            state.set_heart_asset(asset_id, form.active)?;
            if !state.quality_refresh_is_inline()? {
                state.schedule_quality_model_refresh();
            }
            RedirectTarget::ArenaPair { left, right }
        }
        ArenaHandle::Remote(_) => {
            let target = state.set_heart_arena_handle(&handle, form.active)?;
            if !state.quality_refresh_is_inline()? {
                state.schedule_quality_model_refresh();
            }
            target
        }
    };
    Ok(Redirect::to(&target.href()).into_response())
}

pub(super) async fn hide(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<HideForm>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let accept_json = accepts_json(&headers);
    let span = info_span!(
        "arena.action.hide",
        pair_href = %pair_href,
        asset_handle = %form.asset_id,
        cluster_size = form.cluster_ids.len(),
        hide = form.hide,
        accept_json = %accept_json,
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
        let cluster_ids: Vec<RemoteItemId> = form
            .cluster_ids
            .iter()
            .map(|&id| RemoteItemId(id))
            .collect();
        let remote_action = matches!(handle, ArenaHandle::Remote(_));
        let command = ArenaCommand::Hide {
            command_id: ArenaCommandId(form.command.command_id),
            expected_revision: ArenaRevision(form.command.arena_revision),
            expected_sampler_epoch: ArenaSamplerEpoch(form.command.arena_sampler_epoch),
            turn_id: ArenaTurnId(form.command.turn_id),
            action_token: ArenaActionToken(form.command.action_token),
            handle,
            hidden: form.hide,
            cluster_ids,
        };
        let outcome = apply_arena_command_blocking(state.clone(), command, "hide").await?;
        if remote_action {
            state.schedule_quality_model_refresh();
        }
        arena_command_response(state, outcome, accept_json).await
    }
    .instrument(span)
    .await
}

pub(super) async fn veto_thread(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<HandleForm>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let accept_json = accepts_json(&headers);
    let span = info_span!(
        "arena.action.veto_thread",
        pair_href = %pair_href,
        asset_handle = %form.asset_id,
        accept_json = %accept_json,
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
        let command = ArenaCommand::VetoThread {
            command_id: ArenaCommandId(form.command.command_id),
            expected_revision: ArenaRevision(form.command.arena_revision),
            expected_sampler_epoch: ArenaSamplerEpoch(form.command.arena_sampler_epoch),
            turn_id: ArenaTurnId(form.command.turn_id),
            action_token: ArenaActionToken(form.command.action_token),
            handle,
        };
        let outcome = apply_arena_command_blocking(state.clone(), command, "veto_thread").await?;
        arena_command_response(state, outcome, accept_json).await
    }
    .instrument(span)
    .await
}

pub(super) async fn lock_thread(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<ThreadLockForm>,
) -> WebResult<Response> {
    let pair_href = format!("/arena/{left_id}/{right_id}");
    let accept_json = accepts_json(&headers);
    let span = info_span!(
        "arena.action.lock_thread",
        pair_href = %pair_href,
        asset_handle = %form.asset_id,
        active = form.active,
        accept_json = %accept_json,
    );
    async move {
        let Some(state) = state.ready_app() else {
            return Ok(service_unavailable_response());
        };
        let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
        let command = ArenaCommand::LockThread {
            command_id: ArenaCommandId(form.command.command_id),
            expected_revision: ArenaRevision(form.command.arena_revision),
            expected_sampler_epoch: ArenaSamplerEpoch(form.command.arena_sampler_epoch),
            turn_id: ArenaTurnId(form.command.turn_id),
            action_token: ArenaActionToken(form.command.action_token),
            handle,
            active: form.active,
        };
        let outcome = apply_arena_command_blocking(state.clone(), command, "lock_thread").await?;
        arena_command_response(state, outcome, accept_json).await
    }
    .instrument(span)
    .await
}

struct ArenaLookaheadMarkup {
    href: String,
    turn_id: String,
    action_token: String,
    revision: u64,
    sampler_epoch: u64,
    visual_keys: Vec<String>,
    stage: Markup,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct ArenaNextQuery {
    exclude_visual_keys: Option<String>,
    known_turn_ids: Option<String>,
}

impl ArenaNextQuery {
    fn exclude_visual_keys(&self) -> Vec<String> {
        self.exclude_visual_keys
            .as_deref()
            .map(|value| {
                value
                    .split(',')
                    .filter(|entry| !entry.is_empty())
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn known_turn_ids(&self) -> Vec<String> {
        self.known_turn_ids
            .as_deref()
            .map(|value| {
                value
                    .split(',')
                    .filter(|entry| !entry.is_empty())
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn accepts_json(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"))
}

async fn arena_command_response(
    state: SharedAppState,
    outcome: ArenaCommandOutcome,
    accept_json: bool,
) -> WebResult<Response> {
    let mut payload = arena_optional_turn_payload_blocking(state, outcome.current).await?;
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "status".to_owned(),
            serde_json::json!(match outcome.status {
                ArenaCommandStatus::Applied => "applied",
                ArenaCommandStatus::Replayed => "replayed",
                ArenaCommandStatus::Stale => "stale",
            }),
        );
        object.insert(
            "stale".to_owned(),
            serde_json::json!(matches!(outcome.status, ArenaCommandStatus::Stale)),
        );
    }
    if !accept_json {
        let href = payload
            .get("href")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("/arena");
        return Ok(Redirect::to(href).into_response());
    }
    let mut response = no_store_response(axum::Json(payload).into_response());
    if matches!(outcome.status, ArenaCommandStatus::Stale) {
        *response.status_mut() = StatusCode::CONFLICT;
    }
    Ok(response)
}

async fn arena_optional_turn_payload_blocking(
    state: SharedAppState,
    turn: Option<ArenaTurn>,
) -> anyhow::Result<serde_json::Value> {
    tokio::task::spawn_blocking(move || arena_optional_turn_payload(&state, turn.as_ref()))
        .await
        .map_err(|error| anyhow::anyhow!("joining arena payload task: {error:#}"))?
}

fn arena_optional_turn_payload(
    state: &SharedAppState,
    turn: Option<&ArenaTurn>,
) -> anyhow::Result<serde_json::Value> {
    let Some(turn) = turn else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": "/arena",
        }));
    };
    arena_turn_payload(state, turn)
}

fn arena_turn_payload(
    state: &SharedAppState,
    turn: &ArenaTurn,
) -> anyhow::Result<serde_json::Value> {
    let Some(view) = state.arena_turn_view(turn)? else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": turn.href(),
        }));
    };
    let Some(pair) = view.pair else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": turn.href(),
        }));
    };
    Ok(serde_json::json!({
        "ok": true,
        "empty": false,
        "href": turn.href(),
        "turnId": turn.id().0.clone(),
        "actionToken": turn.action_token().0.clone(),
        "revision": turn.revision().0,
        "samplerEpoch": turn.sampler_epoch().0,
        "visualKeys": pair.visual_keys().into_iter().map(|visual_key| visual_key.0).collect::<Vec<_>>(),
        "html": arena_stage_markup(turn, &pair, view.cluster.as_ref()).into_string(),
    }))
}

fn arena_lookahead_markup(
    state: &SharedAppState,
    turn: &ArenaTurn,
) -> anyhow::Result<Option<ArenaLookaheadMarkup>> {
    let Some(view) = state.arena_turn_view(turn)? else {
        return Ok(None);
    };
    let Some(pair) = view.pair else {
        return Ok(None);
    };
    Ok(Some(ArenaLookaheadMarkup {
        href: turn.href(),
        turn_id: turn.id().0.clone(),
        action_token: turn.action_token().0.clone(),
        revision: turn.revision().0,
        sampler_epoch: turn.sampler_epoch().0,
        visual_keys: pair
            .visual_keys()
            .into_iter()
            .map(|visual_key| visual_key.0)
            .collect(),
        stage: arena_stage_markup(turn, &pair, view.cluster.as_ref()),
    }))
}

fn arena_markup(
    turn: Option<ArenaTurn>,
    view: ArenaView,
    lookahead: Option<ArenaLookaheadMarkup>,
    _external_status: ExternalArenaStatus,
) -> Markup {
    html! {
        @if let (Some(turn), Some(pair)) = (turn, view.pair) {
            @let pair_href = arena_pair_href(&pair);
            @let current_visual_keys = pair.visual_keys().into_iter().map(|visual_key| visual_key.0).collect::<Vec<_>>().join(",");
            @let lookahead_visual_keys = lookahead.as_ref().map(|next| next.visual_keys.join(",")).unwrap_or_default();
            div.arena-stage-shell {
                div.arena-stage-layer.is-current data-href=(pair_href) data-turn-id=(turn.id().0.as_str()) data-action-token=(turn.action_token().0.as_str()) data-arena-revision=(turn.revision().0) data-arena-sampler-epoch=(turn.sampler_epoch().0) data-visual-keys=(current_visual_keys) {
                    (arena_stage_markup(&turn, &pair, view.cluster.as_ref()))
                }
                div.arena-stage-layer.is-lookahead.is-hidden data-href=(lookahead.as_ref().map_or("", |next| next.href.as_str())) data-turn-id=(lookahead.as_ref().map_or("", |next| next.turn_id.as_str())) data-action-token=(lookahead.as_ref().map_or("", |next| next.action_token.as_str())) data-arena-revision=(lookahead.as_ref().map_or(0, |next| next.revision)) data-arena-sampler-epoch=(lookahead.as_ref().map_or(0, |next| next.sampler_epoch)) data-visual-keys=(lookahead_visual_keys) aria-hidden="true" {
                    @if let Some(lookahead) = lookahead {
                        (lookahead.stage)
                    }
                }
            }
            (script_block())
        } @else {
            section.empty-state {
                p { "Need at least two visible images." }
            }
        }
    }
}

fn arena_stage_markup(
    turn: &ArenaTurn,
    pair: &crate::model::ArenaPair,
    cluster: Option<&DuplicateCluster>,
) -> Markup {
    let left = pair.left.handle();
    let right = pair.right.handle();
    html! {
        section.arena-stage {
            (arena_panel(turn, &pair.left, &left, &right, None))
            (arena_panel(turn, &pair.right, &left, &right, cluster))
        }
    }
}

fn arena_pair_href(pair: &crate::model::ArenaPair) -> String {
    let left = pair.left.handle();
    let right = pair.right.handle();
    format!("/arena/{}/{}", left.slug(), right.slug())
}

fn arena_panel(
    turn: &ArenaTurn,
    card: &ArenaCard,
    pair_left: &ArenaHandle,
    pair_right: &ArenaHandle,
    cluster: Option<&DuplicateCluster>,
) -> Markup {
    let tooltip = arena_card_tooltip(card);
    let pair_href = format!("/arena/{}/{}", pair_left.slug(), pair_right.slug());
    let handle = card.handle();
    let hearted = arena_card_hearted(card);
    let frame_class = match card {
        ArenaCard::Local(_) => "frame swarm-surface-elevated",
        ArenaCard::Remote(_) => "frame swarm-surface-elevated foreign-frame",
    };
    let remote_thread_title = match card {
        ArenaCard::Remote(card) => Some(card.item.stream_title.as_str()),
        ArenaCard::Local(_) => None,
    };
    html! {
        article class=(frame_class) {
            form.vote-form data-arena-advance="buffered" action=(format!("{pair_href}/vote")) method="post" {
                (arena_command_inputs(turn))
                input type="hidden" name="left_id" value=(pair_left.slug());
                input type="hidden" name="right_id" value=(pair_right.slug());
                input type="hidden" name="winner_id" value=(handle.slug());
            }
            a.vote-surface href="#" title=(tooltip) {
                img
                    class="asset-image arena-image"
                    data-rotation=(arena_card_rotation(card))
                    src=(arena_card_src(card, AssetRendition::Arena))
                    alt="ranked image"
                    loading="eager"
                    decoding="async"
                    fetchpriority="high"
                    draggable="false";
            }
            @if let Some(thread_title) = remote_thread_title {
                @let stream_locked = match card {
                    ArenaCard::Remote(card) => card.stream_locked,
                    ArenaCard::Local(_) => false,
                };
                div.remote-thread-banner-rail title=(thread_title) {
                    @if matches!(card, ArenaCard::Remote(_)) {
                        form.remote-thread-lock-box.arena-lock-thread-form data-arena-advance="refresh-current" action=(format!("{pair_href}/lock-thread")) method="post" {
                            (arena_command_inputs(turn))
                            input type="hidden" name="asset_id" value=(handle.slug());
                            input type="hidden" name="active" value=(if stream_locked { "false" } else { "true" });
                            (tool_button(ImageToolKind::LockThread, true, stream_locked))
                        }
                    }
                    div.frame-banner.remote-thread-banner.swarm-frame-header {
                        span.remote-thread-banner-label { (thread_title) }
                    }
                }
            }
            @if let Some(cluster) = cluster {
                div.cluster-strip {
                    @for satellite in &cluster.satellites {
                        a.cluster-thumb
                            href=(format!("/arena/{}/remote_{}", pair_left.slug(), satellite.item.id.0))
                            title=(format!("d={:.3} · {}", satellite.distance, satellite.item.title)) {
                            img
                                src=(remote_src(&satellite.item, AssetRendition::Board))
                                alt="near-duplicate"
                                loading="lazy"
                                decoding="async"
                                draggable="false";
                        }
                    }
                }
            }
            div.frame-quality-row {
                @match card {
                    ArenaCard::Local(card) => {
                        (asset_quality_chips(card.quality))
                    }
                    ArenaCard::Remote(card) => {
                        (asset_quality_chips(card.quality))
                    }
                }
            }
            div.frame-tools {
                form action=(format!("{pair_href}/rotate")) method="post" {
                    input type="hidden" name="next_rotation" value=(((arena_card_rotation(card) + 3).rem_euclid(4)));
                    input type="hidden" name="asset_id" value=(handle.slug());
                    input type="hidden" name="direction" value="-1";
                    (tool_button(ImageToolKind::RotateLeft, false, false))
                }
                form action=(format!("{pair_href}/rotate")) method="post" {
                    input type="hidden" name="next_rotation" value=(((arena_card_rotation(card) + 1).rem_euclid(4)));
                    input type="hidden" name="asset_id" value=(handle.slug());
                    input type="hidden" name="direction" value="1";
                    (tool_button(ImageToolKind::RotateRight, false, false))
                }
                form action=(format!("{pair_href}/heart")) method="post" {
                    input type="hidden" name="asset_id" value=(handle.slug());
                    input type="hidden" name="active" value="true";
                    (tool_button(ImageToolKind::Heart, false, hearted))
                }
                form.arena-hide-form data-arena-advance="buffered" action=(format!("{pair_href}/hide")) method="post" {
                    (arena_command_inputs(turn))
                    input type="hidden" name="asset_id" value=(handle.slug());
                    input type="hidden" name="hide" value=(if arena_card_hidden(card) { "false" } else { "true" });
                    @if let Some(cluster) = cluster {
                        @if !cluster.satellites.is_empty() {
                            input type="hidden" name="cluster_ids" value=(
                                cluster.satellites.iter()
                                    .map(|s| s.item.id.0.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            );
                        }
                    }
                    (tool_button(ImageToolKind::Hide, false, false))
                }
                @if let ArenaCard::Local(card) = card {
                    (asset_domain_controls(&card.asset.id, card.domain, false, false))
                }
                @if matches!(card, ArenaCard::Remote(_)) {
                    form.arena-veto-thread-form data-arena-advance="authoritative" action=(format!("{pair_href}/veto-thread")) method="post" {
                        (arena_command_inputs(turn))
                        input type="hidden" name="asset_id" value=(handle.slug());
                        (tool_button(ImageToolKind::VetoThread, false, false))
                    }
                }
            }
        }
    }
}

fn arena_command_inputs(turn: &ArenaTurn) -> Markup {
    html! {
        input type="hidden" name="command_id" value=(ArenaCommandId::forge().0);
        input type="hidden" name="turn_id" value=(turn.id().0.as_str());
        input type="hidden" name="action_token" value=(turn.action_token().0.as_str());
        input type="hidden" name="arena_revision" value=(turn.revision().0);
        input type="hidden" name="arena_sampler_epoch" value=(turn.sampler_epoch().0);
    }
}

pub(super) fn arena_mode_menu(external_status: ExternalArenaStatus) -> Markup {
    html! {
        details.rail-menu {
            summary.swarm-frame-header { "arena" }
            div.menu-panel.swarm-frame {
                div.menu-meta {
                    @if !external_status.sources.is_empty() {
                        div { "weighted upstream mix" }
                        div { (format!("{} live / {} blocked / {} cached", external_status.active_streams, external_status.blocked_streams, external_status.cached_items)) }
                        @for source in &external_status.sources {
                            div { (format!("{} × {:.2}", source.label, source.weight)) }
                        }
                    } @else {
                        div { "external off" }
                    }
                }
                div.menu-divider aria-hidden="true" {}
                form.menu-probability-form action="/arena/explore" method="post" {
                    label.menu-probability-label for="arena-explore" { "arena explore / exploit" }
                    div.menu-probability-row {
                        input
                            id="arena-explore"
                            class="menu-probability-slider"
                            type="range"
                            name="percent"
                            min="0"
                            max="100"
                            step="5"
                            value=(external_status.arena_explore_percent)
                            oninput="this.nextElementSibling.value = this.value + '% explore'";
                        output.menu-probability-value { (format!("{}% explore", external_status.arena_explore_percent)) }
                        button type="submit" class="menu-probability-apply" { "set" }
                    }
                }
                form.menu-probability-form action="/external/probability" method="post" {
                    label.menu-probability-label for="external-probability" { "external arena probability" }
                    div.menu-probability-row {
                        input
                            id="external-probability"
                            class="menu-probability-slider"
                            type="range"
                            name="percent"
                            min="0"
                            max="100"
                            step="5"
                            value=(external_status.external_probability)
                            oninput="this.nextElementSibling.value = this.value + '%'";
                        output.menu-probability-value { (format!("{}%", external_status.external_probability)) }
                        button type="submit" class="menu-probability-apply" { "set" }
                    }
                }
                form.menu-probability-form action="/external/dedup-radius" method="post" {
                    label.menu-probability-label for="dedup-radius" { "near-duplicate sensitivity" }
                    div.menu-probability-row {
                        input
                            id="dedup-radius"
                            class="menu-probability-slider"
                            type="range"
                            name="percent"
                            min="0"
                            max="100"
                            step="5"
                            value=(external_status.dedup_radius_percent)
                            oninput="this.nextElementSibling.value = this.value + '%'";
                        output.menu-probability-value { (format!("{}%", external_status.dedup_radius_percent)) }
                        button type="submit" class="menu-probability-apply" { "set" }
                    }
                }
            }
        }
    }
}

fn arena_card_src(card: &ArenaCard, rendition: AssetRendition) -> String {
    match card {
        ArenaCard::Local(card) => asset_src(&card.asset, rendition),
        ArenaCard::Remote(card) => remote_src(&card.item, rendition),
    }
}

fn arena_card_rotation(card: &ArenaCard) -> i32 {
    match card {
        ArenaCard::Local(card) => card.asset.rotation_quarters,
        ArenaCard::Remote(card) => card.item.rotation_quarters,
    }
}

fn arena_card_hidden(card: &ArenaCard) -> bool {
    match card {
        ArenaCard::Local(card) => card.asset.hidden,
        ArenaCard::Remote(_) => false,
    }
}

fn arena_card_hearted(card: &ArenaCard) -> bool {
    match card {
        ArenaCard::Local(card) => card.hearted,
        ArenaCard::Remote(card) => card.hearted,
    }
}

fn local_arena_tooltip(card: &crate::model::ArenaLocalCard) -> String {
    let asset = &card.asset;
    let file_name = asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let mut lines = vec![
        file_name.to_owned(),
        format!("session {:+.2}", card.utility),
        format!("global {:+.2}", asset.alpha),
        format!("{} wins / {} duels", asset.win_count, asset.compare_count),
        format!(
            "certainty {:.0}%",
            crate::model::certainty(asset.compare_count) * 100.0
        ),
    ];
    lines.extend(asset_quality_lines(card.quality));
    lines.join("\n")
}

fn remote_tooltip(item: &RemoteItemRecord, session_utility: f32) -> String {
    format!(
        "{}\n{}\nthread title: {}\nthread {}\npost {}\nsession {session_utility:+.2}",
        item.title, item.source_key, item.stream_title, item.thread_no, item.post_no
    )
}

fn arena_card_tooltip(card: &ArenaCard) -> String {
    match card {
        ArenaCard::Local(card) => local_arena_tooltip(card),
        ArenaCard::Remote(card) => remote_tooltip(&card.item, card.utility),
    }
}
