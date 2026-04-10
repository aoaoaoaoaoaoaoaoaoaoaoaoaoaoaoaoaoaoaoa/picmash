use super::*;
use crate::identity::VisualKey;
use std::collections::HashSet;

pub(super) async fn arena_root(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/arena");
    let target = state.arena_target()?;
    match target {
        RedirectTarget::ArenaRoot => Ok(render_markup(routed_layout(
            "arena-page",
            PageGeometry::Viewport,
            NavPage::Arena,
            Some(arena_mode_menu(state.external_status()?)),
            arena_markup(
                state.arena_empty()?,
                arena_lookahead_markup(&state, None, &HashSet::new())?,
                None,
                state.external_status()?,
            ),
        ))),
        RedirectTarget::ArenaPair { .. } => Ok(Redirect::to(&target.href()).into_response()),
        RedirectTarget::FacemashRoot
        | RedirectTarget::FacemashPair { .. }
        | RedirectTarget::ExploreRoot { .. }
        | RedirectTarget::ExploreTriad { .. } => Ok(Redirect::to("/arena").into_response()),
    }
}

pub(super) async fn arena_reroll() -> WebResult<Response> {
    Ok(Redirect::to("/arena").into_response())
}

pub(super) async fn arena(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/arena/pair");
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    let Some(view) = state.arena_pair(&left, &right)? else {
        let target = state.arena_target()?;
        return Ok(Redirect::to(&target.href()).into_response());
    };
    let current_excluded = view
        .pair
        .as_ref()
        .map(crate::model::ArenaPair::visual_keys)
        .unwrap_or_default();
    let lookahead = arena_lookahead_markup(&state, None, &current_excluded)?;
    let mut preserved_excluded = current_excluded.clone();
    if let Some(lookahead_keys) = lookahead.as_ref() {
        preserved_excluded.extend(lookahead_keys.visual_keys.iter().cloned().map(VisualKey));
    }
    let preserved_lookahead = view
        .pair
        .as_ref()
        .and_then(arena_pair_local_anchor)
        .map_or(Ok(None), |anchor| {
            let mut excluded = preserved_excluded.clone();
            let anchor_visual_key = view
                .pair
                .as_ref()
                .and_then(crate::model::ArenaPair::local_anchor_visual_key)
                .cloned();
            if let Some(anchor_visual_key) = anchor_visual_key {
                excluded.remove(&anchor_visual_key);
            }
            arena_lookahead_markup(&state, Some(anchor), &excluded)
        })?;
    Ok(render_markup(routed_layout(
        "arena-page",
        PageGeometry::Viewport,
        NavPage::Arena,
        Some(arena_mode_menu(state.external_status()?)),
        arena_markup(
            view,
            lookahead,
            preserved_lookahead,
            state.external_status()?,
        ),
    )))
}

pub(super) async fn vote(
    State(state): State<SharedRuntimeState>,
    Path((_left_id, _right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<VoteForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let left = ArenaHandle::from_str(&form.left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&form.right_id).map_err(anyhow::Error::msg)?;
    let winner = ArenaHandle::from_str(&form.winner_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &winner])? {
        return Ok(response);
    }
    let refresh_state = state.clone();
    let target = tokio::task::spawn_blocking(move || state.vote(&left, &right, &winner))
        .await
        .map_err(|error| anyhow::anyhow!("joining arena vote task: {error:#}"))??;
    if !refresh_state.quality_refresh_is_inline()? {
        refresh_state.schedule_quality_model_refresh();
    }
    if headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"))
    {
        return Ok(no_store_response(
            axum::Json(serde_json::json!({ "ok": true })).into_response(),
        ));
    }
    Ok(Redirect::to(&target.href()).into_response())
}

pub(super) async fn api_arena_next(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ArenaNextQuery>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let local_anchor = query.anchor.as_deref().map(|slug| AssetId(slug.to_owned()));
    let excluded_visual_keys = query
        .exclude_visual_keys()
        .into_iter()
        .map(VisualKey)
        .collect::<HashSet<_>>();
    let target = match local_anchor.as_ref() {
        Some(anchor) => state.arena_prefetch_target_preserving_local_anchor_excluding(
            Some(anchor),
            &excluded_visual_keys,
        )?,
        None => state.arena_prefetch_target_excluding(&excluded_visual_keys)?,
    };
    Ok(no_store_response(
        axum::Json(arena_target_payload(&state, &target)?).into_response(),
    ))
}

pub(super) async fn rotate(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    Form(form): Form<RotateForm>,
) -> WebResult<Response> {
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
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &handle])? {
        return Ok(response);
    }
    let cluster_ids: Vec<RemoteItemId> = form
        .cluster_ids
        .iter()
        .map(|&id| RemoteItemId(id))
        .collect();
    let target = state.hide_arena_handle(&handle, form.hide, &cluster_ids, &left, &right)?;
    if matches!(handle, ArenaHandle::Remote(_)) {
        state.schedule_quality_model_refresh();
    }
    if headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"))
    {
        return Ok(no_store_response(
            axum::Json(arena_target_payload(&state, &target)?).into_response(),
        ));
    }
    Ok(Redirect::to(&target.href()).into_response())
}

pub(super) async fn veto_thread(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<HandleForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &handle])? {
        return Ok(response);
    }
    let target = state.veto_external_thread_for_handle(&handle, &left, &right)?;
    if headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"))
    {
        return Ok(no_store_response(
            axum::Json(arena_target_payload(&state, &target)?).into_response(),
        ));
    }
    Ok(Redirect::to(&target.href()).into_response())
}

pub(super) async fn lock_thread(
    State(state): State<SharedRuntimeState>,
    Path((left_id, right_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<ThreadLockForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let handle = ArenaHandle::from_str(&form.asset_id).map_err(anyhow::Error::msg)?;
    let left = ArenaHandle::from_str(&left_id).map_err(anyhow::Error::msg)?;
    let right = ArenaHandle::from_str(&right_id).map_err(anyhow::Error::msg)?;
    if let Some(response) = stale_arena_action_response(&state, [&left, &right, &handle])? {
        return Ok(response);
    }
    let target =
        state.set_external_subsource_lock_for_handle(&handle, form.active, &left, &right)?;
    if headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"))
    {
        return Ok(no_store_response(
            axum::Json(arena_target_payload(&state, &target)?).into_response(),
        ));
    }
    Ok(Redirect::to(&target.href()).into_response())
}

struct ArenaLookaheadMarkup {
    href: String,
    local_anchor: String,
    local_anchor_visual_key: String,
    visual_keys: Vec<String>,
    stage: Markup,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct ArenaNextQuery {
    anchor: Option<String>,
    exclude_visual_keys: Option<String>,
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
}

fn arena_pair_local_anchor(pair: &crate::model::ArenaPair) -> Option<&AssetId> {
    match (&pair.left, &pair.right) {
        (ArenaCard::Local(card), ArenaCard::Remote(_))
        | (ArenaCard::Remote(_), ArenaCard::Local(card)) => Some(&card.asset.id),
        _ => None,
    }
}

fn arena_target_payload(
    state: &SharedAppState,
    target: &RedirectTarget,
) -> anyhow::Result<serde_json::Value> {
    let RedirectTarget::ArenaPair { left, right } = target else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": target.href(),
        }));
    };
    let Some(view) = state.arena_pair(left, right)? else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": target.href(),
        }));
    };
    let Some(pair) = view.pair else {
        return Ok(serde_json::json!({
            "ok": true,
            "empty": true,
            "href": target.href(),
        }));
    };
    Ok(serde_json::json!({
        "ok": true,
        "empty": false,
        "href": target.href(),
        "localAnchor": arena_pair_local_anchor(&pair).map(|asset_id| asset_id.0.clone()).unwrap_or_default(),
        "localAnchorVisualKey": pair.local_anchor_visual_key().map(|visual_key| visual_key.0.clone()).unwrap_or_default(),
        "visualKeys": pair.visual_keys().into_iter().map(|visual_key| visual_key.0).collect::<Vec<_>>(),
        "html": arena_stage_markup(&pair, view.cluster.as_ref()).into_string(),
    }))
}

fn arena_lookahead_markup(
    state: &SharedAppState,
    local_anchor: Option<&AssetId>,
    excluded_visual_keys: &HashSet<VisualKey>,
) -> anyhow::Result<Option<ArenaLookaheadMarkup>> {
    let target = match local_anchor {
        Some(anchor) => state.arena_prefetch_target_preserving_local_anchor_excluding(
            Some(anchor),
            excluded_visual_keys,
        )?,
        None => state.arena_prefetch_target_excluding(excluded_visual_keys)?,
    };
    let RedirectTarget::ArenaPair {
        ref left,
        ref right,
    } = target
    else {
        return Ok(None);
    };
    let Some(view) = state.arena_pair(left, right)? else {
        return Ok(None);
    };
    let Some(pair) = view.pair else {
        return Ok(None);
    };
    Ok(Some(ArenaLookaheadMarkup {
        href: target.href(),
        local_anchor: arena_pair_local_anchor(&pair)
            .map(|asset_id| asset_id.0.clone())
            .unwrap_or_default(),
        local_anchor_visual_key: pair
            .local_anchor_visual_key()
            .map(|visual_key| visual_key.0.clone())
            .unwrap_or_default(),
        visual_keys: pair
            .visual_keys()
            .into_iter()
            .map(|visual_key| visual_key.0)
            .collect(),
        stage: arena_stage_markup(&pair, view.cluster.as_ref()),
    }))
}

fn arena_markup(
    view: ArenaView,
    lookahead: Option<ArenaLookaheadMarkup>,
    preserved_lookahead: Option<ArenaLookaheadMarkup>,
    _external_status: ExternalArenaStatus,
) -> Markup {
    html! {
        @if let Some(pair) = view.pair {
            @let pair_href = arena_pair_href(&pair);
            @let local_anchor = arena_pair_local_anchor(&pair).map_or("", |asset_id| asset_id.0.as_str());
            @let current_visual_keys = pair.visual_keys().into_iter().map(|visual_key| visual_key.0).collect::<Vec<_>>().join(",");
            @let local_anchor_visual_key = pair.local_anchor_visual_key().map_or("", |visual_key| visual_key.0.as_str());
            @let lookahead_visual_keys = lookahead.as_ref().map(|next| next.visual_keys.join(",")).unwrap_or_default();
            @let preserved_visual_keys = preserved_lookahead.as_ref().map(|next| next.visual_keys.join(",")).unwrap_or_default();
            div.arena-stage-shell {
                div.arena-stage-layer.is-current data-href=(pair_href) data-local-anchor=(local_anchor) data-local-anchor-visual-key=(local_anchor_visual_key) data-visual-keys=(current_visual_keys) {
                    (arena_stage_markup(&pair, view.cluster.as_ref()))
                }
                div.arena-stage-layer.is-lookahead.is-hidden data-href=(lookahead.as_ref().map_or("", |next| next.href.as_str())) data-local-anchor=(lookahead.as_ref().map_or("", |next| next.local_anchor.as_str())) data-local-anchor-visual-key=(lookahead.as_ref().map_or("", |next| next.local_anchor_visual_key.as_str())) data-visual-keys=(lookahead_visual_keys) aria-hidden="true" {
                    @if let Some(lookahead) = lookahead {
                        (lookahead.stage)
                    }
                }
                div.arena-stage-layer.is-preserved-lookahead.is-hidden data-href=(preserved_lookahead.as_ref().map_or("", |next| next.href.as_str())) data-local-anchor=(preserved_lookahead.as_ref().map_or("", |next| next.local_anchor.as_str())) data-local-anchor-visual-key=(preserved_lookahead.as_ref().map_or("", |next| next.local_anchor_visual_key.as_str())) data-visual-keys=(preserved_visual_keys) aria-hidden="true" {
                    @if let Some(lookahead) = preserved_lookahead {
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
    pair: &crate::model::ArenaPair,
    cluster: Option<&DuplicateCluster>,
) -> Markup {
    let left = pair.left.handle();
    let right = pair.right.handle();
    html! {
        section.arena-stage {
            (arena_panel(&pair.left, &left, &right, None))
            (arena_panel(&pair.right, &left, &right, cluster))
        }
    }
}

fn arena_pair_href(pair: &crate::model::ArenaPair) -> String {
    let left = pair.left.handle();
    let right = pair.right.handle();
    format!("/arena/{}/{}", left.slug(), right.slug())
}

fn arena_panel(
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
                @let hide_advance = match card {
                    ArenaCard::Remote(_) => "buffered-preserve",
                    ArenaCard::Local(_) => "authoritative",
                };
                form.arena-hide-form data-arena-advance=(hide_advance) action=(format!("{pair_href}/hide")) method="post" {
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
                    @let veto_advance = match card {
                        ArenaCard::Remote(card) if card.stream_locked => "authoritative",
                        ArenaCard::Remote(_) => "buffered-preserve",
                        ArenaCard::Local(_) => "authoritative",
                    };
                    form.arena-veto-thread-form data-arena-advance=(veto_advance) action=(format!("{pair_href}/veto-thread")) method="post" {
                        input type="hidden" name="asset_id" value=(handle.slug());
                        (tool_button(ImageToolKind::VetoThread, false, false))
                    }
                }
            }
        }
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
