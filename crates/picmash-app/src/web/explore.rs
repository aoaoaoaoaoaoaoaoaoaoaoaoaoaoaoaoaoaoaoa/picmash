use super::*;

pub(super) async fn triad_root(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    match ready_app_or_snapshot(&state) {
        Ok(_) => {}
        Err(snapshot) => return Ok(boot_response(snapshot)),
    }
    log_site_loaded("/triad");
    Ok(render_markup(frontend_host_markup(
        NavPage::Triad,
        "triad-page",
        "triad",
    )))
}

pub(super) async fn triad_reroll() -> WebResult<Response> {
    Ok(Redirect::to("/triad").into_response())
}

pub(super) async fn api_triad_bootstrap(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    Ok(no_store_response(
        Json(triad_bootstrap_payload(&state, query.triad_assets()?)?).into_response(),
    ))
}

pub(super) async fn api_triad_train(
    State(state): State<SharedRuntimeState>,
    Json(request): Json<TriadTrainRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let target = state.train_similarity(
        &AssetId(request.triad.asset_a_id),
        &AssetId(request.triad.asset_b_id),
        &AssetId(request.triad.asset_c_id),
        request.choice,
        None,
        ExploreMapMode::Learned,
    )?;
    Ok(no_store_response(
        Json(triad_bootstrap_from_target(&state, &target)?).into_response(),
    ))
}

pub(super) async fn api_triad_rotate(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<RotateAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        return Ok(no_store_response(
            Json(triad_bootstrap_payload(&state, None)?).into_response(),
        ));
    }
    state.rotate_asset(&AssetId(request.asset_id), request.direction)?;
    Ok(no_store_response(
        Json(triad_bootstrap_payload(&state, query.triad_assets()?)?).into_response(),
    ))
}

pub(super) async fn api_triad_heart(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<HeartAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        return Ok(no_store_response(
            Json(triad_bootstrap_payload(&state, None)?).into_response(),
        ));
    }
    state.set_heart_asset(&AssetId(request.asset_id), request.active)?;
    if !state.quality_refresh_is_inline()? {
        state.schedule_quality_model_refresh();
    }
    Ok(no_store_response(
        Json(triad_bootstrap_payload(&state, query.triad_assets()?)?).into_response(),
    ))
}

pub(super) async fn api_triad_domain(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<crate::api::AssetDomainRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        return Ok(no_store_response(
            Json(triad_bootstrap_payload(&state, None)?).into_response(),
        ));
    }
    state.set_asset_domain_label(&AssetId(request.asset_id), request.label)?;
    state.schedule_quality_model_refresh();
    Ok(no_store_response(
        Json(triad_bootstrap_payload(&state, query.triad_assets()?)?).into_response(),
    ))
}

pub(super) async fn api_triad_hide(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<HideAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        return Ok(no_store_response(
            Json(triad_bootstrap_payload(&state, None)?).into_response(),
        ));
    }
    state.hide_asset_from_board(&AssetId(request.asset_id))?;
    Ok(no_store_response(
        Json(triad_bootstrap_payload(&state, query.triad_assets()?)?).into_response(),
    ))
}

pub(super) async fn explore_root(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    match ready_app_or_snapshot(&state) {
        Ok(_) => {}
        Err(snapshot) => return Ok(boot_response(snapshot)),
    }
    log_site_loaded("/explore");
    Ok(render_markup(frontend_host_markup(
        NavPage::Explore,
        "explore-page",
        "explore",
    )))
}

pub(super) async fn api_explore_bootstrap(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let map_mode = query.map_mode();
    let focus_id = query.focus_asset_id();
    Ok(no_store_response(
        Json(explore_bootstrap_payload(
            &state,
            query.triad_assets()?,
            focus_id.as_ref(),
            map_mode,
        )?)
        .into_response(),
    ))
}

pub(super) async fn api_explore_selection(
    State(state): State<SharedRuntimeState>,
    Path(asset_id): Path<String>,
    Query(query): Query<ClientRouteQuery>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let map_mode = query.map_mode();
    let triad = query.triad_assets()?;
    let Some(selection) = explore_selection_payload(&state, triad, &AssetId(asset_id), map_mode)?
    else {
        return Ok((StatusCode::NOT_FOUND, "missing explore selection").into_response());
    };
    Ok(no_store_response(Json(selection).into_response()))
}

pub(super) async fn api_explore_rotate(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<RotateAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        let map_mode = query.map_mode();
        let focus_id = query.focus_asset_id();
        return Ok(no_store_response(
            Json(explore_bootstrap_payload(
                &state,
                None,
                focus_id.as_ref(),
                map_mode,
            )?)
            .into_response(),
        ));
    }
    state.rotate_asset(&AssetId(request.asset_id), request.direction)?;
    let map_mode = query.map_mode();
    let focus_id = query.focus_asset_id();
    Ok(no_store_response(
        Json(explore_bootstrap_payload(
            &state,
            query.triad_assets()?,
            focus_id.as_ref(),
            map_mode,
        )?)
        .into_response(),
    ))
}

pub(super) async fn api_explore_heart(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<HeartAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        let map_mode = query.map_mode();
        let focus_id = query.focus_asset_id();
        return Ok(no_store_response(
            Json(explore_bootstrap_payload(
                &state,
                None,
                focus_id.as_ref(),
                map_mode,
            )?)
            .into_response(),
        ));
    }
    state.set_heart_asset(&AssetId(request.asset_id), request.active)?;
    if !state.quality_refresh_is_inline()? {
        state.schedule_quality_model_refresh();
    }
    let map_mode = query.map_mode();
    let focus_id = query.focus_asset_id();
    Ok(no_store_response(
        Json(explore_bootstrap_payload(
            &state,
            query.triad_assets()?,
            focus_id.as_ref(),
            map_mode,
        )?)
        .into_response(),
    ))
}

pub(super) async fn api_explore_domain(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<crate::api::AssetDomainRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        let map_mode = query.map_mode();
        let focus_id = query.focus_asset_id();
        return Ok(no_store_response(
            Json(explore_bootstrap_payload(
                &state,
                None,
                focus_id.as_ref(),
                map_mode,
            )?)
            .into_response(),
        ));
    }
    state.set_asset_domain_label(&AssetId(request.asset_id), request.label)?;
    state.schedule_quality_model_refresh();
    let map_mode = query.map_mode();
    let focus_id = query.focus_asset_id();
    Ok(no_store_response(
        Json(explore_bootstrap_payload(
            &state,
            query.triad_assets()?,
            focus_id.as_ref(),
            map_mode,
        )?)
        .into_response(),
    ))
}

pub(super) async fn api_explore_hide(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<ClientRouteQuery>,
    Json(request): Json<HideAssetRequestDto>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(request.asset_id.clone()))?
        .is_none()
    {
        let map_mode = query.map_mode();
        let focus_id = query.focus_asset_id();
        return Ok(no_store_response(
            Json(explore_bootstrap_payload(
                &state,
                None,
                focus_id.as_ref(),
                map_mode,
            )?)
            .into_response(),
        ));
    }
    state.hide_asset_from_board(&AssetId(request.asset_id))?;
    let map_mode = query.map_mode();
    let focus_id = query.focus_asset_id();
    Ok(no_store_response(
        Json(explore_bootstrap_payload(
            &state,
            query.triad_assets()?,
            focus_id.as_ref(),
            map_mode,
        )?)
        .into_response(),
    ))
}

fn triad_bootstrap_from_target(
    state: &SharedAppState,
    target: &RedirectTarget,
) -> anyhow::Result<TriadBootstrapDto> {
    match target {
        RedirectTarget::ExploreTriad {
            asset_a,
            asset_b,
            asset_c,
            ..
        } => triad_bootstrap_payload(
            state,
            Some([asset_a.clone(), asset_b.clone(), asset_c.clone()]),
        ),
        RedirectTarget::ExploreRoot { .. }
        | RedirectTarget::FacemashRoot
        | RedirectTarget::FacemashPair { .. }
        | RedirectTarget::ArenaRoot
        | RedirectTarget::ArenaPair { .. } => triad_bootstrap_payload(state, None),
    }
}

fn triad_bootstrap_payload(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
) -> anyhow::Result<TriadBootstrapDto> {
    let hearted = state.hearted_assets()?;
    let triad = resolved_triad(state, requested)?;
    Ok(match triad {
        Some(triad) => TriadBootstrapDto {
            triad: Some(triad_handle_dto(&triad)),
            assets: vec![
                triad_asset_dto(&triad.a, hearted.contains(&triad.a.asset.id)),
                triad_asset_dto(&triad.b, hearted.contains(&triad.b.asset.id)),
                triad_asset_dto(&triad.c, hearted.contains(&triad.c.asset.id)),
            ],
            empty_note: None,
        },
        None => TriadBootstrapDto {
            triad: None,
            assets: Vec::new(),
            empty_note: Some(
                state
                    .dino_status_note()
                    .unwrap_or(TRIAD_EMPTY_NOTE)
                    .to_owned(),
            ),
        },
    })
}

fn explore_bootstrap_payload(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
    focus_id: Option<&AssetId>,
    map_mode: ExploreMapMode,
) -> anyhow::Result<ExploreBootstrapDto> {
    let hearted = state.hearted_assets()?;
    let view = resolved_explore_view(state, requested, focus_id, map_mode)?;
    let active_focus = view
        .selection
        .as_ref()
        .map(|selection| selection.focus.asset.id.0.clone())
        .or_else(|| focus_id.map(|asset_id| asset_id.0.clone()));
    let points = view
        .points
        .iter()
        .map(explore_point_dto)
        .collect::<Vec<_>>();
    Ok(ExploreBootstrapDto {
        mode: view.map_mode,
        triad: view.triad.as_ref().map(triad_handle_dto),
        focus_id: active_focus,
        selection: view
            .selection
            .as_ref()
            .map(|selection| explore_selection_dto(selection, &hearted)),
        empty_note: if points.is_empty() {
            Some(
                state
                    .dino_status_note()
                    .unwrap_or(EXPLORE_EMPTY_NOTE)
                    .to_owned(),
            )
        } else {
            None
        },
        points,
    })
}

fn explore_selection_payload(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
    focus_id: &AssetId,
    map_mode: ExploreMapMode,
) -> anyhow::Result<Option<ExploreSelectionResponseDto>> {
    let hearted = state.hearted_assets()?;
    let Some(selection) = resolved_explore_panels(state, requested, Some(focus_id), map_mode)?
        .and_then(|panels| panels.selection)
    else {
        return Ok(None);
    };
    Ok(Some(ExploreSelectionResponseDto {
        mode: map_mode,
        focus_id: selection.focus.asset.id.0.clone(),
        selection: explore_selection_dto(&selection, &hearted),
    }))
}

fn resolved_explore_panels(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
    focus_id: Option<&AssetId>,
    map_mode: ExploreMapMode,
) -> anyhow::Result<Option<crate::model::ExplorePanels>> {
    if let Some([asset_a, asset_b, asset_c]) = requested
        && let Some(panels) =
            state.explore_panels(&asset_a, &asset_b, &asset_c, focus_id, map_mode)?
    {
        return Ok(Some(panels));
    }

    let RedirectTarget::ExploreTriad {
        asset_a,
        asset_b,
        asset_c,
        ..
    } = state.explore_target(focus_id, map_mode)?
    else {
        return Ok(None);
    };
    state.explore_panels(&asset_a, &asset_b, &asset_c, focus_id, map_mode)
}

fn resolved_triad(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
) -> anyhow::Result<Option<crate::model::ExploreTriad>> {
    Ok(
        resolved_explore_panels(state, requested, None, ExploreMapMode::Learned)?
            .and_then(|panels| panels.triad),
    )
}

fn resolved_explore_view(
    state: &SharedAppState,
    requested: Option<[AssetId; 3]>,
    focus_id: Option<&AssetId>,
    map_mode: ExploreMapMode,
) -> anyhow::Result<ExploreView> {
    if let Some([asset_a, asset_b, asset_c]) = requested
        && let Some(view) = state.explore_view(&asset_a, &asset_b, &asset_c, focus_id, map_mode)?
    {
        return Ok(view);
    }

    Ok(match state.explore_target(focus_id, map_mode)? {
        RedirectTarget::ExploreTriad {
            asset_a,
            asset_b,
            asset_c,
            ..
        } => state
            .explore_view(&asset_a, &asset_b, &asset_c, focus_id, map_mode)?
            .unwrap_or(state.explore_empty(map_mode)?),
        RedirectTarget::ExploreRoot { .. }
        | RedirectTarget::FacemashRoot
        | RedirectTarget::FacemashPair { .. }
        | RedirectTarget::ArenaRoot
        | RedirectTarget::ArenaPair { .. } => state.explore_empty(map_mode)?,
    })
}

fn triad_handle_dto(triad: &crate::model::ExploreTriad) -> TriadHandleDto {
    TriadHandleDto {
        asset_a_id: triad.a.asset.id.0.clone(),
        asset_b_id: triad.b.asset.id.0.clone(),
        asset_c_id: triad.c.asset.id.0.clone(),
    }
}

fn posterior_summary_dto(
    summary: crate::model::PosteriorSummary,
) -> crate::api::PosteriorSummaryDto {
    crate::api::PosteriorSummaryDto {
        mean: summary.mean,
        sigma: summary.sigma,
    }
}

fn quality_summary_dto(
    summary: crate::model::AssetQualitySummary,
) -> crate::api::AssetQualitySummaryDto {
    crate::api::AssetQualitySummaryDto {
        asset: posterior_summary_dto(summary.asset),
        baseline: posterior_summary_dto(summary.baseline),
        semantic: summary.semantic.map(posterior_summary_dto),
        vibe: summary.vibe.map(posterior_summary_dto),
        technical: summary.technical.map(posterior_summary_dto),
        face: summary.face.map(posterior_summary_dto),
    }
}

fn asset_domain_summary_dto(
    domain: crate::model::AssetDomainView,
) -> crate::api::AssetDomainSummaryDto {
    crate::api::AssetDomainSummaryDto {
        manual_label: domain.manual,
        predicted_label: domain.predicted.map(|prediction| prediction.label()),
        predicted_percent: domain
            .predicted
            .map(|prediction| prediction.display_percent()),
    }
}

fn triad_asset_dto(entry: &crate::model::ExploreEntry, hearted: bool) -> TriadAssetDto {
    let asset = &entry.asset;
    TriadAssetDto {
        asset_id: asset.id.0.clone(),
        name: selection_name(asset).to_owned(),
        full_src: asset_src(asset, AssetRendition::Preview),
        domain: asset_domain_summary_dto(entry.domain),
        hearted,
        compare_count: asset.compare_count,
        win_count: asset.win_count,
        quality: quality_summary_dto(entry.quality),
    }
}

fn focus_asset_dto(entry: &crate::model::ExploreEntry, hearted: bool) -> FocusAssetDto {
    let asset = &entry.asset;
    FocusAssetDto {
        asset_id: asset.id.0.clone(),
        name: selection_name(asset).to_owned(),
        thumb_src: asset_src(asset, AssetRendition::Explore),
        preview_src: asset_src(asset, AssetRendition::Preview),
        full_src: asset_src(asset, AssetRendition::Preview),
        domain: asset_domain_summary_dto(entry.domain),
        hearted,
        compare_count: asset.compare_count,
        win_count: asset.win_count,
        global_score: entry.quality.asset.mean,
        quality: quality_summary_dto(entry.quality),
    }
}

fn explore_point_dto(entry: &crate::model::ExploreEntry) -> ExplorePointDto {
    ExplorePointDto {
        asset_id: entry.asset.id.0.clone(),
        name: selection_name(&entry.asset).to_owned(),
        thumb_src: asset_src(&entry.asset, AssetRendition::Explore),
        plot_x: entry.plot[0],
        plot_y: entry.plot[1],
        latent: entry.latent,
        compare_count: entry.asset.compare_count,
        win_count: entry.asset.win_count,
    }
}

fn explore_selection_dto(
    selection: &ExploreSelection,
    hearted: &std::collections::HashSet<AssetId>,
) -> ExploreSelectionDto {
    ExploreSelectionDto {
        focus: focus_asset_dto(
            &selection.focus,
            hearted.contains(&selection.focus.asset.id),
        ),
        neighbors: selection
            .neighbors
            .iter()
            .map(|neighbor| ExploreNeighborDto {
                asset: focus_asset_dto(&neighbor.entry, hearted.contains(&neighbor.entry.asset.id)),
                distance: neighbor.distance,
            })
            .collect(),
    }
}
