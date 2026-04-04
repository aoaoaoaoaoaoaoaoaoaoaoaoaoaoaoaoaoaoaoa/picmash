use super::*;

pub(super) async fn board(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/board");
    let board = state.board()?;
    Ok(render_markup(routed_layout(
        "board-page",
        PageGeometry::Document,
        NavPage::Board,
        None,
        html! {
            section.board-grid {
                @for (index, entry) in board.entries.iter().enumerate() {
                    (board_card(index + 1, entry))
                }
            }
            (board_preview_shell())
            (board_script_block())
        },
    )))
}

pub(super) async fn board_nudge(
    State(state): State<SharedRuntimeState>,
    Form(form): Form<NudgeForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(form.asset_id.clone()))?
        .is_none()
    {
        return Ok(Redirect::to("/board").into_response());
    }
    state.nudge_asset(&AssetId(form.asset_id), form.delta)?;
    if !state.quality_refresh_is_inline()? {
        state.schedule_quality_model_refresh();
    }
    Ok(Redirect::to("/board").into_response())
}

pub(super) async fn asset_heart(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<HeartAssetForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(form.asset_id.clone()))?
        .is_none()
    {
        return Ok(Redirect::to("/board").into_response());
    }
    state.set_heart_asset(&AssetId(form.asset_id), form.active)?;
    if !state.quality_refresh_is_inline()? {
        state.schedule_quality_model_refresh();
    }
    let target = referer_redirect_target(&headers).unwrap_or_else(|| "/board".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn asset_rotate(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<RotateForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(form.asset_id.clone()))?
        .is_none()
    {
        return Ok(Redirect::to("/board").into_response());
    }
    state.rotate_asset(&AssetId(form.asset_id), form.direction)?;
    let target = referer_redirect_target(&headers).unwrap_or_else(|| "/board".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn asset_hide(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<HideAssetForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(form.asset_id.clone()))?
        .is_none()
    {
        return Ok(Redirect::to("/board").into_response());
    }
    state.hide_asset_from_board(&AssetId(form.asset_id))?;
    let target = referer_redirect_target(&headers).unwrap_or_else(|| "/board".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn asset_domain(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<AssetDomainForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    if state
        .maybe_image_asset(&AssetId(form.asset_id.clone()))?
        .is_none()
    {
        return Ok(Redirect::to("/board").into_response());
    }
    state.set_asset_domain_label(&AssetId(form.asset_id), form.label)?;
    state.schedule_quality_model_refresh();
    let target = referer_redirect_target(&headers).unwrap_or_else(|| "/board".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn rescan(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let fallback_target = referer_redirect_target(&headers);
    let (summary, fallback) = tokio::task::spawn_blocking(move || {
        let summary = state.rescan()?;
        let fallback = state.home_target()?.href();
        anyhow::Ok((summary, fallback))
    })
    .await
    .map_err(|e| anyhow::anyhow!("rescan task panicked: {e}"))??;
    info!(
        corpus_id = summary.corpus_id.0,
        session_id = summary.session_id.0,
        visible_assets = summary.visible_assets,
        embedded_assets = summary.embedded_assets,
        "picmash rescan complete"
    );
    let target = fallback_target.unwrap_or(fallback);
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn set_external_probability(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<ExternalProbabilityForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    state.set_external_probability_percent(form.percent.min(100))?;
    let fallback = state.home_target()?.href();
    let target = referer_redirect_target(&headers).unwrap_or(fallback);
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn set_dedup_radius(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<DedupRadiusForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    state.set_dedup_radius(f32::from(form.percent) / 100.0)?;
    let fallback = state.home_target()?.href();
    let target = referer_redirect_target(&headers).unwrap_or(fallback);
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn set_arena_explore(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<ArenaExploreForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    state.set_arena_explore_percent(form.percent.min(100))?;
    let fallback = state.home_target()?.href();
    let target = referer_redirect_target(&headers).unwrap_or(fallback);
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn set_facemash_min_face_side(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<FacemashMinFaceSideForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    state.set_facemash_min_face_side(form.px)?;
    let fallback = state.home_target()?.href();
    let target = referer_redirect_target(&headers).unwrap_or(fallback);
    Ok(Redirect::to(&target).into_response())
}

fn board_card(rank: usize, entry: &BoardEntry) -> Markup {
    let tooltip = board_tooltip(rank, entry);
    let preview_name = entry
        .asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    html! {
        article
            class="board-card swarm-surface"
            title=(tooltip)
            tabindex="0"
            data-asset-id=(entry.asset.id.0)
            data-hearted=(if entry.hearted { "true" } else { "false" })
            data-preview-src=(asset_src(&entry.asset, AssetRendition::Preview))
            data-preview-name=(preview_name)
        {
            div.board-card-tools {
                form.nudge-form action="/board/nudge" method="post" {
                    input type="hidden" name="asset_id" value=(entry.asset.id.0);
                    input type="hidden" name="delta" value="-1";
                    (tool_button(ImageToolKind::Less, true, false))
                }
                form.nudge-form action="/board/nudge" method="post" {
                    input type="hidden" name="asset_id" value=(entry.asset.id.0);
                    input type="hidden" name="delta" value="1";
                    (tool_button(ImageToolKind::More, true, false))
                }
                form.board-heart-form action="/asset/heart" method="post" {
                    input type="hidden" name="asset_id" value=(entry.asset.id.0);
                    input type="hidden" name="active" value="true";
                    (tool_button(ImageToolKind::Heart, true, entry.hearted))
                }
                form.board-hide-form action="/asset/hide" method="post" {
                    input type="hidden" name="asset_id" value=(entry.asset.id.0);
                    (tool_button(ImageToolKind::Hide, true, false))
                }
            }
            img
                class="asset-image"
                src=(asset_src(&entry.asset, AssetRendition::Board))
                alt="ranked image"
                loading="lazy"
                decoding="async"
                onload="this.classList.add('is-ready')"
                onerror="this.classList.add('is-ready')"
                draggable="false";
            div.board-bottom-stack {
                div.board-quality-row {
                    (asset_quality_chips(entry.quality))
                }
            }
        }
    }
}

fn board_preview_shell() -> Markup {
    html! {
        section.board-preview hidden="" aria-hidden="true" data-asset-id="" {
            div.board-preview-tools.frame-tools {
                button.board-preview-action.tool type="button" data-preview-action="rotate-left" aria-label="rotate left" title="rotate left" { (ImageToolKind::RotateLeft.label()) }
                button.board-preview-action.tool type="button" data-preview-action="rotate-right" aria-label="rotate right" title="rotate right" { (ImageToolKind::RotateRight.label()) }
                button.board-preview-action.tool.heart type="button" data-preview-action="heart" aria-label="bless image" title="bless image" { (ImageToolKind::Heart.label()) }
                button.board-preview-action.tool.danger type="button" data-preview-action="hide" aria-label="hide image" title="hide image" { (ImageToolKind::Hide.label()) }
                button.board-preview-close.tool type="button" aria-label="close preview" title="close preview" { (ImageToolKind::Close.label()) }
            }
            div.board-preview-stage {
                img
                    class="board-preview-image"
                    alt="preview image"
                    loading="eager"
                    decoding="async"
                    draggable="false";
            }
            div.board-preview-caption.swarm-frame-header { "" }
        }
    }
}

fn board_script_block() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"
                const previewShell = document.querySelector(".board-preview");
                const previewImage = previewShell?.querySelector(".board-preview-image");
                const previewCaption = previewShell?.querySelector(".board-preview-caption");
                const closePreview = () => {
                  if (!previewShell || !previewImage || !previewCaption) return;
                  previewShell.hidden = true;
                  previewShell.setAttribute("aria-hidden", "true");
                  previewShell.dataset.assetId = "";
                  previewImage.removeAttribute("src");
                  previewCaption.textContent = "";
                  document.body.classList.remove("preview-open");
                };
                const openPreview = (card) => {
                  if (!previewShell || !previewImage || !previewCaption) return;
                  const src = card.dataset.previewSrc;
                  if (!src) return;
                  previewShell.dataset.assetId = card.dataset.assetId || "";
                  previewShell.dataset.hearted = card.dataset.hearted || "false";
                  previewImage.src = src;
                  previewCaption.textContent = card.dataset.previewName || "";
                  previewShell
                    .querySelector('[data-preview-action="heart"]')
                    ?.classList.toggle("active", card.dataset.hearted === "true");
                  previewShell.hidden = false;
                  previewShell.setAttribute("aria-hidden", "false");
                  document.body.classList.add("preview-open");
                };
                const submitBoardMutation = async (action, assetId, extra = {}) => {
                  const body = new URLSearchParams({ asset_id: assetId, ...extra });
                  const response = await fetch(action, {
                    method: "POST",
                    body,
                    credentials: "same-origin",
                    cache: "no-store",
                  });
                  if (!response.ok) throw new Error(`board mutation failed: ${response.status}`);
                };
                for (const card of document.querySelectorAll(".board-card")) {
                  card.addEventListener("click", (event) => {
                    if (event.target.closest("[data-board-aux]")) return;
                    openPreview(card);
                  });
                  card.addEventListener("keydown", (event) => {
                    if (!["Enter", " "].includes(event.key)) return;
                    if (event.target.closest("[data-board-aux]")) return;
                    event.preventDefault();
                    openPreview(card);
                  });
                }
                previewShell?.addEventListener("click", (event) => {
                  if (event.target === previewShell || event.target.closest(".board-preview-close")) {
                    closePreview();
                  }
                });
                previewImage?.addEventListener("click", (event) => {
                  event.stopPropagation();
                });
                for (const button of document.querySelectorAll(".board-preview-action")) {
                  button.addEventListener("click", async (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    const assetId = previewShell?.dataset.assetId;
                    if (!assetId) return;
                    const action = button.dataset.previewAction;
                    try {
                      if (action === "rotate-left") {
                        await submitBoardMutation("/asset/rotate", assetId, { direction: "-1" });
                      } else if (action === "rotate-right") {
                        await submitBoardMutation("/asset/rotate", assetId, { direction: "1" });
                      } else if (action === "heart") {
                        await submitBoardMutation("/asset/heart", assetId, { active: "true" });
                      } else if (action === "hide") {
                        await submitBoardMutation("/asset/hide", assetId);
                      }
                      window.location.reload();
                    } catch (error) {
                      console.error(error);
                      window.location.reload();
                    }
                  });
                }
                for (const tools of document.querySelectorAll("[data-board-aux], .board-card-tools")) {
                  for (const eventName of ["click", "pointerdown"]) {
                    tools.addEventListener(eventName, (event) => {
                      event.stopPropagation();
                    });
                  }
                }
                for (const form of document.querySelectorAll(".nudge-form, .board-hide-form, .board-heart-form")) {
                  form.addEventListener("submit", async (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    const body = new URLSearchParams(new FormData(form));
                    try {
                      const response = await fetch(form.action, {
                        method: "POST",
                        body,
                        credentials: "same-origin",
                        cache: "no-store",
                      });
                      if (!response.ok) throw new Error(`board nudge failed: ${response.status}`);
                      window.location.reload();
                    } catch (error) {
                      console.error(error);
                      window.location.reload();
                    }
                  });
                }
                document.addEventListener("keydown", (event) => {
                  if (event.key === "Escape" && previewShell && !previewShell.hidden) {
                    event.preventDefault();
                    closePreview();
                  }
                });
                "#,
            ))
        }
    }
}
