use super::media::{
    FaceImageQuery, has_rendition_failure_marker_for_key, rendition_cache_path_for_key,
    rendition_failure_marker_path_for_key, write_rendition_cache, write_rendition_failure_marker,
};
use super::*;

pub(super) async fn facemash_root(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/facemash");
    let target = state.facemash_target()?;
    if !matches!(target, RedirectTarget::FacemashRoot) {
        return Ok(Redirect::to(&target.href()).into_response());
    }
    let status = state.facemash_status()?;
    Ok(render_markup(routed_layout(
        "facemash-page",
        PageGeometry::Viewport,
        NavPage::Facemash,
        Some(facemash_mode_menu(status)),
        facemash_markup(None),
    )))
}

pub(super) async fn facemash_reroll() -> WebResult<Response> {
    Ok(Redirect::to("/facemash").into_response())
}

pub(super) async fn facemash_pair(
    State(state): State<SharedRuntimeState>,
    Path((left_face_id, right_face_id)): Path<(i64, i64)>,
) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/facemash");
    let pair = state.facemash_pair_view_by_ids(FaceId(left_face_id), FaceId(right_face_id))?;
    if pair.is_none() {
        return Ok(Redirect::to("/facemash").into_response());
    }
    let status = state.facemash_status()?;
    Ok(render_markup(routed_layout(
        "facemash-page",
        PageGeometry::Viewport,
        NavPage::Facemash,
        Some(facemash_mode_menu(status)),
        facemash_markup(pair),
    )))
}

pub(super) async fn facemash_vote(
    State(state): State<SharedRuntimeState>,
    Form(form): Form<FacemashVoteForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let winner = FaceId(form.winner_face_id.parse().map_err(anyhow::Error::msg)?);
    let loser = FaceId(form.loser_face_id.parse().map_err(anyhow::Error::msg)?);
    if winner == loser {
        return Ok(Redirect::to("/facemash").into_response());
    }
    if let Some(response) = stale_facemash_action_response(&state, winner, loser)? {
        return Ok(response);
    }
    if !state.facemash_vote(winner, loser)? {
        return Ok(Redirect::to("/facemash").into_response());
    }
    state.schedule_quality_model_refresh();
    Ok(Redirect::to(&state.facemash_target()?.href()).into_response())
}

pub(super) async fn facemash_hide_face(
    State(state): State<SharedRuntimeState>,
    Form(form): Form<FacemashHideForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let face_id = FaceId(form.face_id.parse().map_err(anyhow::Error::msg)?);
    if state.facemash_hide_face(face_id)? {
        state.schedule_quality_model_refresh();
    }
    Ok(Redirect::to("/facemash").into_response())
}

pub(super) async fn facemash_status(
    State(state): State<SharedRuntimeState>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    Ok(no_store_response(
        Json(state.facemash_status()?).into_response(),
    ))
}

pub(super) async fn face_image(
    State(state): State<SharedRuntimeState>,
    Path(face_id): Path<i64>,
    Query(query): Query<FaceImageQuery>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let Some(face) = state.face_record(FaceId(face_id))? else {
        return Ok((StatusCode::NOT_FOUND, "unknown face").into_response());
    };
    if query.debug {
        let Some(bytes) = render_debug_face_overlay(&state, &face)? else {
            return Ok((StatusCode::NOT_FOUND, "missing face source").into_response());
        };
        return Ok(cached_image_response(
            bytes,
            HeaderValue::from_static("image/png"),
        ));
    }
    if query.full {
        if let Some(response) = render_full_face_preview(&state, &face).await? {
            return Ok(response);
        }
        let Some(bytes) = state.face_crop_bytes(FaceId(face_id))? else {
            return Ok((StatusCode::NOT_FOUND, "missing face source").into_response());
        };
        return Ok(cached_image_response(
            bytes,
            HeaderValue::from_static("image/png"),
        ));
    }
    let Some(bytes) = state.face_crop_bytes(FaceId(face_id))? else {
        return Ok((StatusCode::NOT_FOUND, "missing face crop").into_response());
    };
    Ok(cached_image_response(
        bytes,
        HeaderValue::from_static("image/png"),
    ))
}

fn facemash_markup(pair: Option<FacemashPairView>) -> Markup {
    html! {
        main.facemash-shell {
            @if let Some(pair) = pair {
                @let left = &pair.left;
                @let right = &pair.right;
                section.facemash-stage {
                    (facemash_panel(left, right, "1"))
                    (facemash_panel(right, left, "2"))
                }
                div.facemash-choice-rail.swarm-frame {
                    a.facemash-skip href="/facemash" { "skip" }
                    span.facemash-hint { "1/2 choose winner" }
                }
                (facemash_script_block())
            } @else {
                section.empty-state {
                    p { "Need at least two embedded local identities." }
                }
            }
        }
    }
}

fn facemash_panel(face: &FacemashFaceView, opponent: &FacemashFaceView, shortcut: &str) -> Markup {
    let face_href = format!("/faces/{}?full=true", face.face.id.0);
    let face_name = face.face.identity.name.as_deref().unwrap_or("");
    html! {
        div.facemash-panel-shell {
            (local_asset_frame_tools(&face.asset))
            form.facemash-form action="/facemash/vote" method="post" {
                input type="hidden" name="winner_face_id" value=(face.face.id.0);
                input type="hidden" name="loser_face_id" value=(opponent.face.id.0);
                button.facemash-panel.swarm-surface-elevated type="submit" {
                    div.facemash-kicker {
                        (shortcut) " · #" (face.face.id.0)
                        @if !face_name.is_empty() {
                            " · " (face_name)
                        }
                    }
                    div.vote-surface.facemash-vote-surface {
                        img
                            class="facemash-image arena-image"
                            data-rotation=(face.asset.rotation_quarters)
                            src=(face_href)
                            alt="source image with face frame"
                            loading="eager"
                            decoding="async"
                            draggable="false";
                    }
                    div.facemash-meta {
                        span.facemash-meta-chip.facemash-meta-chip-beauty {
                            "beauty "
                            (format!(
                                "{:.0} ± {:.0}",
                                face.face.identity.beauty.mean,
                                face.face.identity.beauty.sigma
                            ))
                        }
                        @if let Some(predicted) = face.predicted {
                            span.facemash-meta-chip.facemash-meta-chip-model title="model mean ± epistemic sigma" {
                                "model "
                                (format!("{:.0} ± {:.0}", predicted.mean, predicted.sigma))
                            }
                        }
                        span.facemash-meta-chip { "duels " (face.face.identity.duel_count) }
                    }
                }
            }
            form.facemash-face-hide-form data-facemash-aux="" action="/facemash/hide" method="post" {
                input type="hidden" name="face_id" value=(face.face.id.0);
                button.facemash-face-action.tool.danger.facemash-face-hide data-facemash-aux="" type="submit" title="exclude face from facemash" { "exclude" }
            }
        }
    }
}

fn local_asset_frame_tools(asset: &FacemashLocalAssetView) -> Markup {
    let asset_id = &asset.id.0;
    html! {
        div.frame-tools.facemash-asset-tools data-facemash-aux="" {
            form data-facemash-aux="" action="/asset/rotate" method="post" {
                input type="hidden" name="asset_id" value=(asset_id);
                input type="hidden" name="direction" value="-1";
                (tool_button(ImageToolKind::RotateLeft, false, false))
            }
            form data-facemash-aux="" action="/asset/rotate" method="post" {
                input type="hidden" name="asset_id" value=(asset_id);
                input type="hidden" name="direction" value="1";
                (tool_button(ImageToolKind::RotateRight, false, false))
            }
            form data-facemash-aux="" action="/asset/heart" method="post" {
                input type="hidden" name="asset_id" value=(asset_id);
                input type="hidden" name="active" value="true";
                (tool_button(ImageToolKind::Heart, false, asset.hearted))
            }
            form data-facemash-aux="" action="/asset/hide" method="post" {
                input type="hidden" name="asset_id" value=(asset_id);
                (tool_button(ImageToolKind::Hide, false, false))
            }
            (asset_domain_controls(&asset.id, asset.domain, false, true))
        }
    }
}

fn facemash_script_block() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"
                (() => {
                  const forms = Array.from(document.querySelectorAll(".facemash-form"));
                  const hideForms = Array.from(document.querySelectorAll(".facemash-face-hide-form"));
                  const shieldedControls = Array.from(document.querySelectorAll("[data-facemash-aux]"));
                  const images = Array.from(document.querySelectorAll(".facemash-vote-surface .arena-image"));
                  const normalizeTurns = (value) => ((Number(value) || 0) % 4 + 4) % 4;
                  const imageGeometry = (image) => {
                    if (!image.naturalWidth || !image.naturalHeight) return null;
                    const turns = normalizeTurns(image.dataset.rotation);
                    const rotated = turns % 2 !== 0;
                    const naturalWidth = image.naturalWidth;
                    const naturalHeight = image.naturalHeight;
                    return {
                      turns,
                      naturalWidth,
                      naturalHeight,
                      effectiveWidth: rotated ? naturalHeight : naturalWidth,
                      effectiveHeight: rotated ? naturalWidth : naturalHeight,
                    };
                  };
                  const fitImage = (image) => {
                    const surface = image.closest(".facemash-vote-surface");
                    const geometry = imageGeometry(image);
                    if (!surface || !geometry) return;
                    const boundWidth = surface.clientWidth;
                    const boundHeight = surface.clientHeight;
                    if (!boundWidth || !boundHeight) return;
                    const scale = Math.min(
                      boundWidth / geometry.effectiveWidth,
                      boundHeight / geometry.effectiveHeight,
                    );
                    image.style.width = `${geometry.naturalWidth * scale}px`;
                    image.style.height = `${geometry.naturalHeight * scale}px`;
                    image.style.transform = `translate(-50%, -50%) rotate(${geometry.turns * 90}deg)`;
                    image.classList.add("is-ready");
                  };
                  const seal = (event) => {
                    event.stopPropagation();
                  };
                  for (const node of shieldedControls) {
                    for (const type of ["pointerdown", "pointerup", "mousedown", "mouseup", "click"]) {
                      node.addEventListener(type, seal, true);
                    }
                  }
                  for (const form of document.querySelectorAll(".facemash-face-hide-form")) {
                    form.addEventListener("submit", seal, true);
                  }
                  for (const image of images) {
                    if (image.complete && image.naturalWidth > 0) {
                      fitImage(image);
                      continue;
                    }
                    image.addEventListener("load", () => fitImage(image), { once: true });
                    image.addEventListener("error", () => fitImage(image), { once: true });
                  }
                  window.addEventListener("resize", () => {
                    for (const image of images) fitImage(image);
                  });
                  document.addEventListener("keydown", (event) => {
                    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
                    if (event.key === "1") {
                      event.preventDefault();
                      forms[0]?.requestSubmit();
                    }
                    if (event.key === "2") {
                      event.preventDefault();
                      forms[1]?.requestSubmit();
                    }
                    if (event.altKey && event.key === "1") {
                      event.preventDefault();
                      hideForms[0]?.requestSubmit();
                    }
                    if (event.altKey && event.key === "2") {
                      event.preventDefault();
                      hideForms[1]?.requestSubmit();
                    }
                    if (event.key === " ") {
                      event.preventDefault();
                      window.location.assign("/facemash");
                    }
                  });
                })();
                "#,
            ))
        }
    }
}

fn render_debug_face_overlay(
    state: &SharedAppState,
    face: &FaceRecord,
) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(source_path) = state.face_source_path(face.id)? else {
        return Ok(None);
    };
    let source_bytes = std::fs::read(&source_path)
        .with_context(|| format!("reading face source {}", source_path.display()))?;
    let source = crate::identity::canonical_embedding_image(&source_bytes)
        .with_context(|| format!("canonicalizing face source {}", source_path.display()))?
        .to_rgba8();
    let pad_x = (face.bbox_w * 0.7).max(16.0);
    let pad_y = (face.bbox_h * 0.7).max(16.0);
    let crop_left = (face.bbox_x - pad_x).floor().max(0.0) as u32;
    let crop_top = (face.bbox_y - pad_y).floor().max(0.0) as u32;
    let crop_right = (face.bbox_x + face.bbox_w + pad_x)
        .ceil()
        .min(source.width() as f32) as u32;
    let crop_bottom = (face.bbox_y + face.bbox_h + pad_y)
        .ceil()
        .min(source.height() as f32) as u32;
    let crop_width = crop_right.saturating_sub(crop_left).max(1);
    let crop_height = crop_bottom.saturating_sub(crop_top).max(1);
    let mut image =
        imageops::crop_imm(&source, crop_left, crop_top, crop_width, crop_height).to_image();
    draw_bbox_overlay(
        &mut image,
        face.bbox_x - crop_left as f32,
        face.bbox_y - crop_top as f32,
        face.bbox_w,
        face.bbox_h,
        [92, 224, 255, 255],
    );
    let palette = [
        [255, 96, 96, 255],
        [96, 196, 255, 255],
        [255, 224, 96, 255],
        [144, 255, 144, 255],
        [232, 128, 255, 255],
    ];
    for (index, (x, y)) in face.landmarks.0.into_iter().enumerate() {
        draw_landmark_cross(
            &mut image,
            x - crop_left as f32,
            y - crop_top as f32,
            palette[index],
        );
    }
    let mut encoded = Vec::new();
    PngEncoder::new_with_quality(&mut encoded, CompressionType::Best, PngFilterType::Adaptive)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgba8.into(),
        )
        .context("encoding face debug overlay")?;
    Ok(Some(encoded))
}

async fn render_full_face_preview(
    state: &SharedAppState,
    face: &FaceRecord,
) -> anyhow::Result<Option<Response>> {
    let Some(source_path) = state.face_source_path(face.id)? else {
        return Ok(None);
    };
    let cache_key = format!("face-full-{}", face.id.0);
    let cache_path =
        rendition_cache_path_for_key(state.cache_root(), &cache_key, 0, AssetRendition::Face);
    let failure_marker_path = rendition_failure_marker_path_for_key(
        state.cache_root(),
        &cache_key,
        0,
        AssetRendition::Face,
    );
    if let Ok(body) = fs::read(&cache_path).await {
        return Ok(Some(cached_image_response(
            body,
            HeaderValue::from_static("image/png"),
        )));
    }

    let bytes = match fs::read(&source_path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(anyhow::Error::from(error)
                .context(format!("reading face source {}", source_path.display())));
        }
    };

    let payload = if has_rendition_failure_marker_for_key(
        state.cache_root(),
        &cache_key,
        0,
        AssetRendition::Face,
    )
    .await
    {
        normalize_face_preview_payload(bytes, face.clone())?
    } else {
        match tokio::task::spawn_blocking({
            let face = face.clone();
            move || normalize_face_preview_payload(bytes, face)
        })
        .await
        .context("joining face preview render task")?
        {
            Ok(bytes) => {
                if let Err(error) = write_rendition_cache(&cache_path, &bytes).await {
                    warn!(
                        "failed to persist face preview cache {}: {error:#}",
                        cache_path.display()
                    );
                }
                bytes
            }
            Err(error) => {
                warn!(
                    face_id = face.id.0,
                    "failed to render face preview {}: {error:#}",
                    source_path.display()
                );
                if let Err(marker_error) =
                    write_rendition_failure_marker(&failure_marker_path, &error.to_string()).await
                {
                    warn!(
                        "failed to persist face preview failure marker {}: {marker_error:#}",
                        failure_marker_path.display()
                    );
                }
                return Ok(None);
            }
        }
    };

    Ok(Some(cached_image_response(
        payload,
        HeaderValue::from_static("image/png"),
    )))
}

fn normalize_face_preview_payload(bytes: Vec<u8>, face: FaceRecord) -> anyhow::Result<Vec<u8>> {
    let image = crate::identity::decode_image(&bytes)?;
    let original_width = image.width().max(1);
    let original_height = image.height().max(1);
    let bounded = if original_width.max(original_height) > AssetRendition::Face.max_edge() {
        image.resize(
            AssetRendition::Face.max_edge(),
            AssetRendition::Face.max_edge(),
            FilterType::CatmullRom,
        )
    } else {
        image
    };
    let scale_x = bounded.width() as f32 / original_width as f32;
    let scale_y = bounded.height() as f32 / original_height as f32;
    let mut rgba = bounded.into_rgba8();
    draw_bbox_overlay(
        &mut rgba,
        face.bbox_x * scale_x,
        face.bbox_y * scale_y,
        face.bbox_w * scale_x,
        face.bbox_h * scale_y,
        [214, 245, 255, 255],
    );
    let mut encoded = Vec::new();
    PngEncoder::new_with_quality(&mut encoded, CompressionType::Fast, PngFilterType::Adaptive)
        .write_image(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            ColorType::Rgba8.into(),
        )
        .context("encoding face preview overlay")?;
    Ok(encoded)
}

fn draw_landmark_cross(image: &mut image::RgbaImage, x: f32, y: f32, color: [u8; 4]) {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let radius = 3i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let manhattan = dx.abs() + dy.abs();
            if manhattan > radius || (dx.abs() == 1 && dy.abs() == 1) {
                continue;
            }
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || py < 0 {
                continue;
            }
            let (px, py) = (px as u32, py as u32);
            if px >= image.width() || py >= image.height() {
                continue;
            }
            image.put_pixel(px, py, image::Rgba(color));
        }
    }
}

fn draw_bbox_overlay(image: &mut image::RgbaImage, x: f32, y: f32, w: f32, h: f32, color: [u8; 4]) {
    let left = x.floor().max(0.0) as i32;
    let top = y.floor().max(0.0) as i32;
    let right = (x + w).ceil().min(image.width() as f32 - 1.0) as i32;
    let bottom = (y + h).ceil().min(image.height() as f32 - 1.0) as i32;
    for px in left..=right {
        paint_pixel(image, px, top, color);
        paint_pixel(image, px, bottom, color);
    }
    for py in top..=bottom {
        paint_pixel(image, left, py, color);
        paint_pixel(image, right, py, color);
    }
}

fn paint_pixel(image: &mut image::RgbaImage, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    if x >= image.width() || y >= image.height() {
        return;
    }
    image.put_pixel(x, y, image::Rgba(color));
}

pub(super) fn facemash_mode_menu(status: FacemashStatus) -> Markup {
    html! {
        details.rail-menu {
            summary.swarm-frame-header { "facemash" }
            div.menu-panel.swarm-frame {
                div.menu-meta {
                    div { (format!("identities {}", status.total_identities)) }
                    div { (format!("duels {}", status.comparisons)) }
                    div { (format!("oracle {}", if status.oracle_trained { "trained" } else { "cold" })) }
                }
                div.menu-divider aria-hidden="true" {}
                form.menu-probability-form action="/facemash/min-face-side" method="post" {
                    label.menu-probability-label for="facemash-min-face-side" { "min face side" }
                    div.menu-probability-row {
                        input
                            id="facemash-min-face-side"
                            class="menu-probability-slider"
                            type="range"
                            name="px"
                            min="24"
                            max="256"
                            step="8"
                            value=(status.min_face_side)
                            oninput="this.nextElementSibling.value = this.value + 'px'";
                        output.menu-probability-value { (format!("{}px", status.min_face_side)) }
                        button type="submit" class="menu-probability-apply" { "set" }
                    }
                }
            }
        }
    }
}
