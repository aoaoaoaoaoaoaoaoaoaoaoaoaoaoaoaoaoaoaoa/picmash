use super::*;
use tracing::{error, info_span};

pub(super) fn log_site_loaded(route: &'static str) {
    if SITE_LOAD_LOGGED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        info!(route, "site loaded and ready");
    }
}

pub(super) async fn runtime_status(State(state): State<SharedRuntimeState>) -> Response {
    no_store_response(Json(state.snapshot()).into_response())
}

pub(super) async fn client_event(
    Json(report): Json<crate::telemetry::ClientAnomalyReport>,
) -> WebResult<Response> {
    let _span = info_span!(
        "client.anomaly",
        kind = report.kind.as_str(),
        page_id = %report.page_id.as_ref().map_or("", |id| id.0.as_str()),
        interaction_id = %report.interaction_id.as_ref().map_or("", |id| id.0.as_str()),
    )
    .entered();
    crate::telemetry::log_client_anomaly(&report);
    Ok(no_store_response(
        Json(serde_json::json!({ "ok": true })).into_response(),
    ))
}

pub(super) async fn frontend_javascript() -> Response {
    frontend_module_response(FRONTEND_JS)
}

pub(super) async fn frontend_api_javascript() -> Response {
    frontend_module_response(FRONTEND_API_JS)
}

pub(super) async fn frontend_explore_javascript() -> Response {
    frontend_module_response(FRONTEND_EXPLORE_JS)
}

pub(super) async fn frontend_media_frame_javascript() -> Response {
    frontend_module_response(FRONTEND_MEDIA_FRAME_JS)
}

pub(super) async fn frontend_triad_javascript() -> Response {
    frontend_module_response(FRONTEND_TRIAD_JS)
}

fn frontend_module_response(source: &'static str) -> Response {
    cached_text_response(
        source,
        HeaderValue::from_static("text/javascript; charset=utf-8"),
    )
}

pub(super) async fn frontend_stylesheet() -> Response {
    cached_text_response(
        FRONTEND_CSS,
        HeaderValue::from_static("text/css; charset=utf-8"),
    )
}

pub(super) fn ready_app_or_snapshot(
    state: &SharedRuntimeState,
) -> Result<SharedAppState, RuntimeSnapshot> {
    state.ready_app().ok_or_else(|| state.snapshot())
}

pub(super) fn boot_response(snapshot: RuntimeSnapshot) -> Response {
    let status = match snapshot.phase {
        RuntimePhase::Loading | RuntimePhase::Ready => StatusCode::OK,
        RuntimePhase::Failed => StatusCode::SERVICE_UNAVAILABLE,
    };
    no_store_response(
        (
            status,
            Html(layout("loading-page", boot_markup(&snapshot)).into_string()),
        )
            .into_response(),
    )
}

fn boot_markup(snapshot: &RuntimeSnapshot) -> Markup {
    let status_label = match snapshot.phase {
        RuntimePhase::Loading => "warming the machine",
        RuntimePhase::Ready => "ready",
        RuntimePhase::Failed => "boot fault",
    };
    let detail = match snapshot.phase {
        RuntimePhase::Loading => {
            "ingesting the corpus, extracting embeddings, and forging the live state."
        }
        RuntimePhase::Ready => "the app is ready; this page should reload immediately.",
        RuntimePhase::Failed => snapshot.message.as_deref().unwrap_or("picmash boot failed"),
    };
    html! {
        section.loading-shell {
            div.loading-panel.swarm-surface-elevated {
                div.loading-kicker { "picmash" }
                h1.loading-title { (status_label) }
                p.loading-detail data-loading-message="" { (detail) }
                div.loading-progress aria-hidden="true" {
                    span.loading-bar {}
                }
            }
        }
    }
}

pub(super) fn service_unavailable_response() -> Response {
    no_store_response((StatusCode::SERVICE_UNAVAILABLE, "picmash is still booting").into_response())
}

pub(super) async fn home(State(state): State<SharedRuntimeState>) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/");
    let href = state.home_target()?.href();
    Ok(Redirect::to(&href).into_response())
}

pub(super) fn stale_arena_action_response<const N: usize>(
    state: &SharedAppState,
    handles: [&ArenaHandle; N],
) -> anyhow::Result<Option<Response>> {
    for handle in handles {
        if !state.arena_handle_is_live(handle)? {
            return Ok(Some(
                Redirect::to(&state.arena_target()?.href()).into_response(),
            ));
        }
    }
    Ok(None)
}

pub(super) fn stale_facemash_action_response(
    state: &SharedAppState,
    left_face_id: FaceId,
    right_face_id: FaceId,
) -> anyhow::Result<Option<Response>> {
    if !state.facemash_pair_is_live(left_face_id, right_face_id)? {
        return Ok(Some(Redirect::to("/facemash").into_response()));
    }
    Ok(None)
}

pub(super) type WebResult<T> = Result<T, AppError>;

pub(super) struct AppError(anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!(error = %format!("{:#}", self.0), "web request failed");
        let markup = layout(
            "empty-page",
            html! {
                section.empty-state {
                    div.error-shell {
                        p { (self.0.to_string()) }
                        a href="/" { "return home" }
                    }
                }
            },
        );
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(markup.into_string()),
        )
            .into_response()
    }
}

pub(super) fn no_store_response(mut response: Response) -> Response {
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response
}

pub(super) fn render_markup(markup: Markup) -> Response {
    no_store_response(Html(markup.into_string()).into_response())
}

pub(super) fn cached_text_response(body: &str, mime: HeaderValue) -> Response {
    let mut response = Response::new(body.to_owned().into());
    response.headers_mut().insert(CONTENT_TYPE, mime);
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}

pub(super) fn cached_image_response(body: Vec<u8>, mime: HeaderValue) -> Response {
    let mut response = Response::new(body.into());
    response.headers_mut().insert(CONTENT_TYPE, mime);
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}
