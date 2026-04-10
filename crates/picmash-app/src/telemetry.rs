use std::{fmt, sync::OnceLock, time::Duration};

use anyhow::anyhow;
use axum::{
    extract::MatchedPath,
    http::{HeaderMap, Request, Response},
};
use serde::{Deserialize, Serialize};
use tower_http::classify::ServerErrorsFailureClass;
use tracing::{Span, field, info_span, warn};
use tracing_error::ErrorLayer;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self as tracing_fmt, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};
use ulid::Ulid;

pub const PAGE_ID_HEADER: &str = "x-picmash-page-id";
pub const INTERACTION_ID_HEADER: &str = "x-picmash-interaction-id";
pub const ARENA_EPOCH_HEADER: &str = "x-picmash-arena-epoch";

const SLOW_HTTP_REQUEST_MS: u128 = 100;

static BOOT_ID: OnceLock<String> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BootId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClientPageId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InteractionId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientAnomalyKind {
    ArenaImageError,
    ArenaCurrentLayerEmpty,
    ArenaSalvageCurrentFromHidden,
    ArenaPromotionFailed,
    ArenaRefreshCurrentFailed,
    ArenaPrefetchPayloadRejected,
    ArenaInvariantBreach,
}

impl ClientAnomalyKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArenaImageError => "arena_image_error",
            Self::ArenaCurrentLayerEmpty => "arena_current_layer_empty",
            Self::ArenaSalvageCurrentFromHidden => "arena_salvage_current_from_hidden",
            Self::ArenaPromotionFailed => "arena_promotion_failed",
            Self::ArenaRefreshCurrentFailed => "arena_refresh_current_failed",
            Self::ArenaPrefetchPayloadRejected => "arena_prefetch_payload_rejected",
            Self::ArenaInvariantBreach => "arena_invariant_breach",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientAnomalyReport {
    pub kind: ClientAnomalyKind,
    #[serde(default)]
    pub page_id: Option<ClientPageId>,
    #[serde(default)]
    pub interaction_id: Option<InteractionId>,
    #[serde(default)]
    pub pair_href: Option<String>,
    #[serde(default)]
    pub current_href: Option<String>,
    #[serde(default)]
    pub lookahead_href: Option<String>,
    #[serde(default)]
    pub preserved_href: Option<String>,
    #[serde(default)]
    pub layer_role: Option<String>,
    #[serde(default)]
    pub image_src: Option<String>,
    #[serde(default)]
    pub left_handle: Option<String>,
    #[serde(default)]
    pub right_handle: Option<String>,
    #[serde(default)]
    pub asset_id: Option<String>,
    #[serde(default)]
    pub remote_item_id: Option<String>,
    #[serde(default)]
    pub visual_key: Option<String>,
    #[serde(default)]
    pub arena_epoch: Option<i64>,
}

pub fn install_boot_id(boot_id: BootId) -> anyhow::Result<()> {
    BOOT_ID
        .set(boot_id.0)
        .map_err(|_| anyhow!("boot id already installed"))
}

#[must_use]
pub fn fresh_boot_id() -> BootId {
    BootId(Ulid::new().to_string())
}

#[must_use]
pub fn fresh_writer_command_id() -> String {
    Ulid::new().to_string()
}

#[must_use]
pub fn boot_id() -> &'static str {
    BOOT_ID.get().map(String::as_str).unwrap_or("boot-unset")
}

pub fn init_subscriber() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let log_json = std::env::var("PICMASH_LOG_FORMAT")
        .ok()
        .is_some_and(|format| format.eq_ignore_ascii_case("json"));
    let show_source = std::env::var_os("PICMASH_LOG_SOURCE").is_some();

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(ErrorLayer::default());

    if log_json {
        registry
            .with(
                tracing_fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_events(FmtSpan::CLOSE)
                    .with_thread_names(true)
                    .with_target(false)
                    .with_file(show_source)
                    .with_line_number(show_source),
            )
            .init();
    } else {
        registry
            .with(
                tracing_fmt::layer()
                    .compact()
                    .with_span_events(FmtSpan::CLOSE)
                    .with_thread_names(true)
                    .with_target(false)
                    .with_file(show_source)
                    .with_line_number(show_source),
            )
            .init();
    }
}

#[must_use]
pub fn http_request_span<B>(request: &Request<B>) -> Span {
    let headers = request.headers();
    let matched_route = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str)
        .unwrap_or_else(|| request.uri().path());
    info_span!(
        "http.request",
        boot_id = %boot_id(),
        request_id = %request_id(headers).unwrap_or(""),
        method = %request.method(),
        matched_route = %matched_route,
        uri_path = %request.uri().path(),
        page_id = %header_text(headers, PAGE_ID_HEADER).unwrap_or(""),
        interaction_id = %header_text(headers, INTERACTION_ID_HEADER).unwrap_or(""),
        arena_epoch = %header_text(headers, ARENA_EPOCH_HEADER).unwrap_or(""),
        status = field::Empty,
        latency_ms = field::Empty,
    )
}

pub fn record_http_response<B>(response: &Response<B>, latency: Duration, span: &Span) {
    let elapsed_ms = latency.as_millis();
    span.record("status", field::display(response.status().as_u16()));
    span.record("latency_ms", field::display(elapsed_ms));
    if elapsed_ms > SLOW_HTTP_REQUEST_MS {
        let _entered = span.enter();
        warn!(
            elapsed_ms,
            status = response.status().as_u16(),
            "slow http request"
        );
    }
}

pub fn record_http_failure(failure: ServerErrorsFailureClass, latency: Duration, span: &Span) {
    let elapsed_ms = latency.as_millis();
    span.record("latency_ms", field::display(elapsed_ms));
    let _entered = span.enter();
    warn!(
        elapsed_ms,
        failure = %failure,
        "http request failed"
    );
}

pub fn log_client_anomaly(report: &ClientAnomalyReport) {
    warn!(
        kind = report.kind.as_str(),
        page_id = %report.page_id.as_ref().map_or("", |id| id.0.as_str()),
        interaction_id = %report.interaction_id.as_ref().map_or("", |id| id.0.as_str()),
        pair_href = %report.pair_href.as_deref().unwrap_or(""),
        current_href = %report.current_href.as_deref().unwrap_or(""),
        lookahead_href = %report.lookahead_href.as_deref().unwrap_or(""),
        preserved_href = %report.preserved_href.as_deref().unwrap_or(""),
        layer_role = %report.layer_role.as_deref().unwrap_or(""),
        image_src = %report.image_src.as_deref().unwrap_or(""),
        left_handle = %report.left_handle.as_deref().unwrap_or(""),
        right_handle = %report.right_handle.as_deref().unwrap_or(""),
        asset_id = %report.asset_id.as_deref().unwrap_or(""),
        remote_item_id = %report.remote_item_id.as_deref().unwrap_or(""),
        visual_key = %report.visual_key.as_deref().unwrap_or(""),
        arena_epoch = field::display(OptionalI64(report.arena_epoch)),
        "client anomaly"
    );
}

#[must_use]
pub fn request_id(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

#[must_use]
pub fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

#[derive(Debug, Clone, Copy)]
struct OptionalI64(Option<i64>);

impl fmt::Display for OptionalI64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(f, "{value}"),
            None => f.write_str(""),
        }
    }
}
