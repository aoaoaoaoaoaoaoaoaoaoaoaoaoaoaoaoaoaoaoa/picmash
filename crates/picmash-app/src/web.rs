use std::{
    path::{Path as FsPath, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use axum::{
    Form, Json, Router,
    extract::{Path, Query, State},
    http::{
        HeaderMap, StatusCode,
        header::{CACHE_CONTROL, CONTENT_TYPE, HeaderValue},
    },
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
};
use image::{
    ColorType, DynamicImage, ImageEncoder,
    codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder},
    imageops::{self, FilterType},
};
use maud::{Markup, PreEscaped, html};
use serde::Deserialize;
use tokio::fs;
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};

use crate::{
    api::{
        ExploreBootstrapDto, ExploreNeighborDto, ExplorePointDto, ExploreSelectionDto,
        ExploreSelectionResponseDto, FocusAssetDto, HeartAssetRequestDto, HideAssetRequestDto,
        RotateAssetRequestDto, TriadAssetDto, TriadBootstrapDto, TriadHandleDto,
        TriadTrainRequestDto,
    },
    app::{
        ArenaActionToken, ArenaCommand, ArenaCommandId, ArenaCommandOutcome, ArenaCommandStatus,
        ArenaPairRef, ArenaRevision, ArenaSamplerEpoch, ArenaTurn, ArenaTurnId, FacemashFaceView,
        FacemashLocalAssetView, FacemashPairView, FacemashStatus, IdentityReviewStatus,
        IdentityReviewView,
    },
    app::{RedirectTarget, RuntimePhase, RuntimeSnapshot, SharedAppState, SharedRuntimeState},
    asset_domain::AssetDomainLabel,
    model::{
        ArenaCard, ArenaHandle, ArenaView, AssetDomainView, AssetId, AssetQualitySummary,
        AssetRecord, BoardEntry, DuplicateCluster, ExploreMapMode, ExploreSelection, ExploreView,
        ExternalArenaStatus, FaceId, PosteriorSummary, RemoteItemId, RemoteItemRecord,
    },
    store::FaceRecord,
    telemetry,
};

mod arena;
mod board;
mod explore;
mod facemash;
mod forms;
mod identities;
mod media;
mod runtime;
mod shell;
mod view;
mod vocab;

use self::{
    arena::{
        api_arena_next, arena, arena_reroll, arena_root, heart, hide, lock_thread, rotate,
        veto_thread, vote, vote_get,
    },
    board::{
        asset_domain, asset_heart, asset_hide, asset_rotate, board, board_nudge, rescan,
        set_arena_explore, set_dedup_radius, set_external_probability, set_facemash_min_face_side,
    },
    explore::{
        api_explore_bootstrap, api_explore_domain, api_explore_heart, api_explore_hide,
        api_explore_rotate, api_explore_selection, api_triad_bootstrap, api_triad_domain,
        api_triad_heart, api_triad_hide, api_triad_rotate, api_triad_train, explore_root,
        triad_reroll, triad_root,
    },
    facemash::{
        face_image, facemash_hide_face, facemash_pair, facemash_reroll, facemash_root,
        facemash_status, facemash_vote,
    },
    identities::{
        identities_confirm, identities_focus, identities_name, identities_root,
        identities_set_threshold, identities_veto,
    },
    media::{AssetRendition, asset_image, favicon_asset, font_asset, remote_image},
    vocab::{vocab_lab_reroll, vocab_lab_root},
};
use self::{forms::*, runtime::*, shell::*, view::*};

pub fn router(state: SharedRuntimeState) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/favicon.svg", get(favicon_asset))
        .route("/__status", get(runtime_status))
        .route("/api/client-event", post(client_event))
        .route("/__frontend/picmash-client.js", get(frontend_javascript))
        .route("/__frontend/api.js", get(frontend_api_javascript))
        .route("/__frontend/explore.js", get(frontend_explore_javascript))
        .route(
            "/__frontend/media-frame.js",
            get(frontend_media_frame_javascript),
        )
        .route("/__frontend/triad.js", get(frontend_triad_javascript))
        .route("/__frontend/picmash-client.css", get(frontend_stylesheet))
        .route("/arena", get(arena_root))
        .route("/arena/reroll", get(arena_reroll))
        .route("/arena/{left_id}/{right_id}", get(arena))
        .route("/arena/{left_id}/{right_id}/vote", post(vote).get(vote_get))
        .route("/api/arena/next", get(api_arena_next))
        .route("/arena/{left_id}/{right_id}/rotate", post(rotate))
        .route("/arena/{left_id}/{right_id}/heart", post(heart))
        .route("/arena/{left_id}/{right_id}/hide", post(hide))
        .route("/arena/{left_id}/{right_id}/lock-thread", post(lock_thread))
        .route("/arena/{left_id}/{right_id}/veto-thread", post(veto_thread))
        .route("/board", get(board))
        .route("/board/nudge", post(board_nudge))
        .route("/asset/heart", post(asset_heart))
        .route("/asset/rotate", post(asset_rotate))
        .route("/asset/hide", post(asset_hide))
        .route("/asset/domain", post(asset_domain))
        .route("/rescan", post(rescan))
        .route("/external/probability", post(set_external_probability))
        .route("/arena/explore", post(set_arena_explore))
        .route("/external/dedup-radius", post(set_dedup_radius))
        .route("/explore", get(explore_root))
        .route("/triad", get(triad_root))
        .route("/triad/reroll", get(triad_reroll))
        .route("/vocab-lab", get(vocab_lab_root))
        .route("/vocab-lab/reroll", get(vocab_lab_reroll))
        .route("/api/explore/bootstrap", get(api_explore_bootstrap))
        .route(
            "/api/explore/selection/{asset_id}",
            get(api_explore_selection),
        )
        .route("/api/explore/rotate", post(api_explore_rotate))
        .route("/api/explore/heart", post(api_explore_heart))
        .route("/api/explore/domain", post(api_explore_domain))
        .route("/api/explore/hide", post(api_explore_hide))
        .route("/api/triad/bootstrap", get(api_triad_bootstrap))
        .route("/api/triad/train", post(api_triad_train))
        .route("/api/triad/rotate", post(api_triad_rotate))
        .route("/api/triad/heart", post(api_triad_heart))
        .route("/api/triad/domain", post(api_triad_domain))
        .route("/api/triad/hide", post(api_triad_hide))
        .route("/assets/{asset_id}", get(asset_image))
        .route("/remote/{item_id}", get(remote_image))
        .route("/faces/{face_id}", get(face_image))
        .route("/facemash", get(facemash_root))
        .route("/facemash/reroll", get(facemash_reroll))
        .route(
            "/facemash/{left_face_id}/{right_face_id}",
            get(facemash_pair),
        )
        .route("/facemash/min-face-side", post(set_facemash_min_face_side))
        .route("/facemash/vote", post(facemash_vote))
        .route("/facemash/hide", post(facemash_hide_face))
        .route("/facemash/status", get(facemash_status))
        .route("/identities", get(identities_root))
        .route("/identities/{subject_slug}", get(identities_focus))
        .route(
            "/identities/match-threshold",
            post(identities_set_threshold),
        )
        .route("/identities/name", post(identities_name))
        .route("/identities/confirm", post(identities_confirm))
        .route("/identities/veto", post(identities_veto))
        .route("/__swarm/fonts/{font_name}", get(font_asset))
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(telemetry::http_request_span)
                .on_response(telemetry::record_http_response)
                .on_failure(telemetry::record_http_failure),
        )
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
}

static SITE_LOAD_LOGGED: AtomicBool = AtomicBool::new(false);
const FAVICON_SVG: &str = include_str!("../assets/favicon.svg");
const FRONTEND_JS: &str = include_str!("../assets/web/main.js");
const FRONTEND_API_JS: &str = include_str!("../assets/web/api.js");
const FRONTEND_EXPLORE_JS: &str = include_str!("../assets/web/explore.js");
const FRONTEND_MEDIA_FRAME_JS: &str = include_str!("../assets/web/media-frame.js");
const FRONTEND_TRIAD_JS: &str = include_str!("../assets/web/triad.js");
const FRONTEND_CSS: &str = concat!(
    include_str!("../assets/picmash.css"),
    "\n",
    include_str!("../assets/web/picmash-client.css"),
);
const RENDITION_CACHE_VERSION: u32 = 5;
const EXPLORE_EMPTY_NOTE: &str =
    "Explore needs embedded images. Let DINO finish, then reload or rescan.";
const TRIAD_EMPTY_NOTE: &str = "Need at least three embedded images to train the similarity field.";
