use super::*;
use crate::app::{IdentityReviewCandidateView, IdentityReviewRowView};

pub(super) async fn identities_root(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<IdentitiesQuery>,
) -> WebResult<Response> {
    render_identities(state, None, query.anchor).await
}

pub(super) async fn identities_focus(
    State(state): State<SharedRuntimeState>,
    Path(subject_slug): Path<String>,
) -> WebResult<Response> {
    render_identities(state, Some(subject_slug), None).await
}

pub(super) async fn identities_set_threshold(
    State(state): State<SharedRuntimeState>,
    Form(form): Form<IdentityThresholdForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    state.set_identity_match_threshold_percent(form.percent)?;
    Ok(Redirect::to("/identities").into_response())
}

pub(super) async fn identities_name(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<IdentityNameForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let _changed = state.identity_review_rename(&form.anchor_handle, &form.name)?;
    let target = state
        .identity_anchor_focus_target(&form.anchor_handle)
        .or_else(|| referer_redirect_target(&headers))
        .unwrap_or_else(|| "/identities".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn identities_confirm(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<IdentityPairForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let changed = state.identity_review_confirm(&form.pair_handle)?;
    if changed {
        state.schedule_identity_review_refresh();
    }
    let target = state
        .identity_pair_focus_target(&form.pair_handle)
        .or_else(|| referer_redirect_target(&headers))
        .unwrap_or_else(|| "/identities".to_owned());
    Ok(Redirect::to(&target).into_response())
}

pub(super) async fn identities_veto(
    State(state): State<SharedRuntimeState>,
    headers: HeaderMap,
    Form(form): Form<IdentityPairForm>,
) -> WebResult<Response> {
    let Some(state) = state.ready_app() else {
        return Ok(service_unavailable_response());
    };
    let _ = state.identity_review_veto(&form.pair_handle)?;
    let target = state
        .identity_pair_focus_target(&form.pair_handle)
        .or_else(|| referer_redirect_target(&headers))
        .unwrap_or_else(|| "/identities".to_owned());
    Ok(Redirect::to(&target).into_response())
}

async fn render_identities(
    state: SharedRuntimeState,
    focus_slug: Option<String>,
    focus_anchor_handle: Option<String>,
) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/identities");
    let view = state.identity_review_view(focus_slug.as_deref(), focus_anchor_handle.as_deref())?;
    if (focus_slug.is_some() || focus_anchor_handle.is_some()) && view.rows.is_empty() {
        return Ok(Redirect::to("/identities").into_response());
    }
    Ok(render_markup(routed_layout(
        "identities-page",
        PageGeometry::Document,
        NavPage::Identities,
        Some(identities_mode_menu(view.status)),
        identities_markup(&view),
    )))
}

fn identities_markup(view: &IdentityReviewView) -> Markup {
    html! {
        main.identities-shell {
            @if let Some(focus_name) = &view.focus_name {
                section.identities-focus-banner.swarm-frame {
                    span.menu-chip.is-active { "focused" }
                    span.status-chip { (focus_name) }
                    a.swarm-frame-header href="/identities" { "all subjects" }
                }
            }
            @if !view.recognition_ready {
                section.empty-state {
                    p { "ArcFace recognition is disabled; identities review is unavailable." }
                }
            } @else if view.rows.is_empty() {
                section.empty-state {
                    p { "No active identity review rows yet." }
                }
            } @else {
                @for row in &view.rows {
                    (identity_row(row))
                }
            }
        }
    }
}

fn identity_row(row: &IdentityReviewRowView) -> Markup {
    let focus_href = row
        .name_slug
        .as_ref()
        .map(|slug| format!("/identities/{slug}"));
    html! {
        article.identity-row.swarm-frame {
            section.identity-anchor-column {
                div.identity-anchor-card.swarm-frame {
                    div.face-surface {
                        img
                            class="identity-face-image"
                            src=(row.anchor_face.face_src())
                            alt="subject representative face"
                            loading="lazy"
                            decoding="async"
                            draggable="false";
                    }
                    @if let Some(name) = &row.anchor_name {
                        div.frame-banner.swarm-frame-header { (name) }
                    } @else {
                        div.frame-banner.swarm-frame-header { "unnamed" }
                    }
                }
                div.identity-anchor-meta {
                    form.face-name-form.identity-name-form action="/identities/name" method="post" {
                        input type="hidden" name="anchor_handle" value=(row.anchor_handle);
                        input
                            type="text"
                            name="name"
                            value=(row.anchor_name.as_deref().unwrap_or(""))
                            placeholder="assign name"
                            autocomplete="off"
                            spellcheck="false";
                        button.tool.mini type="submit" { "set" }
                    }
                    div.identity-row-chips {
                        @if let Some(href) = focus_href {
                            a.menu-chip.is-active href=(href) { "focus" }
                        }
                        span.status-chip { "members " (row.member_count) }
                        span.status-chip { "candidates " (row.candidate_count) }
                        @if let Some(top_similarity) = row.top_similarity {
                            span.status-chip.is-model { "top " (format!("{top_similarity:.3}")) }
                        }
                    }
                }
            }
            section.identity-candidates-panel.swarm-frame {
                @if row.candidates.is_empty() {
                    div.identity-candidates-empty { "no unresolved candidates above the current threshold" }
                } @else {
                    div.identity-candidates-grid {
                        @for candidate in &row.candidates {
                            (identity_candidate_card(candidate))
                        }
                    }
                }
            }
        }
    }
}

fn identity_candidate_card(candidate: &IdentityReviewCandidateView) -> Markup {
    let confirm_title = "merge into this subject";
    html! {
        article.identity-candidate-card.swarm-frame {
            div.face-surface {
                img
                    class="identity-face-image"
                    src=(candidate.face.face_src())
                    alt="candidate face"
                    loading="lazy"
                    decoding="async"
                    draggable="false";
            }
            div.frame-tools {
                form action="/identities/confirm" method="post" {
                    input type="hidden" name="pair_handle" value=(candidate.pair_handle);
                    button.tool.mini type="submit" title=(confirm_title) { "✓" }
                }
                form action="/identities/veto" method="post" {
                    input type="hidden" name="pair_handle" value=(candidate.pair_handle);
                    button.tool.mini.danger type="submit" title="reject this candidate" { "x" }
                }
            }
            div.face-footer {
                span.meta-chip { "sim " (format!("{:.3}", candidate.similarity)) }
            }
        }
    }
}

pub(super) fn identities_mode_menu(status: IdentityReviewStatus) -> Markup {
    html! {
        details.rail-menu {
            summary.swarm-frame-header { "identities" }
            div.menu-panel.swarm-frame {
                div.menu-meta {
                    div { (format!("subjects {}", status.total_subjects)) }
                    div { (format!("named {}", status.named_subjects)) }
                    div { (format!("rows {}", status.active_rows)) }
                }
                div.menu-divider aria-hidden="true" {}
                form.menu-probability-form action="/identities/match-threshold" method="post" {
                    label.menu-probability-label for="identity-threshold" { "identity similarity threshold" }
                    div.menu-probability-row {
                        input
                            id="identity-threshold"
                            class="menu-probability-slider"
                            type="range"
                            name="percent"
                            min="0"
                            max="100"
                            step="1"
                            value=(status.threshold_percent)
                            oninput="this.nextElementSibling.value = this.value + '%'";
                        output.menu-probability-value { (format!("{}%", status.threshold_percent)) }
                        button type="submit" class="menu-probability-apply" { "set" }
                    }
                }
            }
        }
    }
}
