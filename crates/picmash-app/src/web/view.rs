use super::*;

pub(super) fn asset_src(asset: &AssetRecord, rendition: AssetRendition) -> String {
    format!(
        "/assets/{}?r={}&kind={}",
        asset.id.0,
        asset.rotation_quarters,
        rendition.as_str()
    )
}

pub(super) fn remote_src(item: &RemoteItemRecord, rendition: AssetRendition) -> String {
    format!(
        "/remote/{}?r={}&kind={}",
        item.id.0,
        item.rotation_quarters,
        rendition.as_str()
    )
}

pub(super) fn board_tooltip(rank: usize, entry: &BoardEntry) -> String {
    let file_name = entry
        .asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let mut lines = vec![
        format!("#{rank}"),
        file_name.to_owned(),
        format!("global {:+.2}", entry.global_score),
        format!("session {:+.2}", entry.session_utility),
        format!("focus {:+.2}", entry.session_focus),
        format!("pull {:.2}", entry.sampling_pull),
        format!(
            "{} wins / {} duels",
            entry.asset.win_count, entry.asset.compare_count
        ),
        format!("certainty {:.0}%", entry.certainty * 100.0),
    ];
    if entry.residual_score.abs() >= 0.02 {
        lines.push(format!("residual {:+.2}", entry.residual_score));
    }
    if entry.session_offset.abs() >= 0.02 {
        lines.push(format!("exact offset {:+.2}", entry.session_offset));
    }
    lines.extend(asset_quality_lines(entry.quality));
    lines.join("\n")
}

fn posterior_brief(summary: PosteriorSummary) -> String {
    format_posterior(summary.mean, summary.sigma)
}

fn format_posterior(mean: f32, sigma: f32) -> String {
    format!("{} ± {}", format_scalar(mean), format_scalar(sigma))
}

fn format_scalar(value: f32) -> String {
    let abs = value.abs();
    if abs >= 100.0 {
        format!("{value:.0}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

pub(super) fn asset_quality_lines(summary: AssetQualitySummary) -> Vec<String> {
    let mut lines = vec![
        format!("asset {}", posterior_brief(summary.asset)),
        format!("base {}", posterior_brief(summary.baseline)),
    ];
    if let Some(semantic) = summary.semantic {
        lines.push(format!("semantic {}", posterior_brief(semantic)));
    }
    if let Some(vibe) = summary.vibe {
        lines.push(format!("vibe {}", posterior_brief(vibe)));
    }
    if let Some(technical) = summary.technical {
        lines.push(format!("k {}", posterior_brief(technical)));
    }
    if let Some(face) = summary.face {
        lines.push(format!("face {}", posterior_brief(face)));
    }
    lines
}

fn quality_chip(label: &str, summary: PosteriorSummary, class: &str) -> Markup {
    html! {
        span class=(format!("meta-chip quality-chip {class}")) {
            (label) " " (posterior_brief(summary))
        }
    }
}

pub(super) fn asset_quality_chips(summary: AssetQualitySummary) -> Markup {
    html! {
        div.quality-chip-row {
            (quality_chip("q", summary.asset, "quality-chip-asset"))
            (quality_chip("b", summary.baseline, "quality-chip-baseline"))
            @if let Some(semantic) = summary.semantic {
                (quality_chip("m", semantic, "quality-chip-semantic"))
            }
            @if let Some(vibe) = summary.vibe {
                (quality_chip("s", vibe, "quality-chip-vibe"))
            }
            @if let Some(technical) = summary.technical {
                (quality_chip("k", technical, "quality-chip-technical"))
            }
            @if let Some(face) = summary.face {
                (quality_chip("f", face, "quality-chip-face"))
            }
        }
    }
}

fn asset_domain_button_classes(
    label: AssetDomainLabel,
    domain: AssetDomainView,
    mini: bool,
) -> String {
    let mut classes = if mini {
        "tool mini domain-tool".to_owned()
    } else {
        "tool domain-tool".to_owned()
    };
    classes.push_str(match label {
        AssetDomainLabel::Real => " domain-tool-3d",
        AssetDomainLabel::Anime => " domain-tool-2d",
    });
    if domain
        .predicted
        .is_some_and(|prediction| prediction.label() == label)
    {
        classes.push_str(" predicted");
    }
    if domain.manual == Some(label) {
        classes.push_str(" manual active");
    }
    classes
}

fn asset_domain_button_title(label: AssetDomainLabel, domain: AssetDomainView) -> String {
    let mut cues = Vec::new();
    if domain.manual == Some(label) {
        cues.push(format!("manual {}", label.display_str()));
    }
    if let Some(prediction) = domain
        .predicted
        .filter(|prediction| prediction.label() == label)
    {
        cues.push(format!(
            "model {} {}%",
            label.display_str(),
            prediction.display_percent()
        ));
    }
    if cues.is_empty() {
        label.title().to_owned()
    } else {
        format!("{} · {}", label.title(), cues.join(" · "))
    }
}

pub(super) fn asset_domain_controls(
    asset_id: &AssetId,
    domain: AssetDomainView,
    mini: bool,
    facemash_aux: bool,
) -> Markup {
    let render_forms = |aux: bool| {
        html! {
            @for label in [AssetDomainLabel::Real, AssetDomainLabel::Anime] {
                @if aux {
                    form action="/asset/domain" method="post" data-facemash-aux="" {
                        input type="hidden" name="asset_id" value=(asset_id.0);
                        input type="hidden" name="label" value=(label.as_str());
                        button
                            class=(asset_domain_button_classes(label, domain, mini))
                            type="submit"
                            title=(asset_domain_button_title(label, domain))
                            aria-label=(asset_domain_button_title(label, domain)) {
                            (label.display_str())
                        }
                    }
                } @else {
                    form action="/asset/domain" method="post" {
                        input type="hidden" name="asset_id" value=(asset_id.0);
                        input type="hidden" name="label" value=(label.as_str());
                        button
                            class=(asset_domain_button_classes(label, domain, mini))
                            type="submit"
                            title=(asset_domain_button_title(label, domain))
                            aria-label=(asset_domain_button_title(label, domain)) {
                            (label.display_str())
                        }
                    }
                }
            }
        }
    };
    html! {
        @if facemash_aux {
            div.asset-domain-controls data-facemash-aux="" {
                (render_forms(true))
            }
        } @else {
            div.asset-domain-controls {
                (render_forms(false))
            }
        }
    }
}

pub(super) fn selection_name(asset: &AssetRecord) -> &str {
    asset
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
}
