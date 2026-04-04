use super::*;
use rand::{prelude::IndexedRandom, rng};

const VOCAB_LAB_SHORTLIST: usize = 48;

#[derive(Debug, Deserialize, Default)]
pub(super) struct VocabLabQuery {
    asset_id: Option<String>,
}

pub(super) async fn vocab_lab_root(
    State(state): State<SharedRuntimeState>,
    Query(query): Query<VocabLabQuery>,
) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    log_site_loaded("/vocab-lab");
    Ok(render_markup(routed_layout(
        "vocab-lab-page",
        PageGeometry::Document,
        NavPage::Vocab,
        None,
        vocab_lab_markup(vocab_lab_asset(&state, query.asset_id.as_deref())?),
    )))
}

pub(super) async fn vocab_lab_reroll(
    State(state): State<SharedRuntimeState>,
) -> WebResult<Response> {
    let state = match ready_app_or_snapshot(&state) {
        Ok(state) => state,
        Err(snapshot) => return Ok(boot_response(snapshot)),
    };
    let Some(asset) = sampled_vocab_asset(&state)? else {
        return Ok(Redirect::to("/vocab-lab").into_response());
    };
    Ok(Redirect::to(&format!("/vocab-lab?asset_id={}", asset.id.0)).into_response())
}

fn vocab_lab_asset(
    state: &SharedAppState,
    requested: Option<&str>,
) -> anyhow::Result<Option<AssetRecord>> {
    if let Some(asset_id) = requested {
        return state
            .maybe_image_asset(&AssetId(asset_id.to_owned()))
            .map(|asset| asset.filter(|asset| !asset.hidden && asset.path.exists()));
    }
    sampled_vocab_asset(state)
}

fn sampled_vocab_asset(state: &SharedAppState) -> anyhow::Result<Option<AssetRecord>> {
    let mut candidates = state
        .board()?
        .entries
        .into_iter()
        .take(VOCAB_LAB_SHORTLIST)
        .map(|entry| entry.asset)
        .filter(|asset| !asset.hidden && asset.path.exists())
        .collect::<Vec<_>>();
    let mut rng = rng();
    Ok(candidates.as_mut_slice().choose(&mut rng).cloned())
}

fn vocab_lab_markup(asset: Option<AssetRecord>) -> Markup {
    let Some(asset) = asset else {
        return html! {
            section.vocab-lab-shell {
                section.vocab-lab-empty.swarm-surface-elevated {
                    h1 { "vocab lab" }
                    p { "No visible local assets are available yet." }
                    a.ts-tool href="/board" { "board" }
                }
            }
        };
    };
    let image_src = asset_src(&asset, AssetRendition::Preview);
    let file_name = selection_name(&asset).to_owned();
    html! {
        section.vocab-lab-shell data-vocab-lab="" {
            section.vocab-lab-stage.swarm-surface-elevated {
                header.vocab-lab-header {
                    div.vocab-lab-kicker { "sam 3 vocab lab" }
                    h1.vocab-lab-title { (file_name) }
                    p.vocab-lab-subtitle {
                        "MVP prompt-workflow lab only. No SAM inference is wired yet; this page builds "
                        "the exact prompt/spec you would later send into a segregated SAM 3 adapter."
                    }
                }
                div.vocab-lab-canvas-shell {
                    div.vocab-lab-canvas
                        id="vocab-lab-canvas"
                        data-asset-id=(asset.id.0)
                        data-asset-name=(selection_name(&asset))
                        data-image-src=(image_src) {
                        img.vocab-lab-image
                            id="vocab-lab-image"
                            src=(image_src)
                            alt=(selection_name(&asset));
                        div.vocab-lab-overlay id="vocab-lab-overlay" {}
                    }
                }
                div.vocab-lab-asset-rail {
                    a.ts-tool href=(format!("/vocab-lab?asset_id={}", asset.id.0)) { "permalink" }
                    a.ts-tool href="/vocab-lab/reroll" { "sample" }
                    a.ts-tool href="/board" { "board" }
                    a.ts-tool href=(format!("/explore?mode=raw&focus={}", asset.id.0)) { "explore" }
                }
            }
            aside.vocab-lab-sidebar {
                section.vocab-lab-panel.swarm-surface-elevated {
                    h2 { "Prompt Draft" }
                    label.vocab-lab-field {
                        span { "concept name" }
                        input
                            id="vocab-concept-name"
                            type="text"
                            placeholder="glasses";
                    }
                    label.vocab-lab-field {
                        span { "measurement" }
                        select id="vocab-measurement" {
                            option value="presence" selected { "presence" }
                            option value="count" { "count" }
                            option value="absence" { "absence" }
                        }
                    }
                    div.vocab-lab-inline {
                        input
                            id="vocab-phrase-input"
                            type="text"
                            placeholder="short noun phrase";
                        button.ts-tool type="button" id="vocab-add-phrase" { "add phrase" }
                    }
                    div.vocab-lab-chip-stack id="vocab-phrases" {}
                    div.vocab-lab-inline.vocab-lab-mode-row {
                        button.ts-tool.is-active type="button" id="vocab-mode-positive" { "+ point" }
                        button.ts-tool type="button" id="vocab-mode-negative" { "- point" }
                        button.ts-tool type="button" id="vocab-clear-points" { "clear points" }
                        button.ts-tool type="button" id="vocab-clear-phrases" { "clear phrases" }
                    }
                    label.vocab-lab-field {
                        span { "notes" }
                        textarea id="vocab-notes" rows="4" placeholder="what this concept should catch, what should not count, known failure modes" {}
                    }
                    div.vocab-lab-inline {
                        button.ts-tool type="button" id="vocab-save-concept" { "save concept" }
                        button.ts-tool type="button" id="vocab-export-concepts" { "export all" }
                    }
                }
                section.vocab-lab-panel.swarm-surface-elevated {
                    h2 { "Workflow" }
                    ol.vocab-lab-steps {
                        li { "Start text-only with one short noun phrase." }
                        li { "If SAM would overfire, add negative points on distractors." }
                        li { "If SAM would underfire, add a positive point on the canonical instance." }
                        li { "Only promote concepts that feel robust across many different images." }
                    }
                }
                section.vocab-lab-panel.swarm-surface-elevated {
                    h2 { "Saved Concepts" }
                    div.vocab-lab-saved id="vocab-saved-concepts" {}
                }
                section.vocab-lab-panel.swarm-surface-elevated {
                    h2 { "Request Preview" }
                    pre.vocab-lab-preview id="vocab-request-preview" {}
                }
            }
            (vocab_lab_script_block())
        }
    }
}

fn vocab_lab_script_block() -> Markup {
    html! {
        script {
            (PreEscaped(r#"
                (() => {
                  const root = document.querySelector('[data-vocab-lab]');
                  if (!root) return;

                  const storageKey = 'picmash.sam3.vocab_lab.concepts.v1';
                  const canvas = document.getElementById('vocab-lab-canvas');
                  const image = document.getElementById('vocab-lab-image');
                  const overlay = document.getElementById('vocab-lab-overlay');
                  const conceptName = document.getElementById('vocab-concept-name');
                  const measurement = document.getElementById('vocab-measurement');
                  const phraseInput = document.getElementById('vocab-phrase-input');
                  const phraseRail = document.getElementById('vocab-phrases');
                  const notes = document.getElementById('vocab-notes');
                  const savedRail = document.getElementById('vocab-saved-concepts');
                  const preview = document.getElementById('vocab-request-preview');
                  const positiveButton = document.getElementById('vocab-mode-positive');
                  const negativeButton = document.getElementById('vocab-mode-negative');
                  const clearPointsButton = document.getElementById('vocab-clear-points');
                  const clearPhrasesButton = document.getElementById('vocab-clear-phrases');
                  const addPhraseButton = document.getElementById('vocab-add-phrase');
                  const saveConceptButton = document.getElementById('vocab-save-concept');
                  const exportConceptsButton = document.getElementById('vocab-export-concepts');

                  const state = {
                    mode: 'positive',
                    phrases: [],
                    points: [],
                    saved: loadSaved(),
                  };

                  function loadSaved() {
                    try {
                      const parsed = JSON.parse(localStorage.getItem(storageKey) || '[]');
                      return Array.isArray(parsed) ? parsed : [];
                    } catch (_) {
                      return [];
                    }
                  }

                  function persistSaved() {
                    localStorage.setItem(storageKey, JSON.stringify(state.saved));
                  }

                  function setMode(mode) {
                    state.mode = mode;
                    positiveButton.classList.toggle('is-active', mode === 'positive');
                    negativeButton.classList.toggle('is-active', mode === 'negative');
                  }

                  function addPhrase(raw) {
                    const phrase = raw.trim();
                    if (!phrase || state.phrases.includes(phrase)) return;
                    state.phrases.push(phrase);
                    phraseInput.value = '';
                    renderPhrases();
                    renderPreview();
                  }

                  function renderPhrases() {
                    phraseRail.replaceChildren();
                    for (const phrase of state.phrases) {
                      const chip = document.createElement('button');
                      chip.type = 'button';
                      chip.className = 'vocab-lab-chip';
                      chip.textContent = phrase;
                      chip.title = 'remove phrase';
                      chip.addEventListener('click', () => {
                        state.phrases = state.phrases.filter((entry) => entry !== phrase);
                        renderPhrases();
                        renderPreview();
                      });
                      phraseRail.append(chip);
                    }
                  }

                  function renderPoints() {
                    overlay.replaceChildren();
                    for (const point of state.points) {
                      const marker = document.createElement('div');
                      marker.className = `vocab-lab-point vocab-lab-point--${point.label}`;
                      marker.style.left = `${point.x * 100}%`;
                      marker.style.top = `${point.y * 100}%`;
                      marker.title = `${point.label} ${(point.x * 100).toFixed(1)}%, ${(point.y * 100).toFixed(1)}%`;
                      marker.textContent = point.label === 'positive' ? '+' : '−';
                      overlay.append(marker);
                    }
                  }

                  function currentConcept() {
                    return {
                      schema_version: 1,
                      concept_name: conceptName.value.trim(),
                      measurement: measurement.value,
                      phrases: [...state.phrases],
                      notes: notes.value.trim(),
                      prompts: {
                        text: [...state.phrases],
                        points: state.points.map((point) => ({
                          x: Number(point.x.toFixed(4)),
                          y: Number(point.y.toFixed(4)),
                          label: point.label,
                        })),
                      },
                    };
                  }

                  function requestPreview() {
                    const assetId = canvas.dataset.assetId;
                    return {
                      asset_id: assetId,
                      workflow: 'sam3_text_then_point_refine',
                      concept: currentConcept(),
                      sam3_request_preview: {
                        text_prompts: [...state.phrases],
                        point_prompts: state.points.map((point) => ({
                          x: Number(point.x.toFixed(4)),
                          y: Number(point.y.toFixed(4)),
                          label: point.label === 'positive' ? 1 : 0,
                        })),
                        selection_policy: state.points.length > 0 ? 'refine ambiguous matches' : 'segment all matching instances',
                      },
                    };
                  }

                  function renderPreview() {
                    preview.textContent = JSON.stringify(requestPreview(), null, 2);
                  }

                  function renderSaved() {
                    savedRail.replaceChildren();
                    if (state.saved.length === 0) {
                      const note = document.createElement('p');
                      note.className = 'vocab-lab-note';
                      note.textContent = 'Saved concepts live in browser localStorage for now.';
                      savedRail.append(note);
                      return;
                    }
                    for (const concept of state.saved) {
                      const row = document.createElement('div');
                      row.className = 'vocab-lab-saved-row';
                      const load = document.createElement('button');
                      load.type = 'button';
                      load.className = 'ts-tool';
                      load.textContent = concept.concept_name || '(unnamed concept)';
                      load.addEventListener('click', () => {
                        conceptName.value = concept.concept_name || '';
                        measurement.value = concept.measurement || 'presence';
                        notes.value = concept.notes || '';
                        state.phrases = Array.isArray(concept.phrases) ? [...concept.phrases] : [];
                        state.points = Array.isArray(concept.prompts?.points)
                          ? concept.prompts.points.map((point) => ({
                              x: point.x,
                              y: point.y,
                              label: point.label === 'negative' ? 'negative' : 'positive',
                            }))
                          : [];
                        renderPhrases();
                        renderPoints();
                        renderPreview();
                      });
                      const drop = document.createElement('button');
                      drop.type = 'button';
                      drop.className = 'ts-tool';
                      drop.textContent = 'drop';
                      drop.addEventListener('click', () => {
                        state.saved = state.saved.filter((entry) => entry !== concept);
                        persistSaved();
                        renderSaved();
                      });
                      row.append(load, drop);
                      savedRail.append(row);
                    }
                  }

                  addPhraseButton.addEventListener('click', () => addPhrase(phraseInput.value));
                  phraseInput.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      addPhrase(phraseInput.value);
                    }
                  });

                  positiveButton.addEventListener('click', () => setMode('positive'));
                  negativeButton.addEventListener('click', () => setMode('negative'));
                  clearPointsButton.addEventListener('click', () => {
                    state.points = [];
                    renderPoints();
                    renderPreview();
                  });
                  clearPhrasesButton.addEventListener('click', () => {
                    state.phrases = [];
                    renderPhrases();
                    renderPreview();
                  });

                  [conceptName, measurement, notes].forEach((element) => {
                    element.addEventListener('input', renderPreview);
                  });

                  saveConceptButton.addEventListener('click', () => {
                    const concept = currentConcept();
                    state.saved.unshift(concept);
                    persistSaved();
                    renderSaved();
                  });

                  exportConceptsButton.addEventListener('click', () => {
                    const blob = new Blob([JSON.stringify(state.saved, null, 2)], { type: 'application/json' });
                    const href = URL.createObjectURL(blob);
                    const anchor = document.createElement('a');
                    anchor.href = href;
                    anchor.download = 'sam3-vocab-concepts.json';
                    anchor.click();
                    URL.revokeObjectURL(href);
                  });

                  image.addEventListener('click', (event) => {
                    const rect = image.getBoundingClientRect();
                    if (!rect.width || !rect.height) return;
                    const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
                    const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
                    state.points.push({ x, y, label: state.mode });
                    renderPoints();
                    renderPreview();
                  });

                  setMode('positive');
                  renderPhrases();
                  renderPoints();
                  renderSaved();
                  renderPreview();
                })();
            "#))
        }
    }
}
