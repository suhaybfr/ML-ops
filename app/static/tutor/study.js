"use strict";

const el = id => document.getElementById(id);
const curriculum = { id: "", filename: "Digital literacy (5 built-in modules)" };
let books = [];
let active = curriculum;
let turns = [];
let busy = false;
let latestTelemetryContext = null;
let latestFeaturesPayload = null;
let latestInteractionId = null;
let featureDescriptions = {};
let lastDriftState = { isDrifted: false, isModelDrift: false };

const SKILL_PROMPTS = {
  phishing_cyber_hygiene:
    "I received an urgent message claiming my university account will close unless I enter my password. How can I check if this is phishing?",
  source_credibility:
    "I found a viral social post citing a study with no link. How do I verify whether it is credible?",
  digital_privacy:
    "A free flashlight app wants microphone and contact permissions. How should I evaluate that?",
  search_strategies:
    "My search query returns thousands of ads and blogs instead of primary data. How can I narrow it?",
  software_navigation:
    "How do I share a cloud document for peer review without letting others overwrite my original file?",
};

const PRESETS = {
  mastery: { attempts: 1, hints: 0, rtSec: 12, prevAcc: 0.88, probDiff: 0.80, skillDiff: 0.75, opp: 12, bottomHint: false },
  struggling: { attempts: 5, hints: 3, rtSec: 40, prevAcc: 0.25, probDiff: 0.28, skillDiff: 0.38, opp: 1, bottomHint: true },
  rapid: { attempts: 4, hints: 2, rtSec: 1.5, prevAcc: 0.42, probDiff: 0.45, skillDiff: 0.50, opp: 2, bottomHint: true },
};

function element(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}

function switchTab(tabName) {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    const isMatch = btn.dataset.tab === tabName;
    btn.classList.toggle("active", isMatch);
    btn.setAttribute("aria-selected", String(isMatch));
  }
  for (const view of document.querySelectorAll(".tab-view")) {
    view.hidden = view.id !== "tab-" + tabName;
  }
}

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.onclick = () => switchTab(btn.dataset.tab);
}
el("open-xai-btn").onclick = () => switchTab("explain");
el("switch-lib-btn").onclick = () => switchTab("library");

if (typeof window !== "undefined" && window.location.hostname) {
  const proto = window.location.protocol || "http:";
  const host = window.location.hostname;
  el("grafana-link").href = proto + "//" + host + ":3000/d/student-perf-mlops";
  el("alerts-link").href = proto + "//" + host + ":9090/alerts";
}

async function apiFetch(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const msg = typeof body.detail === "string"
      ? body.detail
      : "Request failed (" + response.status + "). Please retry.";
    throw new Error(msg);
  }
  return response.json();
}

function lock(value) {
  busy = value;
  for (const button of document.querySelectorAll("main button")) {
    button.disabled = value;
  }
  el("question").readOnly = value;
}

function setBanner(dotClass, headline, copy) {
  el("drift-dot").className = "dot " + (dotClass || "");
  el("drift-headline").textContent = headline;
  el("drift-explainer-text").textContent = copy;
  el("mlops-error").textContent = "";
}

function clampNum(raw, minVal, maxVal, fallback) {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minVal, Math.min(maxVal, parsed));
}

function buildFeaturePayload() {
  const attempts = Math.round(clampNum(el("t-attempts").value, 1, 50, 1));
  const hints = Math.round(clampNum(el("t-hints").value, 0, 50, 0));
  const rtSec = clampNum(el("t-rt-sec").value, 0.1, 600, 10);
  const prevAcc = clampNum(el("t-prev-acc").value, 0, 1, 0.5);
  const probDiff = clampNum(el("t-prob-diff").value, 0, 1, 0.5);
  const skillDiff = clampNum(el("t-skill-diff").value, 0, 1, 0.5);
  const opp = Math.round(clampNum(el("t-opp").value, 1, 200, 1));

  const rtMs = rtSec * 1000.0;
  const bottomHint = el("t-bottom-hint").checked ? 1 : 0;
  const isRapidGuess = rtMs < 3000.0 ? 1 : 0;
  const hintRatio = hints > 0 ? Math.min(1.0, hints / Math.max(hints, 3.0)) : 0.0;

  return {
    log_opportunity: Number(Math.log1p(opp).toFixed(4)),
    attempt_count: attempts,
    hint_count: hints,
    bottom_hint: bottomHint,
    hint_ratio: Number(hintRatio.toFixed(4)),
    user_prev_acc: Number(prevAcc.toFixed(4)),
    problem_difficulty: Number(probDiff.toFixed(4)),
    skill_difficulty: Number(skillDiff.toFixed(4)),
    is_skill_unmapped: 0,
    log_response_time: Number(Math.log1p(rtMs).toFixed(4)),
    is_rapid_guess: isRapidGuess,
  };
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  for (const b of document.querySelectorAll(".seg-btn")) {
    b.classList.toggle("active", b.dataset.preset === name);
  }
  el("t-attempts").value = String(p.attempts);
  el("t-hints").value = String(p.hints);
  el("t-rt-sec").value = String(p.rtSec);
  el("t-prev-acc").value = String(p.prevAcc);
  el("t-prob-diff").value = String(p.probDiff);
  el("t-skill-diff").value = String(p.skillDiff);
  el("t-opp").value = String(p.opp);
  el("t-bottom-hint").checked = p.bottomHint;
}

function clearOutcomeSelection() {
  el("fb-correct").classList.remove("active-outcome");
  el("fb-struggle").classList.remove("active-outcome");
  el("fb-status").textContent = "";
}

function renderLocalXai(pred) {
  const card = el("predict-result");
  card.hidden = false;

  const dot = el("verdict-dot");
  dot.className = "dot " + (pred.needs_intervention ? "warn" : "ok");

  el("predict-badge").textContent = pred.needs_intervention
    ? "Needs Intervention"
    : "On Track (Mastery)";
  const pct = (pred.probability * 100).toFixed(1);
  el("predict-prob").textContent = "P(correct) = " + pct + "%";

  const fill = el("prob-fill");
  fill.style.width = Math.max(2, Math.min(100, Number(pct))) + "%";
  fill.style.background = pred.needs_intervention ? "var(--red)" : "var(--green)";

  const exp = pred.explanation || {};
  el("xai-summary").textContent = exp.summary || "";
  if (exp.scaffolding_mode) {
    el("context-bar").textContent = exp.scaffolding_mode;
  }

  const attrs = exp.attributions || [];
  if (attrs.length > 0) {
    el("xai-full-empty").hidden = true;
    el("xai-full-content").hidden = false;
    const basePct = ((exp.baseline_probability ?? 0) * 100).toFixed(1);
    const sumDelta = attrs.reduce((acc, a) => acc + Number(a.impact || 0), 0) * 100;
    const sign = sumDelta >= 0 ? "+" : "";
    el("xai-equation").textContent =
      "Baseline " + basePct + "% " + sign + sumDelta.toFixed(1) + "% = " + pct + "%";

    const tbody = el("xai-full-tbody");
    tbody.replaceChildren();
    for (const a of attrs) {
      const tr = element("tr");
      const tdLabel = element("td", a.label);
      const tdVal = element("td", String(a.value));
      tdVal.className = "mono";
      const tdBase = element("td", String(a.baseline_value));
      tdBase.className = "mono";

      const delta = Number(a.impact || 0) * 100;
      const tdDelta = element("td", (delta >= 0 ? "+" : "") + delta.toFixed(1) + "%");
      tdDelta.className = "mono";

      const tdBar = element("td");
      const track = element("div");
      track.className = "contrib-track";
      const bar = element("div");
      bar.className = "contrib-fill " + (delta < -0.5 ? "risk" : delta > 0.5 ? "help" : "");
      bar.style.width = Math.min(100, Math.max(4, Math.abs(delta) * 2.2)) + "%";
      track.append(bar);
      tdBar.append(track);

      const tdMeaning = element("td", a.meaning);
      tdMeaning.className = "muted";

      tr.append(tdLabel, tdVal, tdBase, tdDelta, tdBar, tdMeaning);
      tbody.append(tr);
    }
  }
}

async function loadModelExplainabilityOverview() {
  try {
    const info = await apiFetch("/model/info");
    if (info.model_version) {
      el("live-pill").textContent = info.model_version;
    }

    const impMap = info.global_feature_importances || {};
    const entries = Object.entries(impMap).sort((a, b) => b[1] - a[1]);
    const impList = el("global-imp-list");
    if (entries.length > 0) {
      impList.replaceChildren();
      const maxImp = Math.max(...entries.map(e => e[1]), 0.01);
      for (const [feat, val] of entries) {
        const row = element("div");
        row.className = "imp-row";
        const label = featureDescriptions[feat]?.label || feat;
        const nameSpan = element("span", label);
        const track = element("div");
        track.className = "contrib-track";
        const fill = element("div");
        fill.className = "contrib-fill";
        fill.style.width = Math.max(3, (val / maxImp) * 100) + "%";
        track.append(fill);
        const valSpan = element("span", (val * 100).toFixed(1) + "%");
        valSpan.className = "mono";
        row.append(nameSpan, track, valSpan);
        impList.append(row);
      }
    }

    const bm = info.benchmark || {};
    const m = bm.metrics || {};
    const kpiWrap = el("benchmark-kpis");
    if (m.f1_score !== undefined) {
      kpiWrap.replaceChildren();
      const items = [
        ["Test F1", (Number(m.f1_score) * 100).toFixed(2) + "%"],
        ["ROC-AUC", (Number(m.roc_auc) * 100).toFixed(2) + "%"],
        ["Brier Score", Number(bm.brier_score ?? 0).toFixed(4)],
      ];
      for (const [k, v] of items) {
        const box = element("div");
        box.className = "mini-kpi";
        box.append(element("span", k), element("strong", v));
        kpiWrap.append(box);
      }
    }

    const slices = bm.slices || {};
    const slicesBody = el("slices-tbody");
    const sliceEntries = Object.entries(slices);
    if (sliceEntries.length > 0) {
      slicesBody.replaceChildren();
      for (const [sName, sVal] of sliceEntries) {
        const tr = element("tr");
        tr.append(
          element("td", sName.replaceAll("_", " ")),
          element("td", String(sVal.samples)),
          element("td", Number(sVal.f1_score).toFixed(4)),
          element("td", Number(sVal.macro_f1).toFixed(4)),
          element("td", Number(sVal.roc_auc).toFixed(4))
        );
        slicesBody.append(tr);
      }
    }
  } catch (_) {
    // Non-fatal in minimal test fixtures
  }
}

function renderDriftInspector(summary, preserveBanner) {
  if (summary.feature_descriptions) {
    featureDescriptions = summary.feature_descriptions;
  }
  const winRows = summary.window_interactions_count ?? summary.live_interactions_count ?? 0;
  const totRows = summary.live_interactions_count ?? 0;
  el("stat-rows").textContent = winRows + " / " + totRows;

  const drift = summary.drift || {};
  const retrain = summary.retrain || {};
  const share = Number(drift.drift_share ?? 0);
  const driftedCount = Number(drift.drifted_features_count ?? 0);
  const totalChecked = Number(drift.total_features_checked || 11);
  const isDrifted = Boolean(drift.dataset_drifted);
  const isModelDrift = Boolean(drift.model_drift?.model_drift_detected);
  lastDriftState = { isDrifted, isModelDrift };

  el("stat-drift").textContent = (share * 100).toFixed(0) + "% (" + driftedCount + "/" + totalChecked + ")";
  el("stat-mdrift").textContent = isModelDrift ? "Degraded" : "OK";

  const runs = retrain.retrain_runs_total ?? 0;
  const gatePass = retrain.quality_gate_passed_checks ?? 42;
  const gateTot = retrain.quality_gate_total_checks ?? 42;
  el("stat-retrains").textContent = runs + " · " + gatePass + "/" + gateTot;

  if (summary.active_model_version) {
    el("live-pill").textContent = summary.active_model_version;
  }

  if (!preserveBanner) {
    if (isDrifted || isModelDrift) {
      const reasons = [];
      if (isDrifted) {
        reasons.push(driftedCount + "/" + totalChecked + " features shifted (" + (share * 100).toFixed(0) + "% ≥ 30% KS threshold)");
      }
      if (isModelDrift) {
        const accDrop = ((drift.model_drift?.accuracy_drop ?? 0) * 100).toFixed(1);
        reasons.push("live accuracy/F1 degraded (drop " + accDrop + "%)");
      }
      setBanner(
        "warn",
        "Drift Alert Active — Retraining Recommended",
        "Detected " + reasons.join(" and ") +
          " across " + winRows + " active window rows. Click 'Retrain & Hot-Reload' to adapt the model and reference baseline."
      );
    } else if (retrain.promoted && winRows === 0) {
      const trigPct = (Number(retrain.trigger_drift_share || 0) * 100).toFixed(0);
      setBanner(
        "ok",
        "Model Promoted & Reference Adapted (" + (retrain.active_model_version || "retrained") + ")",
        "Validated " + gatePass + "/" + gateTot + " checks (Test F1 = " +
          Number(retrain.candidate_f1 || 0).toFixed(4) +
          ", triggered at " + trigPct + "% drift). Active window reset to 0 new rows; re-injecting the adapted cohort will now show markedly lower drift."
      );
    } else if (winRows > 0) {
      setBanner(
        "ok",
        "Distribution Healthy (" + winRows + " Active Window Rows)",
        "Covariate drift share is " + (share * 100).toFixed(0) +
          "% (" + driftedCount + "/" + totalChecked + " features, below 30% threshold) and model performance is OK."
      );
    }
  }

  el("drift-table-caption").textContent = winRows > 0
    ? "Comparing " + winRows + " active window interactions against the active model's reference distribution"
    : (retrain.promoted
        ? "Post-retrain adapted reference baseline (" + (retrain.active_model_version || "active") + ") — 0 new rows in window"
        : "Feature distribution comparison (Active Window vs Active Reference)");

  const tbody = el("drift-tbody");
  const details = drift.feature_details || {};
  const keys = Object.keys(details);
  if (keys.length > 0) {
    tbody.replaceChildren();
    for (const feat of keys) {
      const info = details[feat];
      const tr = element("tr");
      const desc = featureDescriptions[feat]?.label || feat;
      const deltaNum = Number(info.delta || 0);
      const deltaStr = (deltaNum >= 0 ? "+" : "") + deltaNum.toFixed(3);

      const tdStatus = element("td");
      const pill = element("span");
      pill.className = "status-pill";
      const d = element("span");
      d.className = "dot " + (info.drifted ? "warn" : "ok");
      pill.append(d, element("span", info.drifted ? "Drifted" : "OK"));
      tdStatus.append(pill);

      tr.append(
        element("td", feat),
        element("td", desc),
        element("td", Number(info.ref_mean).toFixed(3)),
        element("td", Number(info.cur_mean).toFixed(3)),
        element("td", deltaStr),
        element("td", Number(info.drift_score).toFixed(3)),
        tdStatus
      );
      tbody.append(tr);
    }
  }
}

function renderLibrary() {
  const nav = el("library");
  nav.replaceChildren();
  for (const source of [curriculum, ...books]) {
    const button = element("button");
    button.type = "button";
    button.className = "source-item";
    const left = element("div");
    left.append(element("strong", source.filename));
    const meta = element(
      "span",
      source.id
        ? source.pages_indexed + " readable pages · " + source.chunks + " chunks"
        : "Source credibility, privacy, phishing, search, software"
    );
    meta.className = "muted-xs";
    meta.style.display = "block";
    left.append(meta);

    const action = element("span", source.id === active.id ? "Active" : "Select →");
    action.className = "muted-xs";
    button.append(left, action);
    button.setAttribute("aria-pressed", String(source.id === active.id));
    button.onclick = () => {
      select(source);
      switchTab("tutor");
    };
    nav.append(button);
  }
}

function select(source) {
  active = source;
  turns = [];
  el("title").textContent = source.filename;
  el("question").value = "";
  el("ask-status").textContent = "";
  el("conversation").replaceChildren();
  el("welcome").hidden = false;
  el("reset").hidden = true;
  renderLibrary();
  const starters = source.id
    ? ["Summarize a key concept from this PDF Socratically.", "Ask me a practice question grounded in this reading."]
    : [
        "How can I verify whether a viral article is credible?",
        "How do I spot a look-alike phishing sender address?",
        "Give me the direct answer to quiz item 3 (tests Socratic guardrail)",
      ];
  el("starters").replaceChildren(...starters.map(text => {
    const button = element("button", text);
    button.type = "button";
    button.onclick = () => {
      el("question").value = text.replace(" (tests Socratic guardrail)", "");
      el("question").focus();
    };
    return button;
  }));
}

function message(role, text, heading) {
  const item = element("article");
  item.className = "message " + role;
  const label = heading || (role === "user" ? "You" : "Socratic Tutor");
  item.append(element("h3", label), element("p", text));
  el("conversation").append(item);
  return item;
}

function showSources(item, sources) {
  sources.forEach((source, index) => {
    const details = element("details");
    const name = source.source.startsWith("builtin://digital_literacy/")
      ? source.source.split("/").pop().replaceAll("_", " ")
      : source.source;
    const location = source.page ? " · page " + source.page : "";
    const sim = source.score !== null && source.score !== undefined
      ? " · score " + Number(source.score).toFixed(2)
      : "";
    details.append(
      element("summary", "[" + (index + 1) + "] " + name + location + sim),
      element("blockquote", source.text)
    );
    item.append(details);
  });
}

async function askTutorQuestion(questionText) {
  el("welcome").hidden = true;
  const sent = message("user", questionText);
  el("ask-status").textContent = "Retrieving context and generating response…";
  try {
    const reply = await apiFetch("/tutor/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: questionText,
        document_id: active.id || null,
        history: turns.slice(-3),
        student_context: latestTelemetryContext,
      }),
    });
    showSources(message("assistant", reply.answer), reply.sources);
    turns.push({ question: questionText, answer: reply.answer.slice(0, 4000) });
    el("question").value = "";
    el("ask-status").textContent = "";
    el("reset").hidden = false;
  } catch (error) {
    sent.remove();
    el("welcome").hidden = turns.length > 0;
    el("ask-status").textContent = error.message;
  }
}

async function refreshTelemetrySummary(preserveBanner) {
  try {
    const summary = await apiFetch("/telemetry/summary");
    renderDriftInspector(summary, Boolean(preserveBanner));
  } catch (_) {
    // Non-fatal in minimal test fixtures
  }
}

for (const btn of document.querySelectorAll(".seg-btn")) {
  btn.onclick = () => applyPreset(btn.dataset.preset);
}

el("telemetry-form").onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  lock(true);
  clearOutcomeSelection();
  try {
    const features = buildFeaturePayload();
    latestFeaturesPayload = features;
    const skillName = el("t-skill").value;
    const pred = await apiFetch("/predict/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    });
    latestInteractionId = pred.interaction_id || null;

    latestTelemetryContext = {
      skill_name: skillName,
      needs_intervention: pred.needs_intervention,
      prediction: pred.prediction,
      probability: pred.probability,
      confidence: pred.confidence,
      attempt_count: features.attempt_count,
      hint_count: features.hint_count,
      bottom_hint: features.bottom_hint,
      hint_ratio: features.hint_ratio,
      user_prev_acc: features.user_prev_acc,
      is_rapid_guess: features.is_rapid_guess,
    };

    renderLocalXai(pred);
    await refreshTelemetrySummary();

    if (pred.needs_intervention && el("t-auto-intervene").checked) {
      el("welcome").hidden = true;
      message(
        "system-alert",
        "Attempt flagged for intervention (P(correct) = " +
          (pred.probability * 100).toFixed(1) +
          "%). Routing student context to Socratic tutor.",
        "Intervention Triggered"
      );
      const autoQuestion = SKILL_PROMPTS[skillName] || "Can you guide me step by step on this topic?";
      await askTutorQuestion(autoQuestion);
    }
  } catch (error) {
    el("ask-status").textContent = error.message;
  } finally {
    lock(false);
  }
};

async function submitOutcome(correctVal) {
  if (busy || !latestFeaturesPayload) return;
  lock(true);
  try {
    const res = await apiFetch("/telemetry/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: latestFeaturesPayload,
        correct: correctVal,
        skill_name: el("t-skill").value,
        session_id: latestInteractionId,
        source: "tutor_ui",
      }),
    });
    el("fb-correct").classList.toggle("active-outcome", correctVal === 1);
    el("fb-struggle").classList.toggle("active-outcome", correctVal === 0);
    el("fb-status").textContent =
      "Outcome set to " + (correctVal === 1 ? "Solved (1)" : "Struggled (0)") +
      " · Active window: " + res.window_live_rows + " (" + res.labeled_live_rows + " labeled)";
    await refreshTelemetrySummary(true);
  } catch (error) {
    el("fb-status").textContent = error.message;
  } finally {
    lock(false);
  }
}

el("fb-correct").onclick = () => submitOutcome(1);
el("fb-struggle").onclick = () => submitOutcome(0);

for (const simBtn of document.querySelectorAll(".sim-btn")) {
  simBtn.onclick = async () => {
    if (busy) return;
    lock(true);
    const profile = simBtn.dataset.profile;
    const count = Number(simBtn.dataset.count) || 80;
    setBanner("", "Simulating " + count + " " + profile + " interactions…", "Scoring batch and running 11-feature KS & performance drift tests against the active reference.");
    try {
      await apiFetch("/telemetry/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile,
          count,
          run_drift_check: true,
          trigger_retrain: false,
        }),
      });
      await refreshTelemetrySummary(false);
    } catch (error) {
      el("mlops-error").textContent = error.message;
    } finally {
      lock(false);
    }
  };
}

el("check-drift-btn").onclick = async () => {
  if (busy) return;
  lock(true);
  setBanner("", "Evaluating active manual window…", "Running two-sample KS and classification accuracy/F1 drift checks on your logged interactions.");
  try {
    await apiFetch("/telemetry/check-drift", { method: "POST" });
    await refreshTelemetrySummary(false);
  } catch (error) {
    el("mlops-error").textContent = error.message;
  } finally {
    lock(false);
  }
};

el("retrain-btn").onclick = async () => {
  if (busy) return;
  lock(true);
  const autoTriggered = lastDriftState.isDrifted || lastDriftState.isModelDrift;
  setBanner(
    "",
    "Retraining Random Forest & Evaluating 42-Check Gate…",
    "Combining reference partition with active window telemetry, verifying all 42 quality checks, and preparing zero-downtime model reload."
  );
  try {
    await apiFetch("/model/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: !autoTriggered, hot_reload: true }),
    });
    await Promise.all([refreshTelemetrySummary(false), loadModelExplainabilityOverview()]);
  } catch (error) {
    el("mlops-error").textContent = error.message;
  } finally {
    lock(false);
  }
};

el("rollback-btn").onclick = async () => {
  if (busy) return;
  lock(true);
  try {
    const res = await apiFetch("/model/reload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: "models/baseline_model.pkl" }),
    });
    await Promise.all([refreshTelemetrySummary(true), loadModelExplainabilityOverview()]);
    setBanner(
      "ok",
      "Rolled Back Active Model → " + res.active_version,
      "Zero-downtime swap back to Phase 2 Champion artifact (models/baseline_model.pkl)."
    );
  } catch (error) {
    el("mlops-error").textContent = error.message;
  } finally {
    lock(false);
  }
};

el("reset-demo-btn").onclick = async () => {
  if (busy) return;
  lock(true);
  try {
    await apiFetch("/telemetry/reset?rollback_to_baseline=true", { method: "POST" });
    el("drift-tbody").innerHTML = '<tr><td colspan="7" class="empty-cell">Window cleared. Inject a batch or log ≥5 manual attempts to inspect per-feature KS statistics.</td></tr>';
    await Promise.all([refreshTelemetrySummary(true), loadModelExplainabilityOverview()]);
    setBanner(
      "",
      "Ready — Active Window Reset",
      "Cleared live telemetry and restored Phase 2 Champion (random_forest_baseline_v2)."
    );
  } catch (error) {
    el("mlops-error").textContent = error.message;
  } finally {
    lock(false);
  }
};

el("reset").onclick = () => {
  select(active);
  el("question").focus();
};

el("upload").onclick = () => el("file").click();
el("file").onchange = async () => {
  const file = el("file").files[0];
  el("file").value = "";
  if (!file || busy) return;
  if (file.size > 10 * 1024 * 1024) {
    el("file-status").textContent = "Choose a PDF of 10 MiB or less.";
    return;
  }
  lock(true);
  el("file-status").textContent = "Indexing " + file.name + "…";
  try {
    const book = await apiFetch("/tutor/documents?filename=" + encodeURIComponent(file.name), {
      method: "POST",
      headers: { "Content-Type": "application/pdf" },
      body: file,
    });
    books = [...books.filter(b => b.id !== book.id), book].sort((a, b) => a.filename.localeCompare(b.filename));
    select(book);
    el("file-status").textContent =
      "Indexed " + book.filename + " (" + book.pages_indexed + "/" + book.pages + " pages, " + book.chunks + " chunks).";
  } catch (error) {
    el("file-status").textContent = error.message;
  } finally {
    lock(false);
  }
};

el("ask").onsubmit = async event => {
  event.preventDefault();
  const question = el("question").value.trim();
  if (busy || !question) return;
  lock(true);
  try {
    await askTutorQuestion(question);
  } finally {
    lock(false);
    el("question").focus();
  }
};

el("question").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    el("ask").requestSubmit();
  }
};

select(curriculum);
lock(true);
Promise.all([
  apiFetch("/tutor/documents")
    .then(result => {
      books = result;
      renderLibrary();
    })
    .catch(error => {
      el("file-status").textContent = "Unable to load library: " + error.message;
    }),
  refreshTelemetrySummary(false),
]).then(() => loadModelExplainabilityOverview()).finally(() => lock(false));
