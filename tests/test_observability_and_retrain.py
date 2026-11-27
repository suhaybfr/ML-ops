"""Offline unit tests for Phase 3 observability, alert wiring, telemetry, simulation & retrain."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import yaml

from scripts.simulate_interactions import simulate_locally
from src.features import ENGINEERED_FEATURES
from src.retrain import retrain_cycle
from src.telemetry import get_live_buffer_stats, load_live_interactions

REPO_ROOT = Path(__file__).resolve().parent.parent


def _extract_promql_metric_names(expr: str) -> set[str]:
    """Extract candidate Prometheus metric names referenced in an alert rule expression."""
    keywords = {
        "rate",
        "increase",
        "sum",
        "by",
        "or",
        "and",
        "vector",
        "count",
        "histogram_quantile",
        "up",
        "ALERTS",
        "job",
        "le",
        "alertstate",
        "firing",
    }
    cleaned = re.sub(r'"[^"]*"', "", expr)
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cleaned))
    return {t for t in tokens if t not in keywords and not t.startswith("student_")}


def test_metrics_endpoint_exposes_all_pipeline_groups_and_defaults(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    expected_metrics = [
        "model_loaded",
        "model_info_active",
        "model_reload_total",
        "champion_offline_metric",
        "quality_gate_passed",
        "quality_gate_checks_total",
        "quality_gate_checks_passed",
        "prediction_requests_total",
        "prediction_latency_seconds",
        "prediction_errors_total",
        "predictions_scored_total",
        "prediction_probability",
        "live_needs_intervention_ratio",
        "live_interactions_logged_total",
        "data_drift_share",
        "dataset_drifted",
        "drifted_features_count",
        "feature_drift_detected",
        "feature_drift_score",
        "model_drift_detected",
        "model_drift_f1_drop",
        "retrain_runs_total",
        "retrain_promotions_total",
        "tutor_requests_total",
        "tutor_errors_total",
        "tutor_latency_seconds",
        "tutor_indexed_documents_total",
    ]
    for metric_name in expected_metrics:
        assert metric_name in text, f"Missing expected Prometheus metric: {metric_name}"

    # Verify all 11 canonical features are initialized in feature_drift_score
    for feat in ENGINEERED_FEATURES:
        assert f'feature_drift_score{{feature="{feat}"}}' in text


def test_every_prometheus_alert_rule_references_a_real_metric(client):
    metrics_text = client.get("/metrics").text
    rules_path = REPO_ROOT / "monitoring" / "prometheus" / "alert_rules.yml"
    rules_doc = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

    alert_names = set()
    for group in rules_doc["groups"]:
        for rule in group["rules"]:
            alert_names.add(rule["alert"])
            referenced = _extract_promql_metric_names(rule["expr"])
            for metric_name in referenced:
                assert metric_name in metrics_text, (
                    f"Alert {rule['alert']!r} references {metric_name!r}, "
                    "which is NOT emitted by GET /metrics!"
                )

    required_alerts = {
        "APIDown",
        "ModelNotLoaded",
        "HighPredictionErrorRate",
        "SlowPredictions",
        "HighDataDriftDetected",
        "ModelPerformanceDegraded",
        "HighInterventionRate",
        "TutorEndpointErrors",
        "QualityGateFailed",
    }
    assert required_alerts <= alert_names


def test_grafana_dashboard_is_valid_5_row_full_pipeline():
    ds_path = REPO_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    assert ds_path.exists(), "Grafana requires YAML datasource provisioning (.yml)"
    ds_cfg = yaml.safe_load(ds_path.read_text(encoding="utf-8"))
    assert ds_cfg["datasources"][0]["uid"] == "prometheus"
    assert ds_cfg["datasources"][0]["isDefault"] is True

    dash_path = REPO_ROOT / "monitoring" / "grafana" / "dashboards" / "mlops_dashboard.json"
    dash = json.loads(dash_path.read_text(encoding="utf-8"))

    assert "svm" not in dash.get("tags", [])
    rows = [p for p in dash["panels"] if p.get("type") == "row"]
    assert len(rows) == 5
    non_row_panels = [p for p in dash["panels"] if p.get("type") != "row"]
    assert len(non_row_panels) >= 20
    assert all(p.get("datasource", {}).get("uid") == "prometheus" for p in non_row_panels)


def test_predict_explain_returns_counterfactual_xai_attributions(
    client, mastery_payload, struggling_payload
):
    if not (REPO_ROOT / "models" / "baseline_model.pkl").exists():
        return

    r_st = client.post("/predict/explain", json=struggling_payload)
    assert r_st.status_code == 200, r_st.text
    st_body = r_st.json()
    assert st_body["needs_intervention"] is True
    exp = st_body["explanation"]
    assert "Heavy Socratic Scaffolding" in exp["scaffolding_mode"]
    assert len(exp["attributions"]) == len(ENGINEERED_FEATURES)
    # Shapley efficiency property: sum(phi_i) == P(x) - P(baseline)
    total_phi = sum(item["impact"] for item in exp["attributions"])
    expected_diff = st_body["probability"] - exp["baseline_probability"]
    assert abs(total_phi - expected_diff) < 0.015
    # attempt_count=5 must carry substantial negative attribution
    attempt_attr = next(a for a in exp["attributions"] if a["feature"] == "attempt_count")
    assert attempt_attr["impact"] < -0.10

    r_ok = client.post("/predict/explain", json=mastery_payload)
    assert r_ok.status_code == 200, r_ok.text
    assert "Light Socratic Extension" in r_ok.json()["explanation"]["scaffolding_mode"]

    # /model/info exposes global_feature_importances and benchmark slices
    info = client.get("/model/info").json()
    assert len(info["global_feature_importances"]) == 11
    assert "mapped_skills" in info["benchmark"]["slices"]

    # Reset endpoint clears live telemetry cleanly
    r_reset = client.post("/telemetry/reset")
    assert r_reset.status_code == 200
    assert get_live_buffer_stats()["total_rows"] == 0


def test_outcome_upsert_deduplication_and_8_manual_bad_outcomes_drift(
    client, mastery_payload
):
    if not (REPO_ROOT / "models" / "baseline_model.pkl").exists():
        return

    client.post("/telemetry/reset")

    # 1. Single attempt evaluated -> 1 row; clicking outcome multiple times updates in-place!
    pred = client.post("/predict/explain", json=mastery_payload).json()
    iid = pred["interaction_id"]
    assert get_live_buffer_stats()["total_rows"] == 1

    fb1 = client.post(
        "/telemetry/feedback",
        json={"features": mastery_payload, "correct": 0, "session_id": iid},
    ).json()
    assert fb1["updated_in_place"] is True
    assert fb1["total_live_rows"] == 1
    assert fb1["labeled_live_rows"] == 1

    fb2 = client.post(
        "/telemetry/feedback",
        json={"features": mastery_payload, "correct": 1, "session_id": iid},
    ).json()
    assert fb2["updated_in_place"] is True
    assert fb2["total_live_rows"] == 1
    assert fb2["labeled_live_rows"] == 1

    # Flip first row back to bad outcome (0) and log 7 more purposeful bad manual outcomes (total 8)
    client.post(
        "/telemetry/feedback",
        json={"features": mastery_payload, "correct": 0, "session_id": iid},
    )
    for _ in range(7):
        p = client.post("/predict/explain", json=mastery_payload).json()
        client.post(
            "/telemetry/feedback",
            json={"features": mastery_payload, "correct": 0, "session_id": p["interaction_id"]},
        )

    assert get_live_buffer_stats()["total_rows"] == 8
    assert get_live_buffer_stats()["labeled_rows"] == 8

    # 2. Check drift on just those 8 manual bad outcomes -> detects concept/performance drift!
    chk = client.post("/telemetry/check-drift")
    assert chk.status_code == 200, chk.text
    chk_body = chk.json()
    assert chk_body["window_rows"] == 8
    assert chk_body["model_drift_detected"] is True
    assert chk_body["accuracy_drop"] > 0.50


def test_re_injecting_drifted_cohort_after_retrain_has_lower_drift(client, tmp_path):
    baseline = REPO_ROOT / "models" / "baseline_model.pkl"
    if not baseline.exists():
        return

    client.post("/telemetry/reset")

    # 1. Inject drifted cohort BEFORE retrain -> high drift (>= 50%)
    sim1 = client.post(
        "/telemetry/simulate",
        json={"profile": "drifted", "count": 120, "seed": 21, "run_drift_check": True},
    ).json()
    assert sim1["dataset_drifted"] is True
    assert sim1["drift_share"] >= 0.50

    # 2. Run retrain cycle to adapt active_reference.csv to the new cohort
    drift_dir = Path(client.get("/telemetry/summary").json()["retrain"].get("noop", ""))
    drift_dir = Path(tmp_path / "drift")
    live_buf = Path(tmp_path / "live" / "interactions.csv")
    synth_train = REPO_ROOT / "data" / "processed" / "train.csv"
    synth_test = REPO_ROOT / "data" / "processed" / "test.csv"

    res = retrain_cycle(
        ref_path=str(synth_train),
        cur_path=str(live_buf),
        test_path=str(synth_test),
        from_live=True,
        force=False,
        register=False,
        output_dir=drift_dir,
        status_path=drift_dir / "latest_status.json",
        challenger_path=tmp_path / "chal.pkl",
        promoted_path=tmp_path / "prom.pkl",
        tracking_uri=None,
    )
    assert res["promoted"] is True

    # 3. Inject the SAME drifted cohort AGAIN after retrain -> drift is now < 30% (adapted!)
    sim2 = client.post(
        "/telemetry/simulate",
        json={"profile": "drifted", "count": 120, "seed": 22, "run_drift_check": True},
    ).json()
    assert sim2["drift_share"] < sim1["drift_share"]
    assert sim2["drift_share"] < 0.30
    assert sim2["dataset_drifted"] is False


def test_model_reload_hot_swap_and_security(client, tmp_path):
    baseline = REPO_ROOT / "models" / "baseline_model.pkl"
    if not baseline.exists():
        return

    # Reload current model in place
    r1 = client.post("/model/reload")
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["status"] == "reloaded"
    assert body1["model_type"] == "RandomForestClassifier"

    # Reload with custom version label
    r2 = client.post(
        "/model/reload",
        json={"model_path": "models/baseline_model.pkl", "version_label": "rf_hot_swapped_v9"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["active_version"] == "rf_hot_swapped_v9"
    assert client.get("/health").json()["model_version"] == "rf_hot_swapped_v9"

    # Reject path traversal outside repo
    assert client.post("/model/reload", json={"model_path": "/etc/passwd"}).status_code == 400
    assert (
        client.post("/model/reload", json={"model_path": "models/does_not_exist.pkl"}).status_code
        == 404
    )


def test_live_telemetry_logging_feedback_and_api_simulation(
    client, mastery_payload, struggling_payload
):
    if not (REPO_ROOT / "models" / "baseline_model.pkl").exists():
        return

    # 1. /predict logs to live buffer
    r_pred = client.post("/predict", json=mastery_payload)
    assert r_pred.status_code == 200
    stats = get_live_buffer_stats()
    assert stats["total_rows"] == 1
    assert stats["labeled_rows"] == 0

    # 2. /telemetry/feedback logs labeled outcome
    r_fb = client.post(
        "/telemetry/feedback",
        json={
            "features": struggling_payload,
            "correct": 0,
            "skill_name": "phishing_cyber_hygiene",
            "source": "tutor_ui",
        },
    )
    assert r_fb.status_code == 200, r_fb.text
    fb_body = r_fb.json()
    assert fb_body["logged"] is True
    assert fb_body["needs_intervention"] is True
    assert fb_body["total_live_rows"] == 2
    assert fb_body["labeled_live_rows"] == 1

    # 3. /telemetry/simulate generates drifted batch, updates drift JSON & /metrics
    r_sim = client.post(
        "/telemetry/simulate",
        json={
            "profile": "drifted",
            "count": 80,
            "seed": 11,
            "run_drift_check": True,
            "trigger_retrain": False,
        },
    )
    assert r_sim.status_code == 200, r_sim.text
    sim = r_sim.json()
    assert sim["generated_rows"] == 80
    assert sim["total_live_rows"] == 82
    assert sim["dataset_drifted"] is True
    assert sim["drift_share"] >= 0.3

    # Verify /metrics surfaces the newly detected drift & live rows
    metrics_text = client.get("/metrics").text
    assert "dataset_drifted 1.0" in metrics_text
    assert "live_interactions_logged_total 82.0" in metrics_text

    # Verify /telemetry/summary reflects live state
    summary = client.get("/telemetry/summary").json()
    assert summary["live_interactions_count"] == 82
    assert summary["drift"]["dataset_drifted"] is True


def test_cli_simulator_local_mode(tmp_path):
    buf = tmp_path / "sim_interactions.csv"
    out = simulate_locally(
        profile="struggling",
        rows=30,
        seed=7,
        model_path=str(REPO_ROOT / "models" / "baseline_model.pkl"),
        buffer_path=buf,
        run_drift=False,
        run_retrain=False,
    )
    assert out["generated_rows"] == 30
    df = load_live_interactions(buffer_path=buf, labeled_only=True)
    assert len(df) == 30


def test_retrain_cycle_writes_status_and_preserves_baseline_model(client, tmp_path):
    baseline = REPO_ROOT / "models" / "baseline_model.pkl"
    baseline_mtime_before = baseline.stat().st_mtime if baseline.exists() else None

    # Populate synthetic drifted live buffer
    live_csv = tmp_path / "live_interactions.csv"
    simulate_locally(
        profile="drifted",
        rows=60,
        seed=19,
        model_path=str(baseline),
        buffer_path=live_csv,
        run_drift=False,
        run_retrain=False,
    )

    drift_dir = tmp_path / "drift"
    status_file = drift_dir / "latest_status.json"
    challenger_pkl = tmp_path / "challenger.pkl"
    promoted_pkl = tmp_path / "promoted.pkl"
    db_uri = f"sqlite:///{tmp_path / 'retrain_mlflow.db'}"

    res = retrain_cycle(
        ref_path=str(tmp_path / "missing_train.csv"),
        cur_path=str(live_csv),
        test_path=str(tmp_path / "missing_test.csv"),
        from_live=True,
        force=True,
        register=False,
        output_dir=drift_dir,
        status_path=status_file,
        challenger_path=challenger_pkl,
        promoted_path=promoted_pkl,
        tracking_uri=db_uri,
        experiment_name="test_retrain_exp",
    )

    assert res["retrained"] is True
    assert status_file.exists()
    persisted = json.loads(status_file.read_text(encoding="utf-8"))
    assert persisted["retrain_runs_total"] == 1
    assert persisted["candidate_f1"] > 0.0
    assert challenger_pkl.exists()
    assert joblib.load(challenger_pkl) is not None

    # Phase 2 champion baseline_model.pkl must NEVER be altered
    if baseline_mtime_before is not None:
        assert baseline.stat().st_mtime == baseline_mtime_before
