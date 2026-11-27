"""Tests for Phase 3 data & model drift induction, detection, and alert rules."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import yaml  # noqa: E402

from src.features import ENGINEERED_FEATURES, select_xy, validate_engineered_dataframe  # noqa: E402
from src.induce_drift import (  # noqa: E402
    generate_synthetic_baseline,
    induce_concept_drift,
    induce_data_drift,
    run_drift_induction,
)
from src.monitoring import compute_drift_report, evaluate_model_drift  # noqa: E402


def test_drift_induction_preserves_11_feature_contract():
    ref = generate_synthetic_baseline(n_rows=300, seed=10)
    data_drifted = induce_data_drift(ref, intensity=1.2, seed=10)
    both_drifted = induce_concept_drift(data_drifted, flip_rate=0.4, seed=10)

    validate_engineered_dataframe(ref)
    validate_engineered_dataframe(data_drifted)
    validate_engineered_dataframe(both_drifted)

    assert data_drifted["hint_count"].mean() > ref["hint_count"].mean()
    assert data_drifted["user_prev_acc"].mean() < ref["user_prev_acc"].mean()
    assert (both_drifted["correct"] != data_drifted["correct"]).mean() > 0.1


def test_compute_drift_report_detects_no_drift_vs_induced_drift(tmp_path):
    ref = generate_synthetic_baseline(n_rows=500, seed=42)
    same_dist = generate_synthetic_baseline(n_rows=500, seed=43)
    drifted = induce_data_drift(same_dist, intensity=1.2, seed=42)

    clean_res = compute_drift_report(
        ref,
        same_dist,
        output_dir=str(tmp_path / "clean"),
        tag="clean",
        log_to_mlflow=False,
    )
    assert clean_res["total_features_checked"] == len(ENGINEERED_FEATURES)
    assert clean_res["drift_share"] < 0.30

    drift_res = compute_drift_report(
        ref,
        drifted,
        output_dir=str(tmp_path / "drifted"),
        tag="drifted",
        log_to_mlflow=False,
    )
    assert drift_res["total_features_checked"] == len(ENGINEERED_FEATURES)
    assert drift_res["drift_share"] >= 0.30
    assert drift_res["dataset_drifted"] is True
    assert Path(drift_res["html_path"]).exists()
    assert Path(drift_res["json_path"]).exists()
    assert Path(drift_res["latest_summary_path"]).exists()


def test_evaluate_model_drift_detects_concept_degradation(tmp_path):
    ref = generate_synthetic_baseline(n_rows=600, seed=99)
    X_ref, y_ref = select_xy(ref)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(X_ref, y_ref)

    cur_base = generate_synthetic_baseline(n_rows=400, seed=100)
    cur_drifted = induce_concept_drift(
        induce_data_drift(cur_base, intensity=1.0, seed=100),
        flip_rate=0.45,
        seed=100,
    )

    model_drift = evaluate_model_drift(
        ref,
        cur_drifted,
        model,
        output_dir=str(tmp_path),
        tag="concept_test",
        f1_drop_threshold=0.05,
    )
    assert model_drift["f1_drop"] >= 0.05
    assert model_drift["model_drift_detected"] is True


def test_run_drift_induction_end_to_end_writes_summary(tmp_path):
    cur_csv = tmp_path / "drifted_batch.csv"
    ref_csv = tmp_path / "ref_sample.csv"

    ref = generate_synthetic_baseline(n_rows=400, seed=1)
    X_ref, y_ref = select_xy(ref)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(X_ref, y_ref)

    import joblib

    model_path = tmp_path / "temp_model.pkl"
    joblib.dump(model, model_path)

    _, _, summary = run_drift_induction(
        input_path=str(tmp_path / "missing.csv"),
        ref_output_path=str(ref_csv),
        cur_output_path=str(cur_csv),
        mode="both",
        intensity=1.1,
        flip_rate=0.40,
        sample_rows=400,
        seed=1,
        run_monitoring=True,
        model_path=str(model_path),
        log_to_mlflow=False,
    )

    assert cur_csv.exists()
    assert summary is not None
    assert summary["dataset_drifted"] is True
    assert summary["model_drift"]["model_drift_detected"] is True

    latest = json.loads((tmp_path / "latest_drift_summary.json").read_text(encoding="utf-8"))
    assert latest["drift_share"] >= 0.30
    assert latest["model_drift"]["model_drift_detected"] is True


def test_prometheus_alert_rules_include_drift_alerts():
    rules_path = REPO_ROOT / "monitoring" / "prometheus" / "alert_rules.yml"
    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    alert_names = {
        rule["alert"]
        for group in data["groups"]
        for rule in group["rules"]
    }
    assert "HighDataDriftDetected" in alert_names
    assert "ModelPerformanceDegraded" in alert_names
    assert "APIDown" in alert_names


if __name__ == "__main__":
    import tempfile

    test_drift_induction_preserves_11_feature_contract()
    test_compute_drift_report_detects_no_drift_vs_induced_drift(Path(tempfile.mkdtemp()))
    test_evaluate_model_drift_detects_concept_degradation(Path(tempfile.mkdtemp()))
    test_run_drift_induction_end_to_end_writes_summary(Path(tempfile.mkdtemp()))
    test_prometheus_alert_rules_include_drift_alerts()
    print("ALL 5 DRIFT TESTS PASSED")