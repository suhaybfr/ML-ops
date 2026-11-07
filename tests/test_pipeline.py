"""Comprehensive Phase 2 PyTest suite for data prep, models, risk gates, registry & API.

Covers:
  1. Evaluation & risk-diagnostic helpers (`src/evaluation.py`)
  2. Real `src/prepare.py` execution on synthetic 17-column engineered data
  3. Serialized model artifacts (`baseline_model.pkl`, `logreg_baseline.pkl`, `svm_model.pkl`),
     11-feature schema enforcement, behavioural/monotonicity invariance, and latency SLAs
  4. Phase 2 Model-Risk & Quality Gate validator (`src/validate_model.py`)
  5. Phase 2 MLflow Model Registry workflow (`src/register_model.py` on local SQLite)
  6. 11-feature drift monitoring (`src/monitoring.py`)
  7. FastAPI inference endpoints (`app/main.py`) across champion RF and Pipeline models
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import (
    classification_metrics,
    confusion_counts,
    model_risk_diagnostics,
    positive_scores,
    squared_correlation,
)
from src.features import (
    ENGINEERED_FEATURES,
    EXPECTED_ENGINEERED_CSV_COLS,
    LEAKY_COLS,
    METADATA_COLS,
    TARGET_COL,
    validate_engineered_dataframe,
)
from src.monitoring import FEATURE_COLS as MONITORING_FEATURES, compute_drift_report
from src.prepare import prepare_data
from src.register_model import (
    find_champion_run,
    register_and_promote_champion,
    verify_registered_model,
)
from src.utils import log_model_and_pkl, standard_tags
from src.validate_model import DEFAULT_METRICS_PATH, validate_baseline_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _make_synthetic_engineered_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a valid 17-column synthetic engineered_features DataFrame."""
    rng = np.random.default_rng(seed)
    correct = np.array([0, 1] * (n // 2))
    rng.shuffle(correct)
    opp = rng.integers(1, 15, size=n)
    attempts = rng.integers(1, 5, size=n).astype(float)
    # Inject extreme outlier to verify attempt_count capping <= 50
    attempts[0] = 500.0
    hints = rng.integers(0, 4, size=n).astype(float)

    df = pd.DataFrame(
        {
            "user_id": rng.integers(1000, 2000, size=n),
            "problem_id": rng.integers(5000, 6000, size=n),
            "skill_name": ["Circle Graph" if i % 4 else "Unknown" for i in range(n)],
            "opportunity": opp,
            "log_opportunity": np.log1p(opp),
            "attempt_count": attempts,
            "hint_count": hints,
            "bottom_hint": (hints >= 3).astype(int),
            "hint_ratio": np.clip(hints / 3.0, 0.0, 1.0),
            "first_try_correct": ((attempts == 1) & (correct == 1) & (hints == 0)).astype(int),
            "user_prev_acc": rng.uniform(0.25, 0.95, size=n),
            "problem_difficulty": rng.uniform(0.20, 0.90, size=n),
            "skill_difficulty": rng.uniform(0.30, 0.85, size=n),
            "is_skill_unmapped": [0 if i % 4 else 1 for i in range(n)],
            "log_response_time": rng.uniform(7.5, 11.5, size=n),
            "is_rapid_guess": [1 if i % 8 == 0 else 0 for i in range(n)],
            "correct": correct,
        }
    )
    assert list(df.columns) == EXPECTED_ENGINEERED_CSV_COLS
    return df


# ---------------------------------------------------------------------------
# 1. Evaluation helper tests
# ---------------------------------------------------------------------------


class TestEvaluationHelpers:
    """Tests for `src/evaluation.py` metric & diagnostic functions."""

    def test_positive_scores_with_proba(self):
        class FakeProbaModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7], [0.6, 0.4]])

        scores = positive_scores(FakeProbaModel(), None)
        np.testing.assert_array_almost_equal(scores, [0.7, 0.4])

    def test_positive_scores_decision_function_fallback(self):
        class FakeDecisionModel:
            def decision_function(self, X):
                return np.array([1.2, -0.5])

        scores = positive_scores(FakeDecisionModel(), None)
        np.testing.assert_array_almost_equal(scores, [1.2, -0.5])

    def test_squared_correlation_perfect_and_constant(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        assert squared_correlation(y, y * 2 + 1) == pytest.approx(1.0)
        assert squared_correlation(np.array([0.0, 1.0, 0.0, 1.0]), np.ones(4)) == 0.0

    def test_classification_metrics_keys_and_perfect_score(self):
        y = np.array([1, 0, 1, 0])
        metrics = classification_metrics(y, y, y.astype(float))
        assert set(metrics.keys()) == {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "r2_score",
            "mcc",
        }
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_score"] == pytest.approx(1.0)
        assert confusion_counts(y, y) == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


# ---------------------------------------------------------------------------
# 2. Real `src/prepare.py` tests on synthetic 17-column data
# ---------------------------------------------------------------------------


class TestDataPreparation:
    """Tests real `src.prepare.prepare_data()` without copy-pasted stubs."""

    def test_prepare_data_stratified_split_leakage_drop_and_cap(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)

        df = _make_synthetic_engineered_df(n=200, seed=7)
        df.to_csv(proc_dir / "engineered_features.csv", index=False)

        params = {
            "prepare": {
                "test_size": 0.2,
                "random_state": 42,
                "sample_size": 150,
                "max_attempt_count": 50,
            }
        }
        with open(tmp_path / "params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f)

        summary = prepare_data()

        train_df = pd.read_csv(proc_dir / "train.csv")
        test_df = pd.read_csv(proc_dir / "test.csv")

        assert len(train_df) == 120
        assert len(test_df) == 30
        assert summary["train_rows"] == 120
        assert summary["test_rows"] == 30
        assert (tmp_path / "reports" / "prepare_summary.json").exists()

        # Verify exact 11 predictors + metadata + target, and ZERO leaky columns
        expected_cols = METADATA_COLS + ENGINEERED_FEATURES + [TARGET_COL]
        assert list(train_df.columns) == expected_cols
        assert list(test_df.columns) == expected_cols
        for leaky in LEAKY_COLS:
            assert leaky not in train_df.columns
            assert leaky not in test_df.columns

        # Verify outlier capping (raw max was 500.0)
        assert summary["attempt_count_cap"]["raw_max_attempt_count"] == 500.0
        assert train_df["attempt_count"].max() <= 50.0
        assert test_df["attempt_count"].max() <= 50.0

    def test_schema_validation_rejects_missing_or_null_predictors(self):
        df = _make_synthetic_engineered_df(n=20)
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_engineered_dataframe(df.drop(columns=["hint_ratio"]))

        df_null = df.copy()
        df_null.loc[0, "user_prev_acc"] = np.nan
        with pytest.raises(ValueError, match="Unexpected nulls"):
            validate_engineered_dataframe(df_null)


# ---------------------------------------------------------------------------
# 3. Model artifact, 11-feature contract, invariance & latency tests
# ---------------------------------------------------------------------------


class TestModelArtifactsAndInvariants:
    """Verify champion Random Forest and comparison models on the real 11-feature contract."""

    @pytest.mark.skipif(
        not (MODELS_DIR / "baseline_model.pkl").exists(),
        reason="models/baseline_model.pkl not materialized — run: dvc pull",
    )
    def test_champion_random_forest_contract_invariance_and_latency(
        self, mastery_payload, struggling_payload
    ):
        rf = joblib.load(MODELS_DIR / "baseline_model.pkl")
        assert isinstance(rf, RandomForestClassifier)
        assert list(rf.feature_names_in_) == list(ENGINEERED_FEATURES)

        X_pair = pd.DataFrame([mastery_payload, struggling_payload])[ENGINEERED_FEATURES]
        preds = rf.predict(X_pair)
        probs = rf.predict_proba(X_pair)

        # Calibrated probability bounds & sum-to-one invariant
        assert probs.shape == (2, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)

        # Behavioural directional invariance: mastery student > struggling student
        assert int(preds[0]) == 1
        assert int(preds[1]) == 0
        assert float(probs[0, 1]) > 0.80
        assert float(probs[1, 1]) < 0.25

        # Determinism invariance
        probs_repeat = rf.predict_proba(X_pair)
        np.testing.assert_allclose(probs, probs_repeat, atol=1e-12)

        # Single-row warm latency SLA (< 100 ms)
        _ = rf.predict_proba(X_pair.iloc[:1])
        timings_ms = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = rf.predict_proba(X_pair.iloc[:1])
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        assert float(np.median(timings_ms)) < 100.0

    @pytest.mark.skipif(
        not (MODELS_DIR / "svm_model.pkl").exists(),
        reason="models/svm_model.pkl not materialized — run: dvc pull",
    )
    def test_svm_and_logreg_pipelines_enforce_11_features(
        self, mastery_payload, struggling_payload
    ):
        X_pair = pd.DataFrame([mastery_payload, struggling_payload])[ENGINEERED_FEATURES]

        for fname in ("svm_model.pkl", "logreg_baseline.pkl"):
            path = MODELS_DIR / fname
            if not path.exists():
                continue
            model = joblib.load(path)
            assert isinstance(model, Pipeline)
            preds = model.predict(X_pair)
            assert len(preds) == 2
            assert set(preds).issubset({0, 1})

            # Obsolete 5-column input must raise ValueError at scaler step
            bad_df = pd.DataFrame(
                [{"skill_id": 5, "attempt_count": 1, "ms_first_response": 5000, "hint_count": 0, "overlap_time": 0.0}]
            )
            with pytest.raises(ValueError, match="Feature names"):
                model.predict(bad_df)


# ---------------------------------------------------------------------------
# 4. Phase 2 Model-Risk & Quality Gate validator (`src/validate_model.py`)
# ---------------------------------------------------------------------------


class TestModelQualityGate:
    """Tests for `src/validate_model.py` positive and negative gate conditions."""

    def test_real_baseline_metrics_pass_all_gates(self):
        report = validate_baseline_metrics(
            metrics_path=DEFAULT_METRICS_PATH,
            require_artifact=(MODELS_DIR / "baseline_model.pkl").exists(),
        )
        assert report["passed"] is True
        assert report["failed_checks"] == 0

    def test_quality_gate_catches_calibration_recall_slice_and_leakage_regressions(self, tmp_path):
        with open(DEFAULT_METRICS_PATH, encoding="utf-8") as f:
            base = json.load(f)

        broken = copy.deepcopy(base)
        broken["features"].append("first_try_correct")
        rf_risk = broken["models"]["random_forest_baseline"]["risk_and_slice_diagnostics"]
        rf_risk["brier_score"] = 0.25
        rf_risk["struggling_student_class0"]["recall_specificity"] = 0.55
        rf_risk["slices"]["cold_start_opportunity_1"]["roc_auc"] = 0.72

        bad_path = tmp_path / "bad_metrics.json"
        bad_path.write_text(json.dumps(broken), encoding="utf-8")

        report = validate_baseline_metrics(metrics_path=bad_path, require_artifact=False)
        assert report["passed"] is False
        failed_names = {item["check"] for item in report["failures"]}
        assert "exact_11_engineered_features" in failed_names
        assert "no_target_leakage_columns" in failed_names
        assert "calibration_brier_score" in failed_names
        assert "struggling_student_class0_recall" in failed_names
        assert "slice_cold_start_opportunity_1_roc_auc" in failed_names


# ---------------------------------------------------------------------------
# 5. Offline MLflow Model Registry workflow (`src/register_model.py`)
# ---------------------------------------------------------------------------


class TestModelRegistryWorkflow:
    """Test champion lookup, idempotency, and promotion rules on local SQLite MLflow."""

    def test_registry_ignores_drift_runs_is_idempotent_and_guards_promotion(
        self, tmp_path, monkeypatch
    ):
        db_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
        exp_name = "test_student_baseline"
        model_name = "test_student_predictor"
        mlflow.set_tracking_uri(db_uri)
        mlflow.set_experiment(exp_name)

        X_tiny = _make_synthetic_engineered_df(n=30)[ENGINEERED_FEATURES].astype(float)
        y_tiny = np.array([0, 1] * 15)
        clf = DummyClassifier(strategy="most_frequent").fit(X_tiny, y_tiny)

        # 1. Log champion Random Forest run (F1 = 0.9621)
        with mlflow.start_run(run_name="random_forest_baseline_v2") as champ_run:
            mlflow.set_tags(
                standard_tags(
                    stage="train",
                    model="random_forest",
                    feature_set="engineered_11",
                    kind="baseline_holdout",
                    extra={"registry_candidate": "true"},
                )
            )
            mlflow.log_metrics({"f1_score": 0.9621, "roc_auc": 0.9773, "brier_score": 0.0434})
            log_model_and_pkl(clf, tmp_path / "champ.pkl", X_example=X_tiny)

        # 2. Log a subsequent drift_check run with NO f1_score (simulates old CI bug)
        with mlflow.start_run(run_name="drift_check_deadbeef"):
            mlflow.log_metrics({"drift_share": 0.0})

        client = mlflow.tracking.MlflowClient(tracking_uri=db_uri)
        found = find_champion_run(client, experiment_name=exp_name)
        assert found.info.run_id == champ_run.info.run_id

        # 3. Register champion -> v1 promoted
        res1 = register_and_promote_champion(
            model_name=model_name,
            experiment_name=exp_name,
            tracking_uri=db_uri,
            enforce_gate=False,
        )
        assert res1["version"] == "1"
        assert res1["created_new_version"] is True
        assert res1["promoted"] is True

        # 4. Re-running on same run_id is idempotent (reuses v1)
        res_idem = register_and_promote_champion(
            model_name=model_name,
            run_id=champ_run.info.run_id,
            experiment_name=exp_name,
            tracking_uri=db_uri,
            enforce_gate=False,
        )
        assert res_idem["version"] == "1"
        assert res_idem["created_new_version"] is False
        assert res_idem["promoted"] is True

        # 5. Log weaker challenger run (F1 = 0.9100 < 0.9621) -> NOT promoted!
        with mlflow.start_run(run_name="weaker_challenger") as weak_run:
            mlflow.set_tags(
                standard_tags(
                    stage="train",
                    model="random_forest",
                    feature_set="engineered_11",
                    kind="challenger",
                )
            )
            mlflow.log_metrics({"f1_score": 0.9100, "roc_auc": 0.9300, "brier_score": 0.0750})
            log_model_and_pkl(clf, tmp_path / "weak.pkl", X_example=X_tiny)

        res_weak = register_and_promote_champion(
            model_name=model_name,
            run_id=weak_run.info.run_id,
            experiment_name=exp_name,
            tracking_uri=db_uri,
            enforce_gate=False,
        )
        assert res_weak["version"] == "2"
        assert res_weak["promoted"] is False

        # Active verified champion must still have F1 = 0.9621 (v1)
        verified = verify_registered_model(
            model_name=model_name,
            experiment_name=exp_name,
            tracking_uri=db_uri,
            min_f1=0.94,
        )
        assert verified["version"] == "1"
        assert verified["f1_score"] == pytest.approx(0.9621)


# ---------------------------------------------------------------------------
# 6. 11-Feature Drift Monitoring (`src/monitoring.py`)
# ---------------------------------------------------------------------------


class TestDriftMonitoring:
    def test_monitoring_uses_all_11_engineered_features(self, tmp_path):
        assert list(MONITORING_FEATURES) == list(ENGINEERED_FEATURES)
        ref = _make_synthetic_engineered_df(n=120, seed=1)
        cur = _make_synthetic_engineered_df(n=120, seed=2)
        summary = compute_drift_report(
            ref, cur, output_dir=str(tmp_path / "drift"), tag="unit", log_to_mlflow=False
        )
        assert summary["evaluated_features"] == list(ENGINEERED_FEATURES)
        assert len(summary["feature_drift"]) == 11
        assert Path(summary["html_path"]).exists()
        assert Path(summary["json_path"]).exists()


# ---------------------------------------------------------------------------
# 7. FastAPI Inference Service (`app/main.py`)
# ---------------------------------------------------------------------------


class TestFastAPIEndpoints:
    """Integration & boundary tests for the FastAPI inference service."""

    @staticmethod
    def _ensure_model_or_mock(monkeypatch):
        import app.main as app_main

        if Path(app_main.MODEL_PATH).exists():
            return

        class FallbackRF:
            feature_names_in_ = np.array(ENGINEERED_FEATURES)
            n_estimators = 100
            max_depth = 10
            random_state = 42

            def predict(self, X):
                probs = self.predict_proba(X)[:, 1]
                return (probs >= 0.5).astype(int)

            def predict_proba(self, X):
                p1 = np.where(X["attempt_count"].to_numpy() <= 1.0, 0.92, 0.08)
                return np.column_stack([1.0 - p1, p1])

        monkeypatch.setattr(app_main, "_model", FallbackRF())
        monkeypatch.setattr(app_main, "_model_version", "random_forest_baseline_v2")
        monkeypatch.setattr(app_main, "_model_source", "mock_fallback")
        monkeypatch.setattr(
            app_main,
            "_loaded_cache_key",
            (app_main.MLFLOW_MODEL_URI, str(Path(app_main.MODEL_PATH).resolve())),
        )

    def test_root_health_and_metrics_endpoints(self, client):
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert r_root.json()["feature_count"] == 11

        r_health = client.get("/health")
        assert r_health.status_code == 200
        h = r_health.json()
        assert h["status"] == "ok"
        if (MODELS_DIR / "baseline_model.pkl").exists():
            assert h["model_loaded"] is True
            assert h["model_version"] == "random_forest_baseline_v2"

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "prediction_requests" in r_metrics.text or "prometheus_client" in r_metrics.text

    def test_model_info_handles_random_forest_and_pipeline_models(self, client, monkeypatch):
        import app.main as app_main

        self._ensure_model_or_mock(monkeypatch)
        resp = client.get("/model/info")
        assert resp.status_code == 200
        info = resp.json()
        assert info["feature_columns"] == list(ENGINEERED_FEATURES)
        assert info["feature_count"] == 11
        assert info["supports_predict_proba"] is True

        if (MODELS_DIR / "baseline_model.pkl").exists():
            assert info["model_type"] == "RandomForestClassifier"

        # Also verify Pipeline models (svm_model.pkl & logreg_baseline.pkl) never crash /model/info
        for pkl_name, expected_type in (
            ("svm_model.pkl", "SVC"),
            ("logreg_baseline.pkl", "LogisticRegression"),
        ):
            p = MODELS_DIR / pkl_name
            if p.exists():
                monkeypatch.setattr(app_main, "MODEL_PATH", str(p))
                app_main.reset_model_cache()
                r_pipe = client.get("/model/info")
                assert r_pipe.status_code == 200
                p_info = r_pipe.json()
                assert p_info["model_type"] == expected_type
                assert len(p_info["pipeline_steps"]) == 2

    def test_predict_single_mastery_and_struggling_students(
        self, client, monkeypatch, mastery_payload, struggling_payload
    ):
        self._ensure_model_or_mock(monkeypatch)

        r_ok = client.post("/predict", json=mastery_payload)
        assert r_ok.status_code == 200
        d_ok = r_ok.json()
        assert d_ok["prediction"] == 1
        assert 0.5 <= d_ok["probability"] <= 1.0
        assert d_ok["needs_intervention"] is False
        assert d_ok["model_version"] == "random_forest_baseline_v2"

        r_struggle = client.post("/predict", json=struggling_payload)
        assert r_struggle.status_code == 200
        d_st = r_struggle.json()
        assert d_st["prediction"] == 0
        assert 0.0 <= d_st["probability"] < 0.5
        assert d_st["needs_intervention"] is True

    def test_predict_rejects_missing_obsolete_leaky_and_out_of_bounds_inputs(
        self, client, mastery_payload
    ):
        # Missing fields -> 422
        assert client.post("/predict", json={"attempt_count": 1}).status_code == 422

        # Obsolete 5-feature payload (`skill_id`, `overlap_time`) -> 422 (never silently faked!)
        legacy_payload = {
            "skill_id": 5,
            "attempt_count": 2,
            "ms_first_response": 4500,
            "hint_count": 1,
            "overlap_time": 0.1,
        }
        assert client.post("/predict", json=legacy_payload).status_code == 422

        # Extra leaky column (`first_try_correct`) -> 422
        with_leaky = {**mastery_payload, "first_try_correct": 1}
        assert client.post("/predict", json=with_leaky).status_code == 422

        # Out-of-range probability feature (`user_prev_acc > 1.0`) -> 422
        bad_acc = {**mastery_payload, "user_prev_acc": 1.5}
        assert client.post("/predict", json=bad_acc).status_code == 422

        # Out-of-range binary flag (`bottom_hint = 2`) -> 422
        bad_flag = {**mastery_payload, "bottom_hint": 2}
        assert client.post("/predict", json=bad_flag).status_code == 422

    def test_predict_batch_valid_and_boundary_errors(
        self, client, monkeypatch, mastery_payload, struggling_payload
    ):
        self._ensure_model_or_mock(monkeypatch)

        r_batch = client.post(
            "/predict/batch", json={"instances": [mastery_payload, struggling_payload]}
        )
        assert r_batch.status_code == 200
        body = r_batch.json()
        assert body["count"] == 2
        assert body["predictions"][0]["prediction"] == 1
        assert body["predictions"][0]["needs_intervention"] is False
        assert body["predictions"][1]["prediction"] == 0
        assert body["predictions"][1]["needs_intervention"] is True

        # Empty batch -> 400
        assert client.post("/predict/batch", json={"instances": []}).status_code == 400
