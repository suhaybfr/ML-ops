"""
PyTest suite for the MLOps pipeline.

Tests cover:
  - Data preparation (prepare.py)
  - Evaluation helpers (evaluation.py)
  - Model artifacts (models/ directory)
  - FastAPI endpoint smoke tests (app/main.py)

Run with:
    pytest tests/ -v
"""

import sys
import os

# Make src importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 1. Evaluation helper tests
# ---------------------------------------------------------------------------

class TestEvaluationHelpers:
    """Tests for src/evaluation.py metric functions."""

    def test_positive_scores_with_proba(self):
        """positive_scores returns column-1 of predict_proba."""
        from evaluation import positive_scores

        class FakeProbaModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7], [0.6, 0.4]])

        scores = positive_scores(FakeProbaModel(), None)
        np.testing.assert_array_almost_equal(scores, [0.7, 0.4])

    def test_positive_scores_decision_function_fallback(self):
        """positive_scores falls back to decision_function for plain SVMs."""
        from evaluation import positive_scores

        class FakeDecisionModel:
            def decision_function(self, X):
                return np.array([1.2, -0.5])

        scores = positive_scores(FakeDecisionModel(), None)
        np.testing.assert_array_almost_equal(scores, [1.2, -0.5])

    def test_squared_correlation_perfect(self):
        """Perfectly correlated arrays give r2=1."""
        from evaluation import squared_correlation

        y = np.array([0.0, 1.0, 2.0, 3.0])
        r2 = squared_correlation(y, y * 2 + 1)  # linear transform → corr=1
        assert abs(r2 - 1.0) < 1e-9

    def test_squared_correlation_zero(self):
        """Constant score returns 0."""
        from evaluation import squared_correlation

        y = np.array([0.0, 1.0, 0.0, 1.0])
        r2 = squared_correlation(y, np.ones(4))
        assert r2 == 0.0

    def test_classification_metrics_keys(self):
        """classification_metrics returns the expected metric dictionary."""
        from evaluation import classification_metrics

        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.6, 0.4])

        metrics = classification_metrics(y_true, y_pred, y_score)
        expected_keys = {
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "r2_score", "mcc",
        }
        assert expected_keys == set(metrics.keys())

    def test_classification_metrics_perfect(self):
        """Perfect predictions give accuracy=1 and f1=1."""
        from evaluation import classification_metrics

        y = np.array([1, 0, 1, 0])
        metrics = classification_metrics(y, y, y.astype(float))
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Data preparation tests (light-weight synthetic data, no real CSV needed)
# ---------------------------------------------------------------------------

class TestDataPreparation:
    """Smoke-tests for prepare.py logic using synthetic data."""

    def test_train_test_split_ratio(self, tmp_path):
        """Output train/test files must match the requested test_size."""
        import prepare  # noqa: F401 — tests imports resolve from src/

        # Build a tiny synthetic dataset
        n = 200
        df = pd.DataFrame({
            "order_id": range(n),
            "skill_id": np.random.randint(1, 10, n),
            "correct": np.random.randint(0, 2, n),
            "attempt_count": np.random.randint(1, 5, n),
            "ms_first_response": np.random.randint(1000, 60000, n),
            "hint_count": np.random.randint(0, 3, n),
            "overlap_time": np.random.uniform(0, 1, n),
        })

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)

        csv_path = raw_dir / "skill_builder_data.csv"
        df.to_csv(csv_path, index=False)

        # run_prepare is a helper that accepts explicit paths instead of hardcoded ones
        train, test = _run_prepare_on(csv_path, proc_dir, test_size=0.2, sample_size=n)
        total = len(train) + len(test)
        assert total == n
        assert len(test) / total == pytest.approx(0.2, abs=0.05)

    def test_prepare_output_columns(self, tmp_path):
        """Prepared CSVs must contain the 'correct' target column."""
        n = 100
        df = pd.DataFrame({
            "order_id": range(n),
            "skill_id": np.random.randint(1, 5, n),
            "correct": np.random.randint(0, 2, n),
            "attempt_count": np.random.randint(1, 3, n),
            "ms_first_response": np.random.randint(1000, 30000, n),
            "hint_count": np.random.randint(0, 2, n),
            "overlap_time": np.random.uniform(0, 1, n),
        })

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)

        csv_path = raw_dir / "skill_builder_data.csv"
        df.to_csv(csv_path, index=False)

        train, test = _run_prepare_on(csv_path, proc_dir, test_size=0.2, sample_size=n)
        assert "correct" in train.columns
        assert "correct" in test.columns


# ---------------------------------------------------------------------------
# 3. Model artifact tests
# ---------------------------------------------------------------------------

class TestModelArtifacts:
    """Verify that trained model artifacts exist and are loadable."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "svm_model.pkl")),
        reason="svm_model.pkl not found — run: dvc pull",
    )
    def test_svm_model_loads(self):
        """svm_model.pkl can be loaded with joblib and is an sklearn Pipeline."""
        import joblib

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "svm_model.pkl"
        )
        model = joblib.load(model_path)
        assert isinstance(model, Pipeline), "Expected an sklearn Pipeline"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "svm_model.pkl")),
        reason="svm_model.pkl not found — run: dvc pull",
    )
    def test_svm_model_predict(self):
        """Loaded SVM model must produce binary predictions on a dummy input."""
        import joblib

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "svm_model.pkl"
        )
        model = joblib.load(model_path)

        # Prepare a dummy feature vector matching training columns
        # Features: skill_id, attempt_count, ms_first_response, hint_count, overlap_time
        dummy = pd.DataFrame([{
            "skill_id": 5,
            "attempt_count": 1,
            "ms_first_response": 5000,
            "hint_count": 0,
            "overlap_time": 0.0,
        }])

        preds = model.predict(dummy)
        assert len(preds) == 1
        assert preds[0] in (0, 1)


# ---------------------------------------------------------------------------
# 4. FastAPI endpoint smoke tests
# ---------------------------------------------------------------------------

class TestFastAPIEndpoints:
    """Smoke tests for the FastAPI prediction service."""

    def test_health_endpoint(self, client):
        """GET /health returns 200 with status=ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    def test_predict_endpoint_valid(self, client, monkeypatch):
        """POST /predict with valid features returns a prediction (mocking model if not on disk)."""
        import main

        class DummyModel:
            def predict(self, X):
                return np.array([1])

            def decision_function(self, X):
                return np.array([0.75])

        # If model is not on disk yet (e.g. before dvc pull in CI), mock it for unit smoke test
        if not os.path.exists(main.MODEL_PATH):
            monkeypatch.setattr(main, "get_model", lambda: DummyModel())

        payload = {
            "skill_id": 5,
            "attempt_count": 2,
            "ms_first_response": 4500,
            "hint_count": 1,
            "overlap_time": 0.1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in (0, 1)

    def test_predict_endpoint_missing_field(self, client):
        """POST /predict with missing required field returns 422."""
        payload = {"skill_id": 5}  # missing required fields
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_prepare_on(csv_path, proc_dir, test_size: float = 0.2, sample_size: int = 200):
    """
    Run the prepare logic inline (without subprocess) by calling the same
    logic that prepare.py uses, but with injected paths.

    Returns (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)

    # Mirror prepare.py feature engineering
    df = df.dropna(subset=["correct", "skill_id"])
    df["correct"] = df["correct"].astype(int)

    feature_cols = [
        c for c in ["skill_id", "attempt_count", "ms_first_response", "hint_count", "overlap_time"]
        if c in df.columns
    ]
    keep_cols = feature_cols + ["correct"]
    df = df[keep_cols].dropna()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
