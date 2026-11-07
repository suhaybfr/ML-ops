"""Automated tests for Phase 1 data schema, metrics, models, and validation files."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    METRIC_NAMES,
    classification_metrics,
    confusion_counts,
    model_risk_diagnostics,
    positive_scores,
    squared_correlation,
)
from src.features import (
    ENGINEERED_FEATURES,
    LEAKY_COLS,
    TARGET_COL,
    select_xy,
    validate_engineered_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_feature_definitions_exclude_leakage():
    assert len(ENGINEERED_FEATURES) == 11
    for leaky in LEAKY_COLS + [TARGET_COL, "user_id", "problem_id", "skill_name"]:
        assert leaky not in ENGINEERED_FEATURES


def test_evaluation_metrics_and_scale_invariant_r2():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    # Uncalibrated margin score (like SVC.decision_function) on arbitrary scale
    y_margin = np.array([-4.2, 0.3, 3.8, 5.1, 2.9, -2.5, -0.4, -3.1])

    metrics = classification_metrics(y_true, y_pred, y_margin)
    assert set(metrics.keys()) == set(METRIC_NAMES)
    for val in metrics.values():
        assert np.isfinite(val)
    assert 0.0 <= metrics["r2_score"] <= 1.0

    # Scale-invariance check: multiplying margin by 100 must not change R^2
    assert pytest.approx(squared_correlation(y_true, y_margin), rel=1e-7) == squared_correlation(
        y_true, y_margin * 100.0 + 50.0
    )

    cm = confusion_counts(y_true, y_pred)
    assert cm == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}


def test_model_risk_diagnostics_slices():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "skill_name": ["Circle Graph"] * 60 + ["Unknown"] * 60,
            "is_rapid_guess": [0] * 80 + [1] * 40,
            "opportunity": [1] * 50 + [5] * 70,
        }
    )
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.7 + rng.uniform(0.05, 0.25, size=n), 0.01, 0.99)
    y_pred = (y_prob >= 0.5).astype(int)

    diag = model_risk_diagnostics(df, y_true, y_pred, y_prob, is_probability=True)
    assert "struggling_student_class0" in diag
    assert "brier_score" in diag
    assert "mapped_skills" in diag["slices"]
    assert "unmapped_skills" in diag["slices"]
    assert "rapid_guess_lt_3s" in diag["slices"]


def test_processed_partitions_integrity():
    train_path = REPO_ROOT / "data" / "processed" / "train.csv"
    test_path = REPO_ROOT / "data" / "processed" / "test.csv"
    if not (train_path.exists() and test_path.exists()):
        pytest.skip("Processed train/test partitions not materialized locally")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    validate_engineered_dataframe(train_df)
    validate_engineered_dataframe(test_df)

    for col in LEAKY_COLS:
        assert col not in train_df.columns
        assert col not in test_df.columns

    assert train_df["attempt_count"].max() <= 50
    assert test_df["attempt_count"].max() <= 50
    assert abs(train_df[TARGET_COL].mean() - test_df[TARGET_COL].mean()) < 0.01


def test_serialized_models_roundtrip_inference():
    test_path = REPO_ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        pytest.skip("test.csv not materialized locally")

    X_sample, _ = select_xy(pd.read_csv(test_path, nrows=25))

    for model_file in ["logreg_baseline.pkl", "baseline_model.pkl", "svm_model.pkl"]:
        p = REPO_ROOT / "models" / model_file
        assert p.exists(), f"Missing trained model artifact: {p}"
        model = joblib.load(p)
        preds = model.predict(X_sample)
        scores = positive_scores(model, X_sample)
        assert len(preds) == len(X_sample)
        assert set(np.unique(preds)).issubset({0, 1})
        assert np.all(np.isfinite(scores))


def test_validation_files_and_quality_gates():
    required_files = [
        "reports/prepare_summary.json",
        "reports/baseline_metrics.csv",
        "reports/baseline_metrics.json",
        "reports/svm_metrics.json",
        "reports/experiment_results.csv",
        "reports/cv_results.csv",
        "reports/figures/baseline_confusion_matrices.png",
        "reports/figures/baseline_feature_importance.png",
        "reports/figures/svm_roc_curve.png",
        "reports/figures/svm_confusion_matrix.png",
        "reports/figures/classification_roc_pr.png",
        "reports/figures/f1_by_split.png",
    ]
    for rel in required_files:
        fp = REPO_ROOT / rel
        assert fp.exists() and fp.stat().st_size > 0, f"Missing or empty validation artifact: {rel}"

    with open(REPO_ROOT / "reports" / "baseline_metrics.json", encoding="utf-8") as f:
        base = json.load(f)

    rf = base["models"]["random_forest_baseline"]
    lr = base["models"]["logistic_regression_baseline"]
    assert rf["metrics"]["f1_score"] >= 0.94
    assert rf["metrics"]["roc_auc"] >= 0.96
    assert rf["metrics"]["mcc"] >= 0.84
    assert rf["risk_and_slice_diagnostics"]["struggling_student_class0"]["recall_specificity"] >= 0.80

    assert lr["metrics"]["f1_score"] >= 0.90
    assert lr["metrics"]["roc_auc"] >= 0.92
    assert lr["risk_and_slice_diagnostics"]["struggling_student_class0"]["recall_specificity"] >= 0.80

    with open(REPO_ROOT / "reports" / "svm_metrics.json", encoding="utf-8") as f:
        svm = json.load(f)
    assert svm["metrics"]["f1_score"] >= 0.93
    assert svm["metrics"]["roc_auc"] >= 0.90
    assert svm["metrics"]["mcc"] >= 0.80

    exp_df = pd.read_csv(REPO_ROOT / "reports" / "experiment_results.csv")
    cv_df = pd.read_csv(REPO_ROOT / "reports" / "cv_results.csv")
    assert len(exp_df) == 9  # 3 splits x 3 models
    assert len(cv_df) == 3   # 3 models
