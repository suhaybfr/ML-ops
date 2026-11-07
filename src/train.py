"""Stage 2 (train): Baseline model training, evaluation, and MLflow artifact logging.

Trains both Phase 1 baseline models on the versioned 11 engineered features:
1. Standardized Logistic Regression (``models/logreg_baseline.pkl``)
2. Random Forest Classifier (``models/baseline_model.pkl``)

Saves Git-tracked validation files (``reports/baseline_metrics.csv``,
``reports/baseline_metrics.json``) and diagnostic plots
(``reports/figures/baseline_confusion_matrices.png``,
``reports/figures/baseline_feature_importance.png``), and logs complete models,
signatures, tags, metrics, and validation artifacts to MLflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import (
    classification_metrics,
    model_risk_diagnostics,
    positive_scores,
    save_validation_json,
)
from src.features import ENGINEERED_FEATURES, select_xy
from src.utils import DEFAULT_EXPERIMENT, log_model_and_pkl, setup_mlflow, standard_tags


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["train"]


def save_baseline_figures(
    y_test: pd.Series,
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    lr_pipeline: Pipeline,
    rf_model: RandomForestClassifier,
    feature_cols: list[str],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("Logistic Regression Baseline")
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=axes[1], colorbar=False)
    axes[1].set_title("Random Forest Baseline")
    fig.suptitle("Held-out Test Confusion Matrices (20,000 samples)", fontsize=12)
    fig.tight_layout()
    cm_path = out_dir / "baseline_confusion_matrices.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # 2. Standardized Log-Odds Weights (LR) & Gini Feature Importances (RF)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    coefs = pd.Series(
        lr_pipeline.named_steps["classifier"].coef_[0], index=feature_cols
    ).sort_values()
    colors_lr = ["#c44e52" if v < 0 else "#4c72b0" for v in coefs.values]
    axes[0].barh(coefs.index, coefs.values, color=colors_lr)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Standardized log-odds coefficient")
    axes[0].set_title("Logistic Regression feature weights")

    imps = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    axes[1].barh(imps.index, imps.values, color="#55a868")
    axes[1].set_xlabel("Gini importance")
    axes[1].set_title("Random Forest feature importances")

    fig.tight_layout()
    fi_path = out_dir / "baseline_feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)

    return cm_path, fi_path


def train_baselines() -> dict:
    params = load_params()
    seed = params["random_state"]
    lr_cfg = params["logreg"]
    rf_cfg = params["random_forest"]

    setup_mlflow(experiment_name=DEFAULT_EXPERIMENT)

    train_df = pd.read_csv(Path("data") / "processed" / "train.csv")
    test_df = pd.read_csv(Path("data") / "processed" / "test.csv")

    X_train, y_train = select_xy(train_df)
    X_test, y_test = select_xy(test_df)
    feature_cols = list(ENGINEERED_FEATURES)

    print(
        f"[Train] Training baseline models on {len(X_train):,} samples "
        f"({len(feature_cols)} engineered features), evaluating on {len(X_test):,} samples"
    )

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(lr_cfg["C"]),
                    max_iter=int(lr_cfg["max_iter"]),
                    class_weight=lr_cfg["class_weight"],
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    rf_model = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        random_state=seed,
        n_jobs=-1,
    )

    lr_pipeline.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    lr_pred = lr_pipeline.predict(X_test)
    lr_score = positive_scores(lr_pipeline, X_test)
    lr_metrics = classification_metrics(y_test, lr_pred, lr_score)
    lr_risk = model_risk_diagnostics(test_df, y_test, lr_pred, lr_score, is_probability=True)

    rf_pred = rf_model.predict(X_test)
    rf_score = positive_scores(rf_model, X_test)
    rf_metrics = classification_metrics(y_test, rf_pred, rf_score)
    rf_risk = model_risk_diagnostics(test_df, y_test, rf_pred, rf_score, is_probability=True)

    for label, m_dict in [("LogisticRegression", lr_metrics), ("RandomForest", rf_metrics)]:
        print(
            f"[Train] {label:<20} acc={m_dict['accuracy']:.4f} "
            f"f1={m_dict['f1_score']:.4f} roc_auc={m_dict['roc_auc']:.4f} mcc={m_dict['mcc']:.4f}"
        )

    # Persist Git-tracked validation CSV, JSON, and diagnostic plots BEFORE logging runs
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_path, fi_path = save_baseline_figures(
        y_test, lr_pred, rf_pred, lr_pipeline, rf_model, feature_cols, figures_dir
    )

    csv_rows = [
        {
            "model": "logistic_regression_baseline",
            "artifact_path": "models/logreg_baseline.pkl",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_cols),
            **{k: round(v, 6) for k, v in lr_metrics.items()},
        },
        {
            "model": "random_forest_baseline",
            "artifact_path": "models/baseline_model.pkl",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_cols),
            **{k: round(v, 6) for k, v in rf_metrics.items()},
        },
    ]
    csv_path = reports_dir / "baseline_metrics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    lr_coefs = {
        col: round(float(val), 6)
        for col, val in zip(feature_cols, lr_pipeline.named_steps["classifier"].coef_[0])
    }
    rf_importances = {
        col: round(float(val), 6)
        for col, val in zip(feature_cols, rf_model.feature_importances_)
    }

    json_payload = {
        "stage": "train",
        "feature_set": "engineered_11",
        "features": feature_cols,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "models": {
            "logistic_regression_baseline": {
                "artifact_path": "models/logreg_baseline.pkl",
                "hyperparameters": {
                    "C": lr_cfg["C"],
                    "max_iter": lr_cfg["max_iter"],
                    "class_weight": lr_cfg["class_weight"],
                    "solver": "lbfgs",
                    "scaler": "StandardScaler",
                    "random_state": seed,
                },
                "metrics": {k: round(v, 6) for k, v in lr_metrics.items()},
                "risk_and_slice_diagnostics": lr_risk,
                "standardized_coefficients": lr_coefs,
            },
            "random_forest_baseline": {
                "artifact_path": "models/baseline_model.pkl",
                "hyperparameters": {
                    "n_estimators": rf_cfg["n_estimators"],
                    "max_depth": rf_cfg["max_depth"],
                    "random_state": seed,
                },
                "metrics": {k: round(v, 6) for k, v in rf_metrics.items()},
                "risk_and_slice_diagnostics": rf_risk,
                "gini_feature_importances": rf_importances,
            },
        },
        "figures": [str(cm_path), str(fi_path)],
    }
    json_path = reports_dir / "baseline_metrics.json"
    save_validation_json(json_payload, json_path)

    lr_model_path = Path("models") / "logreg_baseline.pkl"
    rf_model_path = Path("models") / "baseline_model.pkl"

    # Log Logistic Regression baseline run to MLflow
    with mlflow.start_run(run_name="logistic_regression_baseline_v2") as lr_run:
        mlflow.set_tags(
            standard_tags(
                stage="train",
                model="logistic_regression",
                feature_set="engineered_11",
                kind="baseline_holdout",
                extra={"registry_candidate": "false"},
            )
        )
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
                "C": lr_cfg["C"],
                "max_iter": lr_cfg["max_iter"],
                "class_weight": str(lr_cfg["class_weight"]),
                "solver": "lbfgs",
                "random_state": seed,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_cols),
                "features": ",".join(feature_cols),
            }
        )
        mlflow.log_metrics(lr_metrics)
        mlflow.log_metrics(
            {
                "macro_f1": lr_risk["macro_f1"],
                "class0_recall": lr_risk["struggling_student_class0"]["recall_specificity"],
                "brier_score": lr_risk["brier_score"],
            }
        )
        log_model_and_pkl(lr_pipeline, lr_model_path, X_example=X_test)
        mlflow.log_artifact(str(csv_path), artifact_path="validation")
        mlflow.log_artifact(str(json_path), artifact_path="validation")
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(fi_path), artifact_path="figures")
        print(f"[Train] Logged logistic_regression_baseline (run_id={lr_run.info.run_id})")

    # Log Random Forest baseline run to MLflow
    with mlflow.start_run(run_name="random_forest_baseline_v2") as rf_run:
        mlflow.set_tags(
            standard_tags(
                stage="train",
                model="random_forest",
                feature_set="engineered_11",
                kind="baseline_holdout",
                extra={"registry_candidate": "true"},
            )
        )
        mlflow.log_params(
            {
                "model_type": "RandomForest",
                "n_estimators": rf_cfg["n_estimators"],
                "max_depth": rf_cfg["max_depth"],
                "random_state": seed,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_cols),
                "features": ",".join(feature_cols),
            }
        )
        mlflow.log_metrics(rf_metrics)
        mlflow.log_metrics(
            {
                "macro_f1": rf_risk["macro_f1"],
                "class0_recall": rf_risk["struggling_student_class0"]["recall_specificity"],
                "brier_score": rf_risk["brier_score"],
            }
        )
        log_model_and_pkl(rf_model, rf_model_path, X_example=X_test)
        mlflow.log_artifact(str(csv_path), artifact_path="validation")
        mlflow.log_artifact(str(json_path), artifact_path="validation")
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(fi_path), artifact_path="figures")
        print(f"[Train] Logged random_forest_baseline (run_id={rf_run.info.run_id})")

    return json_payload


if __name__ == "__main__":
    train_baselines()
