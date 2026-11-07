"""Stage 4 (experiments): Multi-split hold-out and stratified k-fold CV suite.

Compares Logistic Regression, RBF-kernel SVM, and Random Forest across 80/20,
70/30, and 60/40 stratified splits plus 5-fold stratified cross-validation on
the 11 engineered features. Persists Git-tracked CSV validation tables
(``reports/experiment_results.csv``, ``reports/cv_results.csv``) and figures,
and logs tagged parent + nested child runs with attached validation artifacts
to MLflow.
"""

from __future__ import annotations

import os
import sys
import tempfile
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
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation import (
    METRIC_NAMES,
    classification_metrics,
    positive_scores,
    save_validation_json,
)
from src.features import ENGINEERED_FEATURES, select_xy
from src.utils import DEFAULT_EXPERIMENT, setup_mlflow, standard_tags


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["experiments"]


def build_models(random_state: int) -> dict:
    """Return freshly initialized candidate estimators per split/fold."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def load_pooled_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    train_df = pd.read_csv(Path("data") / "processed" / "train.csv")
    test_df = pd.read_csv(Path("data") / "processed" / "test.csv")
    pooled = pd.concat([train_df, test_df], ignore_index=True)
    X, y = select_xy(pooled)
    return X, y, list(ENGINEERED_FEATURES)


def stratified_cap(
    X: pd.DataFrame, y: pd.Series, sample_size: int | None, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size and sample_size < len(X):
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=int(sample_size), random_state=random_state, stratify=y
        )
        return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def save_curves(curve_data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y_true, y_score) in curve_data.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name, ax=axes[1])
    axes[0].set_title("ROC curves (80/20 split)")
    axes[1].set_title("Precision-Recall curves (80/20 split)")
    fig.tight_layout()
    roc_pr_path = out_dir / "classification_roc_pr.png"
    fig.savefig(roc_pr_path, dpi=150)
    plt.close(fig)
    return roc_pr_path


def save_f1_chart(results_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot = results_df.pivot(index="test_size", columns="model", values="f1_score")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_xlabel("Test split fraction")
    ax.set_ylabel("F1 score")
    ax.set_title("Hold-out F1 by train/test split and model")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    f1_path = out_dir / "f1_by_split.png"
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)
    return f1_path


def _log_child_validation_json(payload: dict, filename: str) -> None:
    with tempfile.TemporaryDirectory(prefix="mlflow_val_") as tmp:
        p = Path(tmp) / filename
        save_validation_json(payload, p)
        mlflow.log_artifact(str(p), artifact_path="validation")


def run_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = load_params()
    random_state = int(exp["random_state"])
    sample_size = int(exp["sample_size"])
    test_sizes = [float(t) for t in exp["test_sizes"]]
    folds = int(exp["cv_folds"])

    setup_mlflow(experiment_name=DEFAULT_EXPERIMENT)

    X_all, y_all, feature_cols = load_pooled_data()
    print(f"[Experiments] Pooled modelling frame: {X_all.shape}")

    results = []
    cv_records = []
    curve_data = {}

    with mlflow.start_run(run_name="svm_validation_suite_v2") as parent:
        mlflow.log_params(
            {
                "test_sizes": ",".join(str(t) for t in test_sizes),
                "cv_folds": folds,
                "sample_size": sample_size,
                "random_state": random_state,
                "feature_count": len(feature_cols),
                "features": ",".join(feature_cols),
            }
        )
        mlflow.set_tags(
            standard_tags(
                stage="experiments",
                model="multi_model_suite",
                feature_set="engineered_11",
                kind="validation_suite_parent",
                extra={"suite": "split_and_cv_validation"},
            )
        )

        # ---- 1. Hold-out study over multiple train/test ratios ----
        for test_size in test_sizes:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all,
                y_all,
                test_size=test_size,
                random_state=random_state,
                stratify=y_all,
            )
            X_tr, y_tr = stratified_cap(X_tr, y_tr, sample_size, random_state)
            print(
                f"[Experiments] Split test_size={test_size}: "
                f"train={len(X_tr):,} test={len(X_te):,}"
            )

            for name, model in build_models(random_state).items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_score = positive_scores(model, X_te)
                metrics = classification_metrics(y_te, y_pred, y_score)

                row = {
                    "test_size": test_size,
                    "model": name,
                    "train_samples": int(len(X_tr)),
                    "test_samples": int(len(X_te)),
                    **metrics,
                }
                results.append(row)

                run_label = f"{name}_test{int(round(test_size * 100))}"
                with mlflow.start_run(run_name=run_label, nested=True):
                    mlflow.log_params(
                        {
                            "model": name,
                            "test_size": test_size,
                            "train_samples": len(X_tr),
                            "test_samples": len(X_te),
                            "feature_count": len(feature_cols),
                        }
                    )
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags(
                        standard_tags(
                            stage="experiments",
                            model=name,
                            feature_set="engineered_11",
                            kind="holdout",
                            extra={
                                "suite": "split_and_cv_validation",
                                "test_size": str(test_size),
                                "validation_file": "reports/experiment_results.csv",
                            },
                        )
                    )
                    _log_child_validation_json(row, f"{run_label}_metrics.json")

                if test_size == test_sizes[0]:
                    curve_data[name] = (y_te, y_score)

                print(
                    f"    {name:<20} f1={metrics['f1_score']:.4f} "
                    f"roc_auc={metrics['roc_auc']:.4f} mcc={metrics['mcc']:.4f}"
                )

        # ---- 2. Stratified k-fold cross-validation ----
        X_cv, y_cv = stratified_cap(X_all, y_all, sample_size, random_state)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        for name in build_models(random_state):
            fold_metrics = {m: [] for m in METRIC_NAMES}
            for train_idx, val_idx in skf.split(X_cv, y_cv):
                model = build_models(random_state)[name]
                X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                y_score = positive_scores(model, X_val)
                for m, v in classification_metrics(y_val, y_pred, y_score).items():
                    fold_metrics[m].append(v)

            record = {"model": name, "folds": folds, "samples": int(len(X_cv))}
            for m in METRIC_NAMES:
                record[m] = float(np.mean(fold_metrics[m]))
                record[f"{m}_std"] = float(np.std(fold_metrics[m]))
            cv_records.append(record)

            cv_label = f"cv{folds}_{name}"
            with mlflow.start_run(run_name=cv_label, nested=True):
                mlflow.log_params(
                    {
                        "model": name,
                        "cv_folds": folds,
                        "samples": len(X_cv),
                        "feature_count": len(feature_cols),
                    }
                )
                mlflow.log_metrics({m: record[m] for m in METRIC_NAMES})
                mlflow.log_metrics({f"{m}_std": record[f"{m}_std"] for m in METRIC_NAMES})
                mlflow.set_tags(
                    standard_tags(
                        stage="experiments",
                        model=name,
                        feature_set="engineered_11",
                        kind="cross_validation",
                        extra={
                            "suite": "split_and_cv_validation",
                            "validation_file": "reports/cv_results.csv",
                        },
                    )
                )
                _log_child_validation_json(record, f"{cv_label}_metrics.json")

            print(
                f"[CV] {name:<20} f1={record['f1_score']:.4f} "
                f"(+/-{record['f1_score_std']:.4f}) roc_auc={record['roc_auc']:.4f}"
            )

        # ---- 3. Persist Git-tracked validation CSVs & figures ----
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        exp_csv_path = reports_dir / "experiment_results.csv"
        cv_csv_path = reports_dir / "cv_results.csv"

        results_df = pd.DataFrame(results)
        results_df.to_csv(exp_csv_path, index=False)

        cv_df = pd.DataFrame(cv_records)
        cv_df.to_csv(cv_csv_path, index=False)

        fig_dir = reports_dir / "figures"
        roc_pr_path = save_curves(curve_data, fig_dir)
        f1_path = save_f1_chart(results_df, fig_dir)

        # Log best summary metrics on parent run too so DAGsHub table shows parent scores
        rf_20 = results_df[
            (results_df["test_size"] == test_sizes[0]) & (results_df["model"] == "random_forest")
        ].iloc[0]
        svm_20 = results_df[
            (results_df["test_size"] == test_sizes[0]) & (results_df["model"] == "svm_rbf")
        ].iloc[0]
        mlflow.log_metrics(
            {
                "best_rf_f1_80_20": float(rf_20["f1_score"]),
                "best_rf_roc_auc_80_20": float(rf_20["roc_auc"]),
                "svm_f1_80_20": float(svm_20["f1_score"]),
                "svm_roc_auc_80_20": float(svm_20["roc_auc"]),
            }
        )
        mlflow.log_artifact(str(roc_pr_path), artifact_path="figures")
        mlflow.log_artifact(str(f1_path), artifact_path="figures")
        mlflow.log_artifact(str(exp_csv_path), artifact_path="validation")
        mlflow.log_artifact(str(cv_csv_path), artifact_path="validation")

        print(f"[Experiments] Parent run logged: {parent.info.run_id}")

    return results_df, cv_df


if __name__ == "__main__":
    run_experiments()
