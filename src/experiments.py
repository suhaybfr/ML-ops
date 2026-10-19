"""Model comparison across several train/test splits and k-fold CV.

Trains logistic regression, SVM and random forest on 80/20, 70/30 and 60/40
splits plus 5-fold cross-validation, writes the result tables to reports/ and
logs everything to DAGsHub MLflow.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import classification_metrics, positive_scores
from utils import setup_mlflow

EXPERIMENT_NAME = "student_performance_baseline"
METRIC_NAMES = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "r2_score", "mcc"]


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_models(random_state):
    """Return freshly initialised candidate estimators.

    A new dictionary is built per split so no fitted state leaks between runs.
    """
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        ),
    }


def load_pooled_data():
    train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    test_df = pd.read_csv(os.path.join("data", "processed", "test.csv"))
    pooled = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = [c for c in pooled.columns if c != "correct"]
    return pooled[feature_cols], pooled["correct"], feature_cols


def cap(X, y, sample_size, random_state):
    if sample_size and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state)
        y = y.loc[X.index]
    return X, y


def save_curves(curve_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y_true, y_score) in curve_data.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name, ax=axes[1])
    axes[0].set_title("ROC curves (80/20 split)")
    axes[1].set_title("Precision-Recall curves (80/20 split)")
    fig.tight_layout()
    roc_pr_path = os.path.join(out_dir, "classification_roc_pr.png")
    fig.savefig(roc_pr_path, dpi=150)
    plt.close(fig)
    return roc_pr_path


def save_f1_chart(results_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pivot = results_df.pivot(index="test_size", columns="model", values="f1_score")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_xlabel("Test split fraction")
    ax.set_ylabel("F1 score")
    ax.set_title("Hold-out F1 by train/test split and model")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    f1_path = os.path.join(out_dir, "f1_by_split.png")
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)
    return f1_path


def run_experiments():
    params = load_params()
    exp = params["experiments"]
    random_state = exp["random_state"]
    sample_size = exp["sample_size"]
    test_sizes = exp["test_sizes"]
    folds = exp["cv_folds"]

    setup_mlflow(experiment_name=EXPERIMENT_NAME)

    X_all, y_all, feature_cols = load_pooled_data()
    print(f"[Experiments] Pooled modelling frame: {X_all.shape}")

    results = []
    cv_records = []
    curve_data = {}

    with mlflow.start_run(run_name="svm_validation_suite") as parent:
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
        mlflow.set_tags({"suite": "split_and_cv_validation", "task": "student_correctness"})

        # ---- Hold-out study over multiple train/test ratios ----
        for test_size in test_sizes:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
            )
            X_tr, y_tr = cap(X_tr, y_tr, sample_size, random_state)
            print(f"[Experiments] Split test_size={test_size}: train={len(X_tr)} test={len(X_te)}")

            for name, model in build_models(random_state).items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_score = positive_scores(model, X_te)
                metrics = classification_metrics(y_te, y_pred, y_score)

                row = {
                    "test_size": test_size,
                    "model": name,
                    "train_samples": len(X_tr),
                    "test_samples": len(X_te),
                }
                row.update(metrics)
                results.append(row)

                with mlflow.start_run(run_name=f"{name}_test{int(round(test_size * 100))}", nested=True):
                    mlflow.log_params(
                        {
                            "model": name,
                            "test_size": test_size,
                            "train_samples": len(X_tr),
                            "test_samples": len(X_te),
                        }
                    )
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags({"kind": "holdout", "model": name, "test_size": str(test_size)})

                if test_size == test_sizes[0]:
                    curve_data[name] = (y_te, y_score)

                print(
                    f"    {name:<20} f1={metrics['f1_score']:.4f} "
                    f"roc_auc={metrics['roc_auc']:.4f} r2={metrics['r2_score']:.4f}"
                )

        # ---- Stratified k-fold cross-validation ----
        X_cv, y_cv = cap(X_all, y_all, sample_size, random_state)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        for name, _ in build_models(random_state).items():
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

            record = {"model": name, "folds": folds, "samples": len(X_cv)}
            for m in METRIC_NAMES:
                record[m] = float(np.mean(fold_metrics[m]))
                record[f"{m}_std"] = float(np.std(fold_metrics[m]))
            cv_records.append(record)

            with mlflow.start_run(run_name=f"cv{folds}_{name}", nested=True):
                mlflow.log_params({"model": name, "cv_folds": folds, "samples": len(X_cv)})
                mlflow.log_metrics({m: record[m] for m in METRIC_NAMES})
                mlflow.log_metrics({f"{m}_std": record[f"{m}_std"] for m in METRIC_NAMES})
                mlflow.set_tags({"kind": "cross_validation", "model": name})

            print(
                f"[CV] {name:<20} f1={record['f1_score']:.4f} "
                f"(+/-{record['f1_score_std']:.4f}) roc_auc={record['roc_auc']:.4f}"
            )

        # ---- Persist results and figures ----
        os.makedirs("reports", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join("reports", "experiment_results.csv"), index=False)

        cv_df = pd.DataFrame(cv_records)
        cv_df.to_csv(os.path.join("reports", "cv_results.csv"), index=False)

        fig_dir = os.path.join("reports", "figures")
        roc_pr_path = save_curves(curve_data, fig_dir)
        f1_path = save_f1_chart(results_df, fig_dir)
        mlflow.log_artifact(roc_pr_path, artifact_path="figures")
        mlflow.log_artifact(f1_path, artifact_path="figures")
        mlflow.log_artifact(os.path.join("reports", "experiment_results.csv"), artifact_path="results")
        mlflow.log_artifact(os.path.join("reports", "cv_results.csv"), artifact_path="results")

        print(f"[Experiments] Parent run: {parent.info.run_id}")

    return results_df, cv_df


if __name__ == "__main__":
    run_experiments()
