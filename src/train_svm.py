"""Stage 3 (train_svm): RBF-kernel SVM training, evaluation, and MLflow logging.

Trains a scaled RBF-kernel SVM on the 11 engineered features, evaluates on the
full held-out test set, persists ``models/svm_model.pkl``,
``reports/svm_metrics.json``, and diagnostic figures, and logs the complete
model bundle, signature, tags, metrics, and validation files to MLflow.
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
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation import (
    classification_metrics,
    model_risk_diagnostics,
    positive_scores,
    save_validation_json,
)
from src.features import ENGINEERED_FEATURES, TARGET_COL, select_xy
from src.utils import DEFAULT_EXPERIMENT, log_model_and_pkl, setup_mlflow, standard_tags


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["svm"]


def train_svm() -> dict:
    svm_params = load_params()
    random_state = int(svm_params["random_state"])

    setup_mlflow(experiment_name=DEFAULT_EXPERIMENT)

    train_df = pd.read_csv(Path("data") / "processed" / "train.csv")
    test_df = pd.read_csv(Path("data") / "processed" / "test.csv")

    # Stratified subsample of the training split for quadratic kernel SVM cost;
    # the held-out 20,000-row test split remains identical to `src/train.py`.
    sample_size = svm_params.get("sample_size")
    if sample_size and sample_size < len(train_df):
        train_df, _ = train_test_split(
            train_df,
            train_size=int(sample_size),
            random_state=random_state,
            stratify=train_df[TARGET_COL],
        )
        train_df = train_df.reset_index(drop=True)

    X_train, y_train = select_xy(train_df)
    X_test, y_test = select_xy(test_df)
    feature_cols = list(ENGINEERED_FEATURES)

    print(
        f"[SVM] Training {svm_params['kernel']} kernel SVM "
        f"(C={svm_params['C']}, gamma={svm_params['gamma']}) "
        f"on {len(X_train):,} samples ({len(feature_cols)} features)"
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel=svm_params["kernel"],
                    C=float(svm_params["C"]),
                    gamma=svm_params["gamma"],
                    random_state=random_state,
                ),
            ),
        ]
    )

    figures_dir = Path("reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / "svm_roc_curve.png"
    cm_path = figures_dir / "svm_confusion_matrix.png"
    json_path = Path("reports") / "svm_metrics.json"
    model_path = Path("models") / "svm_model.pkl"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)
    metrics = classification_metrics(y_test, y_pred, y_score)
    risk = model_risk_diagnostics(test_df, y_test, y_pred, y_score, is_probability=False)

    for name, value in metrics.items():
        print(f"[SVM] {name:<10}: {value:.4f}")

    RocCurveDisplay.from_predictions(y_test, y_score, name="SVM (RBF)")
    plt.title("SVM ROC curve - held-out test set")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("SVM confusion matrix - held-out test set")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    payload = {
        "stage": "train_svm",
        "model": "svm_rbf_holdout",
        "artifact_path": str(model_path),
        "feature_set": "engineered_11",
        "features": feature_cols,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "hyperparameters": {
            "kernel": svm_params["kernel"],
            "C": svm_params["C"],
            "gamma": str(svm_params["gamma"]),
            "scaler": "StandardScaler",
            "random_state": random_state,
        },
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "risk_and_slice_diagnostics": risk,
        "figures": [str(roc_path), str(cm_path)],
    }
    save_validation_json(payload, json_path)

    with mlflow.start_run(run_name="svm_rbf_holdout_v2") as run:
        mlflow.set_tags(
            standard_tags(
                stage="train_svm",
                model="svm_rbf",
                feature_set="engineered_11",
                kind="svm_holdout",
                extra={"registry_candidate": "false"},
            )
        )
        mlflow.log_params(
            {
                "model_type": "SVM",
                "kernel": svm_params["kernel"],
                "C": svm_params["C"],
                "gamma": str(svm_params["gamma"]),
                "scaler": "StandardScaler",
                "random_state": random_state,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_cols),
                "features": ",".join(feature_cols),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {
                "macro_f1": risk["macro_f1"],
                "class0_recall": risk["struggling_student_class0"]["recall_specificity"],
            }
        )
        log_model_and_pkl(model, model_path, X_example=X_test)
        mlflow.log_artifact(str(json_path), artifact_path="validation")
        mlflow.log_artifact(str(roc_path), artifact_path="figures")
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        print(f"[SVM] Logged svm_rbf_holdout (run_id={run.info.run_id})")

    return payload


if __name__ == "__main__":
    train_svm()
