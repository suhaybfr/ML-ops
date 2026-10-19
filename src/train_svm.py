"""Kernel SVM training and evaluation.

Trains an RBF-kernel SVM on the processed ASSISTments split, evaluates it on
the held-out test set and logs the run to DAGsHub MLflow.
"""

import os

import joblib
import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import classification_metrics, positive_scores
from utils import setup_mlflow

EXPERIMENT_NAME = "student_performance_baseline"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def train_svm():
    params = load_params()
    svm_params = params["svm"]
    random_state = svm_params["random_state"]

    setup_mlflow(experiment_name=EXPERIMENT_NAME)

    train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    test_df = pd.read_csv(os.path.join("data", "processed", "test.csv"))

    target_col = "correct"
    feature_cols = [c for c in train_df.columns if c != target_col]

    # Kernel SVM is quadratic in the number of training rows, so cap the
    # training set while keeping the held-out test set complete.
    sample_size = svm_params.get("sample_size")
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=random_state)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(
        f"[SVM] Training {svm_params['kernel']} kernel SVM "
        f"(C={svm_params['C']}, gamma={svm_params['gamma']}) "
        f"on {len(X_train)} samples"
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_params["kernel"],
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            random_state=random_state,
        ),
    )

    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    roc_path = os.path.join("reports", "figures", "svm_roc_curve.png")
    cm_path = os.path.join("reports", "figures", "svm_confusion_matrix.png")
    model_path = os.path.join("models", "svm_model.pkl")

    with mlflow.start_run(run_name="svm_rbf_holdout") as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = positive_scores(model, X_test)
        metrics = classification_metrics(y_test, y_pred, y_score)

        for name, value in metrics.items():
            print(f"[SVM] {name:<10}: {value:.4f}")
        print(f"[SVM] Run ID: {run.info.run_id}")
        print(f"[SVM] Confusion matrix:\n{pd.crosstab(y_test, y_pred)}")

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
                "features": ",".join(feature_cols),
            }
        )
        mlflow.log_metrics(metrics)

        # Diagnostic figures logged as MLflow artifacts.
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

        mlflow.log_artifact(roc_path, artifact_path="figures")
        mlflow.log_artifact(cm_path, artifact_path="figures")

        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, name="model")
        print(f"[SVM] Saved model artifact to: {model_path}")

    return metrics


if __name__ == "__main__":
    train_svm()
