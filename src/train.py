import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn
from utils import setup_mlflow


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["train"]


def train_model():
    params = load_params()

    # Configure MLflow tracking for DAGsHub
    setup_mlflow(experiment_name="student_performance_baseline")

    # Load preprocessed datasets
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    print(f"[Train] Loading preprocessed data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "correct"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[Train] Training {params['model_type']} on {len(X_train)} samples with features: {feature_cols}")

    with mlflow.start_run(run_name="random_forest_baseline"):
        # Model initialization
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=-1,
        )

        # Train model
        model.fit(X_train, y_train)

        # Predictions & Probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"[Train] Test Accuracy:  {acc:.4f}")
        print(f"[Train] Test Precision: {prec:.4f}")
        print(f"[Train] Test Recall:    {rec:.4f}")
        print(f"[Train] Test F1-Score:  {f1:.4f}")
        print(f"[Train] Test ROC-AUC:   {roc_auc:.4f}")

        # Log parameters
        mlflow.log_params({
            "model_type": params["model_type"],
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": params["random_state"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": ",".join(feature_cols),
        })

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
        })

        # Save model locally
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "baseline_model.pkl")
        joblib.dump(model, model_path)
        print(f"[Train] Saved local model artifact to: {model_path}")

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        print(f"[Train] Model successfully logged to MLflow on DAGsHub!")


if __name__ == "__main__":
    train_model()
