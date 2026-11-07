"""MLflow / DAGsHub configuration, standardized tagging, and model artifact helpers."""

from __future__ import annotations

import configparser
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DAGSHUB_MLFLOW_URI = "https://dagshub.com/suhaybfr/ML-ops.mlflow"
DEFAULT_EXPERIMENT = "student_performance_baseline"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dagshub_credentials(repo_root: Path = REPO_ROOT) -> bool:
    """Populate MLFLOW_TRACKING_USERNAME / PASSWORD from .dvc/config.local if needed."""
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return True

    dvc_local_config = repo_root / ".dvc" / "config.local"
    if dvc_local_config.exists():
        config = configparser.ConfigParser()
        config.read(dvc_local_config)
        for section in config.sections():
            if "remote" in section:
                user = config[section].get("user")
                password = config[section].get("password")
                if user and password:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()
                    return True
    return False


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    allow_local_fallback: bool = True,
) -> str:
    """Configure MLflow tracking URI for DAGsHub with clean offline fallback.

    Set ``MLFLOW_OFFLINE=1`` to force local ``mlruns/`` tracking (for example in
    sandboxed unit tests or offline development).
    """
    offline = os.environ.get("MLFLOW_OFFLINE", "0") == "1"
    has_creds = load_dagshub_credentials()

    if not offline and has_creds:
        tracking_uri = DAGSHUB_MLFLOW_URI
    elif allow_local_fallback:
        local_mlruns = REPO_ROOT / "mlruns"
        local_mlruns.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{local_mlruns / 'mlflow.db'}"
    else:
        tracking_uri = DAGSHUB_MLFLOW_URI

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking URI: {tracking_uri}")
    print(f"[MLflow] Active Experiment: {experiment_name}")
    return tracking_uri


def standard_tags(
    stage: str,
    model: str,
    feature_set: str,
    kind: str,
    extra: dict | None = None,
) -> dict[str, str]:
    """Build consistent MLflow tags across all scripts, experiments, and notebooks."""
    tags = {
        "phase": "phase_1_cleanup",
        "task": "student_correctness",
        "stage": str(stage),
        "model": str(model),
        "feature_set": str(feature_set),
        "kind": str(kind),
    }
    if extra:
        for k, v in extra.items():
            tags[str(k)] = str(v)
    return tags


def log_model_and_pkl(
    model,
    local_pkl_path: str | Path,
    X_example: pd.DataFrame | None = None,
    artifact_subdir: str = "model",
) -> None:
    """Save joblib pickle locally and upload full MLflow Model bundle + pickle to run artifacts.

    Includes an inferred ``ModelSignature`` and ``input_example`` when ``X_example`` is
    provided, and uploads via ``mlflow.log_artifacts`` so hosted MLflow servers (such as
    DAGsHub) always attach ``runs:/<run_id>/model`` directly to run artifacts for Phase 2
    Model Registry registration and FastAPI loading.
    """
    local_pkl_path = Path(local_pkl_path)
    local_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, local_pkl_path)

    mlflow.log_artifact(str(local_pkl_path), artifact_path="serialized_models")

    signature = None
    input_example = None
    if X_example is not None and len(X_example) > 0:
        input_example = X_example.head(5).copy()
        preds = model.predict(input_example)
        signature = infer_signature(input_example, preds)

    tmp_dir = Path(tempfile.mkdtemp(prefix="mlflow_sklearn_"))
    try:
        bundle_dir = tmp_dir / "sklearn_bundle"
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(bundle_dir),
            serialization_format="cloudpickle",
            signature=signature,
            input_example=input_example,
        )
        mlflow.log_artifacts(str(bundle_dir), artifact_path=artifact_subdir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
