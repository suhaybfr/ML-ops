import os
import sys
import configparser
import mlflow

# Ensure UTF-8 output on Windows consoles to prevent emoji encode errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_mlflow(experiment_name: str = "student_performance_prediction"):
    """
    Configure MLflow tracking URI and authentication for DAGsHub.
    Credentials are automatically loaded from .dvc/config.local (if available)
    or from standard MLFLOW environment variables.
    """
    tracking_uri = "https://dagshub.com/suhaybfr/ML-ops.mlflow"

    # Try loading credentials from .dvc/config.local if not already in env
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        dvc_local_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".dvc", "config.local")
        if os.path.exists(dvc_local_config):
            config = configparser.ConfigParser()
            config.read(dvc_local_config)
            for section in config.sections():
                if "remote" in section:
                    user = config[section].get("user")
                    password = config[section].get("password")
                    if user and password:
                        os.environ["MLFLOW_TRACKING_USERNAME"] = user
                        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
                        break

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Connected to tracking URI: {tracking_uri}")
    print(f"[MLflow] Active Experiment: {experiment_name}")
    return tracking_uri
