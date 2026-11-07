"""AI-Based Digital Learning Assistant - MLOps package."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
