"""PyTest shared fixtures and path setup for Phase 1 & Phase 2 tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def mastery_payload() -> dict:
    """Representative 11-feature payload for a high-mastery student attempt."""
    return {
        "log_opportunity": 2.5649,
        "attempt_count": 1.0,
        "hint_count": 0.0,
        "bottom_hint": 0,
        "hint_ratio": 0.0,
        "user_prev_acc": 0.88,
        "problem_difficulty": 0.80,
        "skill_difficulty": 0.75,
        "is_skill_unmapped": 0,
        "log_response_time": 9.35,
        "is_rapid_guess": 0,
    }


@pytest.fixture
def struggling_payload() -> dict:
    """Representative 11-feature payload for a struggling student needing intervention."""
    return {
        "log_opportunity": 0.6931,
        "attempt_count": 5.0,
        "hint_count": 3.0,
        "bottom_hint": 1,
        "hint_ratio": 1.0,
        "user_prev_acc": 0.25,
        "problem_difficulty": 0.28,
        "skill_difficulty": 0.38,
        "is_skill_unmapped": 0,
        "log_response_time": 10.60,
        "is_rapid_guess": 0,
    }


@pytest.fixture
def client(monkeypatch):
    """Create a FastAPI TestClient bound to canonical ``app.main``."""
    from fastapi.testclient import TestClient

    import app.main as app_main

    # Keep alias for any test importing `main` directly
    sys.modules["main"] = app_main
    monkeypatch.setattr(app_main, "MODEL_PATH", app_main.DEFAULT_MODEL_PATH)
    monkeypatch.setattr(app_main, "MLFLOW_MODEL_URI", "")
    app_main.reset_model_cache()
    if os.path.exists(app_main.DEFAULT_MODEL_PATH):
        app_main.get_model()

    with TestClient(app_main.app) as tc:
        yield tc

    app_main.reset_model_cache()
