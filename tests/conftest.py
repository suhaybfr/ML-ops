"""
pytest conftest.py — shared fixtures and configuration.

This file is automatically loaded by pytest before any test module.
It configures sys.path so that `src/` and `app/` modules are importable without
installing the package.
"""

import sys
import os
import pytest

# ─── Make src/ and app/ importable ───────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
APP_DIR = os.path.join(ROOT_DIR, "app")

for p in [SRC_DIR, APP_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient

        app_path = os.path.join(APP_DIR, "main.py")
        if not os.path.exists(app_path):
            pytest.skip("app/main.py not found — FastAPI app not yet deployed")

        from main import app  # noqa: F401
        return TestClient(app)
    except ImportError:
        pytest.skip("fastapi or httpx not installed")
