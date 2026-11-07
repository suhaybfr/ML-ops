"""
pytest conftest.py — shared fixtures and configuration.

This file is automatically loaded by pytest before any test module.
It configures sys.path so that `src/` modules are importable without
installing the package.
"""

import sys
import os

# ─── Make src/ importable ────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
APP_DIR = os.path.join(ROOT_DIR, "app")

for p in [SRC_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
