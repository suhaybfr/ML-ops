# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Model Registry, Quality Gates & FastAPI Serving Alignment]
### Added
- **Model-Risk & Quality Gate (`src/validate_model.py`):** Standalone validator enforcing 42 checks across the 11-feature schema, leakage exclusion, classification floors (Accuracy, F1, ROC-AUC, MCC, Macro-F1), Brier probability calibration (`<= 0.08`), struggling-student (`class 0`) recall (`>= 0.80`), all 6 subgroup slices, and live `models/baseline_model.pkl` inference latency (`< 100 ms`).
- **MLflow Model Registry Manager (`src/register_model.py`):** Idempotent registration, `@champion` alias promotion, and `--verify-only` CI check for `student_performance_predictor` (`random_forest_baseline_v2`, `v1`) on DAGsHub MLflow without deprecated stage-transition calls.
- **Lean Serving Requirements (`requirements-serving.txt`):** Dedicated production dependency list used by `Dockerfile` to exclude Jupyter, DVC, plotting, and test bloat from runtime containers.

### Changed & Fixed
- **FastAPI 11-Feature Service (`app/main.py`):** Switched default `MODEL_PATH` to `models/baseline_model.pkl` (Champion Random Forest) with optional `MLFLOW_MODEL_URI` loading, replaced the obsolete 5-field stub and hardcoded feature fakes with strict validation over all 11 `ENGINEERED_FEATURES`, returned calibrated `probability` (`0–1`) and `needs_intervention` (`bool`), and fixed `GET /model/info` so both standalone estimators and `Pipeline` models work cleanly.
- **Comprehensive Real-Model PyTest Suite (`tests/test_pipeline.py`, `tests/conftest.py`, `setup.cfg`):** Removed duplicate `[tool:pytest]` in `setup.cfg`, replaced the 5-column stub with real `src.prepare.prepare_data()` tests, and added 11-feature artifact, monotonicity invariance, latency SLA, quality-gate, registry, drift, and FastAPI endpoint tests.
- **Docker & Compose (`Dockerfile`, `docker-compose.yml`, `.dockerignore`):** Pointed `MODEL_PATH` to `/app/models/baseline_model.pkl`, fixed the duplicate `/etc/grafana/provisioning/dashboards` volume mount collision, and placed Phase 3 services (`prometheus`, `grafana`, `qdrant`) behind optional Compose `profiles` (`monitoring`, `tutor`, `full`).
- **Focused Phase 2 CI/CD (`.github/workflows/ci.yml`):** Removed push-triggered `drift-check` / `retrain-if-drift` jobs (which polluted `student_performance_baseline` with `drift_check_*` runs) and wasteful `docker-image.tar.gz` artifact transfers; added real-model DVC quality gate, MLflow Registry verification, and strict 11-feature container `/predict` assertions.
- **Monitoring & Retraining Fixes (`src/monitoring.py`, `src/retrain.py`):** Updated `FEATURE_COLS` to all 11 `ENGINEERED_FEATURES`, isolated optional drift runs to `student_performance_monitoring`, defaulted retraining to `train` (Random Forest), filtered runs by explicit tags, and fixed `promoted` reporting.
- **Cross-Python Dependencies & Benchmarks (`requirements.txt`, `README.md`):** Replaced `pandas<2.2.0` (`numpy<2` conflict) with `numpy>=2.0.0` + PEP 508 markers for Python 3.11–3.14 compatibility, and restored exact 11-feature benchmark tables in `README.md`.

---

## [Phase 2 Scaffolding — Monitoring, Artifact Logging & CI/CD]
### Added
- Initial scaffolding for FastAPI (`app/main.py`, `app/tutor.py`), Prometheus/Grafana (`monitoring/`), drift & retraining scripts (`src/monitoring.py`, `src/retrain.py`), multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow.

## [Phase 1 Cleanup — Phase 2 Readiness]
### Added
- **Unified Feature Contract (`src/features.py`):** Defined the canonical 11 leakage-free engineered features (`ENGINEERED_FEATURES`) from `data/processed/engineered_features.csv` across all DVC stages (`prepare`, `train`, `train_svm`, `experiments`).
- **Model-Risk & Slice Diagnostics (`src/evaluation.py`):** Added struggling-student (class 0) specificity/precision, macro-F1, Brier probability calibration, threshold search, and subgroup performance slices.
- **Git-Tracked Validation Files (`reports/`) & Phase 1 Tests (`tests/test_phase1_pipeline.py`).**
- **DAGsHub MLflow `MLmodel` Bundles (`src/utils.py`).**
