# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Phase 2 — Monitoring, Artifact Logging & CI/CD]
### Added
- **MLflow Artifact Logging:** Comprehensive logging of diagnostic figures (`figures/`), validation summaries (`validation/`), configuration files (`config/params.yaml`), and interactive Evidently drift reports (`drift_reports/`) to DAGsHub MLflow.
- **Evidently Drift Detection:** Added `src/monitoring.py` using Evidently `DataDriftPreset` and `ClassificationPreset` — generates HTML + JSON reports under `reports/drift/` and logs them to MLflow.
- **Retraining Feedback Loop:** Added `src/retrain.py` — detects drift, triggers `dvc repro train_svm`, logs drift metrics and reports to MLflow, and promotes the new model in the MLflow Model Registry only if F1 improves.
- **Prometheus & Grafana Monitoring:** Added Prometheus configuration (`monitoring/prometheus/`) and auto-provisioned Grafana dashboard (`monitoring/grafana/dashboards/mlops_dashboard.json`).
- **FastAPI Serving & AI Tutor:** Added `app/main.py` (FastAPI prediction endpoints + `/metrics`) and `app/tutor.py` (Gemini + Qdrant RAG tutor).
- **Docker Orchestration:** Added multi-stage `Dockerfile` and `docker-compose.yml` for API, Prometheus, Grafana, and Qdrant.
- **CI/CD Pipeline:** Added 7-job GitHub Actions workflow (`.github/workflows/ci.yml`) covering linting, PyTest with coverage, DVC reproduction, drift checking, automated retraining, and Docker builds.

## [Phase 1 Cleanup — Phase 2 Readiness]
### Added
- **Unified Feature Contract (`src/features.py`):** Defined the canonical 11 leakage-free engineered features (`ENGINEERED_FEATURES`) from `data/processed/engineered_features.csv` across all DVC stages (`prepare`, `train`, `train_svm`, `experiments`) while retaining metadata identifiers for subgroup slice checks.
- **Model-Risk & Slice Diagnostics (`src/evaluation.py`):** Added struggling-student (class 0) specificity/precision, macro-F1, Brier probability calibration, threshold search, and subgroup performance slices (mapped vs unmapped skills, rapid-guess `<3s` vs normal latency, cold-start `opportunity==1` vs practiced).
- **Git-Tracked Validation Files (`reports/`):** Added `reports/prepare_summary.json`, `reports/baseline_metrics.csv`, `reports/baseline_metrics.json`, `reports/svm_metrics.json`, and un-ignored `reports/experiment_results.csv`, `reports/cv_results.csv`, and `reports/figures/*.png` (`cache: false` in `dvc.yaml`).
- **Automated Phase 1 Verification Suite (`tests/test_phase1_pipeline.py`):** Added PyTest suite covering schema/leakage invariants, split integrity, model serialization & inference contracts, metric thresholds, and subgroup risk checks.

### Fixed
- **DAGsHub MLflow Model Artifacts (`src/utils.py`):** Fixed MLflow 3 compatibility so every run uploads both a full `MLmodel` bundle (`ModelSignature` + `input_example`, `cloudpickle` format) under `model/` and `.pkl` artifacts under `serialized_models/`, alongside validation JSON/CSV reports and plots, while keeping all original October MLflow runs intact.
- **Git vs DVC Tracking Conflict:** Untracked `models/baseline_model.pkl` and `reports/.gitignore` from Git so DVC manages binary `.pkl` models (`baseline_model.pkl`, `logreg_baseline.pkl`, `svm_model.pkl`) cleanly without SCM collisions.

---

### Previous Releases
- **SVM Model:** Added an RBF-kernel SVM stage (`src/train_svm.py`) that trains on scaled features, saves `models/svm_model.pkl` and logs metrics and figures to MLflow.
- **Experiment Comparison:** Added `src/experiments.py` to compare logistic regression, SVM and random forest across three train/test splits and 5-fold cross-validation.
- **Shared Metrics:** Added `src/evaluation.py` so every script scores models with the same metric functions.
- **Interim & Processed Dataset Pipelines:** Exported cleaned base dataset to `data/interim/cleaned_data.csv` and engineered feature matrix to `data/processed/engineered_features.csv` via DVC.
- **Baseline Modeling:** Added Random Forest baseline, evaluation reports, and DAGsHub remote tracking.
