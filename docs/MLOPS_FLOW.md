# MLOps Flow and Project Guide

This guide explains the Phase 1 MLOps foundation implemented in the AI-Based Digital Learning Assistant repository and how the pipeline artifacts prepare the project for Phase 2 (model testing, risk/quality gates, model registry, FastAPI serving, Docker, and CI/CD).

## Current flow

```text
ASSISTments raw CSV (data/raw/skill_builder_data.csv.dvc)
   | DVC dataset versioning & cleaning / feature engineering
   v
data/interim/cleaned_data.csv.dvc  -->  data/processed/engineered_features.csv.dvc
   | (17 columns: 11 leak-free predictors + 4 entity/slice identifiers + target)
   v
python -m src.prepare
   | schema validation, outlier cap (attempt_count <= 50), stratified 80/20 split
   +--> data/processed/train.csv, data/processed/test.csv (DVC cached)
   +--> reports/prepare_summary.json (Git-tracked validation file)
   |
   +--> python -m src.train
   |      | Logistic Regression + Random Forest baselines
   |      +--> models/logreg_baseline.pkl, models/baseline_model.pkl (DVC cached)
   |      +--> reports/baseline_metrics.csv, reports/baseline_metrics.json (Git-tracked)
   |      +--> reports/figures/baseline_confusion_matrices.png, baseline_feature_importance.png
   |      +--> DAGsHub MLflow runs: tags, params, metrics, ModelSignature + MLmodel, validation files
   |
   +--> python -m src.train_svm
   |      | scaled RBF-kernel SVM (30,000-row stratified cap)
   |      +--> models/svm_model.pkl (DVC cached)
   |      +--> reports/svm_metrics.json, reports/figures/svm_*.png (Git-tracked)
   |      +--> DAGsHub MLflow run: tags, params, metrics, ModelSignature + MLmodel, validation files
   |
   +--> python -m src.experiments
          | Logistic Regression vs SVM vs Random Forest across 80/20, 70/30, 60/40
          | and stratified 5-fold cross-validation
          +--> reports/experiment_results.csv, reports/cv_results.csv (Git-tracked)
          +--> reports/figures/classification_roc_pr.png, f1_by_split.png (Git-tracked)
          +--> DAGsHub MLflow parent + 12 nested runs (all tagged with validation JSON/CSV artifacts)
```

## Tools and Separation of Concerns

### Git and GitHub
Git versions source code (`src/`), tests (`tests/`), pipeline configuration (`dvc.yaml`, `params.yaml`, `dvc.lock`), documentation (`docs/`, `reports/*.md`), and **lightweight validation metrics & plots** (`reports/*.json`, `reports/*.csv`, `reports/figures/*.png` configured with `cache: false` in `dvc.yaml`). Keeping small JSON/CSV validation outputs in Git ensures every tagged experiment on DAGsHub has a matching validation file in the repository and allows automated `pytest` / CI quality gates to verify metrics without downloading large datasets.

Large CSV datasets and `.pkl` binary model files remain excluded from Git via `.gitignore` and managed exclusively by DVC.

### DVC and DAGsHub Storage
DVC versions large datasets and fitted `.pkl` models (`data/raw/skill_builder_data.csv`, `data/interim/cleaned_data.csv`, `data/processed/engineered_features.csv`, `data/processed/train.csv`, `data/processed/test.csv`, `models/baseline_model.pkl`, `models/logreg_baseline.pkl`, `models/svm_model.pkl`) against the DAGsHub HTTP remote (`https://dagshub.com/suhaybfr/ML-ops.dvc`). Credentials stay strictly in `.dvc/config.local` (gitignored).

### MLflow on DAGsHub
`src/utils.py` configures MLflow against `https://dagshub.com/suhaybfr/ML-ops.mlflow` (with an automatic local SQLite fallback when `MLFLOW_OFFLINE=1` or credentials are absent). Every run logs:
- **Standardized Tags (`standard_tags`):** `phase`, `task`, `stage`, `model`, `feature_set`, `kind`, and `registry_candidate`.
- **Parameters & Metrics:** Hyperparameters, sample counts, feature lists, all 8 shared classification metrics (`src/evaluation.py`), plus `macro_f1`, `class0_recall` (struggling-student specificity), and `brier_score`.
- **Model & Validation Artifacts (`log_model_and_pkl`):** Uploads both the raw `.pkl` (`serialized_models/`) and a full MLflow `MLmodel` bundle (`model/`) complete with `ModelSignature` and `input_example`, alongside `validation/` JSON/CSV files and `figures/` PNG plots.

## File Map

| Path | Role |
|---|---|
| `.dvc/` | DVC configuration and local metadata. |
| `.dvc/config.local` | Local remote credentials; never commit. |
| `.github/workflows/ci.yml` | 7-stage CI/CD pipeline (lint, test, DVC, drift, retrain, build, push). |
| `app/main.py` | FastAPI REST service with Prometheus metrics and prediction endpoints. |
| `app/tutor.py` | Gemini-powered conversational tutor with RAG (LlamaIndex + Qdrant). |
| `data/raw/` | Original DVC-managed input data (`skill_builder_data.csv.dvc`). |
| `data/interim/` | Cleaned intermediate dataset (`cleaned_data.csv.dvc`). |
| `data/processed/` | Processed 17-column feature matrix & train/test splits (`engineered_features.csv.dvc`). |
| `monitoring/` | Prometheus scraping/alert rules and Grafana dashboard configurations. |
| `src/features.py` | Shared 11-feature schema, leakage guardrails, and select_xy() helper. |
| `src/prepare.py` | Preprocessing, outlier capping, and stratified split stage. |
| `src/train.py` | Logistic Regression & Random Forest baseline training, slice diagnostics, and MLflow logging. |
| `src/train_svm.py` | Scaled RBF-kernel SVM training, evaluation, figures, and MLflow logging. |
| `src/experiments.py` | Multi-split (80/20, 70/30, 60/40) and 5-fold CV comparison suite. |
| `src/evaluation.py` | Shared 8-metric definitions plus struggling-student & subgroup slice checks. |
| `src/monitoring.py` | Evidently AI data drift detection and reporting. |
| `src/retrain.py` | Drift-triggered automated retraining and model registry promotion. |
| `src/utils.py` | DAGsHub MLflow auth, offline fallback, standardized run tagging, and MLmodel upload. |
| `models/` | DVC-tracked local model artifacts (`baseline_model.pkl`, `logreg_baseline.pkl`, `svm_model.pkl`). |
| `Dockerfile` | Multi-stage Docker container build. |
| `docker-compose.yml` | Full-stack orchestration (API, Prometheus, Grafana, Qdrant). |
| `dvc.yaml` | Pipeline stages, dependencies, parameters, and outputs. |
| `dvc.lock` | Locked hashes for a pipeline state. |
| `params.yaml` | Preparation, training, SVM, and experiment configuration. |
| `reports/baseline_metrics.json` | Full baseline metrics, calibration & slice checks. |
| `reports/svm_metrics.json` | Standalone SVM metrics & slice checks. |
| `reports/baseline_model_report.md` | Baseline results and architecture report. |
| `reports/model_validation_report.md` | SVM and model-comparison evaluation report. |
| `docs/MODEL_VALIDATION.md` | Model validation methods, metrics and results. |
| `notebooks/` | Exploratory and baseline modelling work. |
| `tests/` | PyTest test suite (tests/test_pipeline.py, tests/test_phase1_pipeline.py). |
| `requirements.txt` | Python dependencies. |

## Implemented versus Planned

Implemented:
- Git/GitHub workflow and 7-stage GitHub Actions CI/CD automation.
- DVC dataset tracking and DAGsHub remote storage.
- Unified 11-feature schema with leakage prevention (`src/features.py`).
- Reproducible DVC preparation and training stages (`prepare` -> `train` -> `train_svm` -> `experiments`).
- Logistic Regression, Random Forest, RBF-kernel SVM, and multi-split / 5-fold cross-validation experiment suite.
- Comprehensive MLflow experiment and artifact logging (plots, model bundles, JSON summaries).
- FastAPI REST prediction service with Prometheus metrics exposition (`/metrics`).
- Conversational LLM tutor with RAG using Google Gemini, LlamaIndex, and Qdrant.
- Evidently AI data drift detection generating HTML/JSON reports.
- Prometheus alerting rules and auto-provisioned Grafana monitoring dashboards.
- Automated retraining loop with MLflow Model Registry staging and production promotion.
- Multi-stage Docker image and Docker Compose local orchestration.
- Comprehensive test suite with PyTest.

Planned for future iteration:
- Production Kubernetes (k8s/Helm) deployment manifests if cloud cluster hosting is required.
- Integration of PostgreSQL database for persistent student interaction logging.

## Teammate Workflow

1. Change preprocessing in `src/prepare.py` or feature definitions in `src/features.py`.
2. Add relevant configuration to `params.yaml` and `dvc.yaml`.
3. Update features or model logic in `src/train.py` or `src/train_svm.py`.
4. Keep held-out test evaluation comparable to the baseline.
5. Log new parameters, metrics, and artifacts to the same MLflow experiment.
6. Run `dvc repro` and inspect the new MLflow run.
7. Update the report when a new model becomes the accepted benchmark.

## Reproducing and Verifying Locally

From the repository root, with dependencies installed and DVC data available:
```bash
dvc pull
dvc repro
pytest tests/ -v
```
