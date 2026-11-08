# MLOps Flow and Project Guide

This guide documents the Phase 1 modelling pipeline and Phase 2 validation, registry, serving, containerisation, and CI/CD architecture in the AI-Based Digital Learning Assistant repository.

## End-to-End Flow

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
   |      | Logistic Regression + Champion Random Forest baselines
   |      +--> models/logreg_baseline.pkl, models/baseline_model.pkl (DVC cached)
   |      +--> reports/baseline_metrics.csv, reports/baseline_metrics.json (Git-tracked)
   |      +--> DAGsHub MLflow runs (`random_forest_baseline_v2` with `MLmodel` + signature)
   |
   +--> python -m src.train_svm  &  python -m src.experiments
   |      +--> models/svm_model.pkl, reports/svm_metrics.json, reports/experiment_results.csv, cv_results.csv
   |
   v
Phase 2 Quality Gate, Registry, Serving & CI/CD
   +--> python -m src.validate_model --require-artifact
   |      | Enforces 42 schema, metric, Brier calibration, class-0 recall, slice & latency gates
   +--> python -m src.register_model
   |      | Registers `student_performance_predictor` (`@champion`) on DAGsHub Model Registry
   +--> FastAPI (`app/main.py`) + Multi-stage `Dockerfile` (`requirements-serving.txt`)
          | Serves `models/baseline_model.pkl` on strict 11-feature Pydantic contract
          +--> Verified & deployed via `.github/workflows/ci.yml` -> GHCR / Docker Hub
```

## Tools and Separation of Concerns

### Git and GitHub
Git versions source code (`src/`, `app/`), automated tests (`tests/`), pipeline configuration (`dvc.yaml`, `params.yaml`, `dvc.lock`), container/CI manifests (`Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`), documentation, and lightweight validation artifacts (`reports/*.json`, `reports/*.csv`, `reports/figures/*.png` configured with `cache: false` in `dvc.yaml`).

Large CSV datasets and `.pkl` binary model files remain excluded from Git and managed exclusively by DVC.

### DVC and DAGsHub Storage
DVC versions large datasets and fitted `.pkl` models (`data/raw/skill_builder_data.csv`, `data/interim/cleaned_data.csv`, `data/processed/engineered_features.csv`, `data/processed/train.csv`, `data/processed/test.csv`, `models/baseline_model.pkl`, `models/logreg_baseline.pkl`, `models/svm_model.pkl`) against `https://dagshub.com/suhaybfr/ML-ops.dvc`. Credentials remain strictly in gitignored `.dvc/config.local`.

### MLflow Tracking & Model Registry on DAGsHub
- **Experiment Tracking (`src/utils.py`):** Logs tagged runs, 8 classification metrics + `macro_f1`, `class0_recall`, `brier_score`, `MLmodel` bundles (`model/`), `.pkl` files (`serialized_models/`), and validation JSON/CSV artifacts to `student_performance_baseline`.
- **Model Registry (`src/register_model.py`):** Promotes the validated Random Forest champion (`random_forest_baseline_v2`) under `student_performance_predictor` (`@champion` alias) after passing `src/validate_model.py`.

### FastAPI Serving, Docker & Optional Phase 3 Profiles
- `app/main.py` loads `models/baseline_model.pkl` by default (with optional `MLFLOW_MODEL_URI` support), validates all 11 `ENGINEERED_FEATURES`, and returns `prediction`, calibrated `probability` (`0–1`), `needs_intervention` (`bool`), and `model_version`.
- `docker compose up --build` starts only the Phase 2 FastAPI container; Phase 3 services (`prometheus`, `grafana`, `qdrant`) are opt-in via `--profile monitoring`, `--profile tutor`, or `--profile full`.

## File Map

| Path | Role |
|---|---|
| `src/features.py` | Shared 11-feature schema, leakage guardrails, and `select_xy()` helper. |
| `src/prepare.py` | Validation, `attempt_count` outlier capping, and stratified train/test partitioning. |
| `src/train.py` | Logistic Regression & Random Forest baseline training, slice diagnostics, and MLflow logging. |
| `src/train_svm.py` | Scaled RBF-kernel SVM training, evaluation, figures, and MLflow logging. |
| `src/experiments.py` | Multi-split (`80/20`, `70/30`, `60/40`) and 5-fold CV comparison suite. |
| `src/evaluation.py` | Shared 8-metric definitions plus struggling-student (`class 0`) & subgroup slice checks. |
| `src/validate_model.py` | Phase 2 Model-Risk, calibration, subgroup slice, and artifact latency quality gate. |
| `src/register_model.py` | Phase 2 MLflow Model Registry registration, `@champion` aliasing, and CI verification. |
| `src/monitoring.py` | Phase 3 11-feature drift detection (Evidently AI with KS fallback). |
| `src/retrain.py` | Phase 3 drift-gated retraining & quality-gated champion promotion. |
| `app/main.py` | Phase 2 FastAPI inference service + `/metrics` Prometheus exposition. |
| `app/tutor.py` | Optional Phase 3 Gemini + LlamaIndex + Qdrant conversational tutor router. |
| `Dockerfile` / `requirements-serving.txt` | Lean multi-stage production container image for FastAPI serving. |
| `docker-compose.yml` | Phase 2 API default orchestration + optional Phase 3 `monitoring` / `tutor` / `full` profiles. |
| `.github/workflows/ci.yml` | 5-job Phase 2 CI/CD workflow (lint/pytest -> DVC gate/registry -> Docker smoke -> push -> PR summary). |
| `tests/` | Automated PyTest suites (`test_phase1_pipeline.py`, `test_pipeline.py`). |

## Reproducing and Verifying Locally

```bash
dvc pull
dvc status
python -m src.validate_model --require-artifact
python -m src.register_model --verify-only
pytest -v
```
