# AI-Based Digital Learning Assistant — MLOps Project

[![MLOps CI/CD Pipeline](https://github.com/suhaybfr/ML-ops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/suhaybfr/ML-ops/actions)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blue)](https://dagshub.com/suhaybfr/ML-ops.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Versioned-9cf)](https://dagshub.com/suhaybfr/ML-ops.dvc)

## Overview

An AI-based digital learning assistant that predicts student problem-attempt correctness (`correct` in `{0, 1}`) on the **ASSISTments Skill Builder** dataset (`525,534` rows) so a downstream tutor knows when a student is struggling (`correct == 0`, `needs_intervention == true`) and requires timely instructional support.

The system implements an end-to-end **MLOps lifecycle**:
- **Phase 1 (Data, Features & Benchmarks):** DVC dataset/model versioning, unified 11-feature leakage-free contract (`src/features.py`), Logistic Regression / Random Forest / RBF-SVM training, multi-split & 5-fold cross-validation, and subgroup slice diagnostics (`src/evaluation.py`).
- **Phase 2 (Quality Gates, Model Registry, Serving, Docker & CI/CD):** Automated model-risk & calibration quality gates (`src/validate_model.py`), DAGsHub MLflow Model Registry management (`src/register_model.py`), FastAPI + Pydantic 11-feature inference service (`app/main.py`) serving the **Champion Random Forest** (`models/baseline_model.pkl`), lean multi-stage Docker containerization, and GitHub Actions CI/CD.
- **Phase 3 Scaffolding (Optional Profiles / Extensions):** Evidently/KS 11-feature drift checks (`src/monitoring.py`), drift-gated retraining (`src/retrain.py`), Prometheus + Grafana dashboards (`monitoring/`), and Gemini + LlamaIndex + Qdrant RAG tutor endpoints (`app/tutor.py`).

---

## Technology Stack

| Component | Tool |
|---|---|
| Language | Python (3.11 in Docker/CI; 3.11–3.14 supported locally) |
| Data & ML | Pandas, NumPy (`>=2.0`), scikit-learn (`RandomForestClassifier`, `SVC`, `LogisticRegression`) |
| Dataset Versioning | DVC + DAGsHub HTTP Remote (`https://dagshub.com/suhaybfr/ML-ops.dvc`) |
| Experiment Tracking & Model Registry | MLflow (`3.18.0`) on DAGsHub (`https://dagshub.com/suhaybfr/ML-ops.mlflow`) |
| Inference API | FastAPI + Pydantic v2 + Uvicorn |
| Testing & Quality Gates | PyTest, `src/validate_model.py`, Flake8 |
| Containerisation | Multi-stage `Dockerfile` + Docker Compose (`profiles` for Phase 3 add-ons) |
| CI/CD | GitHub Actions (`.github/workflows/ci.yml`) + GHCR / Docker Hub |
| Phase 3 Extensions | Prometheus, Grafana, Evidently AI, Gemini + LlamaIndex + Qdrant |

---

## Project Structure

```text
ML-ops/
├── .dvc/                               # DVC configuration (credentials in gitignored config.local)
├── .github/workflows/ci.yml            # Phase 2 CI/CD workflow (lint, gate, registry, docker, push)
├── app/
│   ├── main.py                         # FastAPI 11-feature inference service + Prometheus /metrics
│   └── tutor.py                        # Optional Phase 3 LLM/RAG tutor router (Gemini + Qdrant)
├── data/
│   ├── raw/                            # Raw 525,534-row ASSISTments CSV (.dvc pointer)
│   ├── interim/                        # Cleaned intermediate dataset (.dvc pointer)
│   └── processed/                      # 17-column engineered_features.csv & train/test splits
├── docs/
│   ├── MLOPS_FLOW.md                   # End-to-end MLOps architecture & workflow guide
│   └── MODEL_VALIDATION.md             # Validation methodology, metrics & model comparison
├── models/                             # DVC-tracked serialized models
│   ├── baseline_model.pkl              # Champion Random Forest (served by default)
│   ├── logreg_baseline.pkl             # Standardized Logistic Regression reference
│   └── svm_model.pkl                   # Scaled RBF-kernel SVM comparison model
├── monitoring/                         # Optional Phase 3 Prometheus & Grafana configs
├── notebooks/
│   ├── 01_baseline_model.ipynb
│   └── 02_svm_model_validation.ipynb
├── reports/
│   ├── baseline_metrics.csv            # Git-tracked baseline summary table
│   ├── baseline_metrics.json           # Full metrics, Brier calibration & subgroup slice checks
│   ├── svm_metrics.json                # Standalone SVM metrics & slice checks
│   ├── experiment_results.csv          # 80/20, 70/30, 60/40 hold-out comparison table
│   ├── cv_results.csv                  # Stratified 5-fold cross-validation table
│   ├── prepare_summary.json            # Partition & outlier-cap summary
│   └── figures/                        # Confusion matrices, ROC/PR curves, feature importances
├── src/
│   ├── features.py                     # Canonical 11-feature schema & leakage guardrails
│   ├── prepare.py                      # Stratified 80k/20k split & attempt_count capping
│   ├── train.py                        # Logistic Regression & Random Forest training + MLflow
│   ├── train_svm.py                    # Scaled RBF-SVM training + MLflow
│   ├── experiments.py                  # Multi-split & 5-fold CV comparison suite
│   ├── evaluation.py                   # Shared 8 metrics, Brier calibration & slice diagnostics
│   ├── validate_model.py               # Phase 2 Model-Risk, Calibration & Latency Quality Gate
│   ├── register_model.py               # Phase 2 MLflow Model Registry registration & verification
│   ├── monitoring.py                   # Phase 3 11-feature drift detection (Evidently / KS)
│   ├── retrain.py                      # Phase 3 drift-gated retraining & champion promotion
│   └── utils.py                        # DAGsHub MLflow setup, offline fallback & MLmodel upload
├── tests/
│   ├── conftest.py                     # Shared FastAPI TestClient & 11-feature fixtures
│   ├── test_phase1_pipeline.py         # Phase 1 schema, partition & artifact tests
│   └── test_pipeline.py                # Phase 2 prep, invariance, gate, registry & API tests
├── Dockerfile                          # Multi-stage slim FastAPI runtime image
├── docker-compose.yml                  # Phase 2 API default + optional Phase 3 profiles
├── dvc.yaml                            # Reproducible DVC pipeline stages
├── params.yaml                         # Pipeline & model hyperparameters
├── requirements.txt                    # Full dev, training & test dependencies (Python 3.11–3.14)
└── requirements-serving.txt            # Lean runtime dependencies for Docker serving
```

---

## Quick Start

### 1. Install dependencies & pull DVC artifacts

```bash
git clone https://github.com/suhaybfr/ML-ops.git
cd ML-ops
pip install -r requirements.txt
dvc pull
```

### 2. Run Model Quality Gate, Registry Check & Tests

```bash
python -m src.validate_model --require-artifact
python -m src.register_model --verify-only
pytest -v
```

### 3. Run the FastAPI Service (Local or Docker)

```bash
# Local Uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Containerized Phase 2 API service (default)
docker compose up --build

# Optional Phase 3 profiles (Prometheus + Grafana + Qdrant)
docker compose --profile monitoring up --build -d
docker compose --profile full up --build -d
```

---

## FastAPI Inference Service (`app/main.py`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service overview and endpoint directory |
| `GET` | `/health` | Liveness and model-readiness probe |
| `GET` | `/model/info` | Deployed model metadata, hyperparameters & 11-feature contract |
| `GET` | `/metrics` | Prometheus metrics exposition |
| `POST` | `/predict` | Single student-attempt prediction (`11` engineered features) |
| `POST` | `/predict/batch` | Vectorized batch predictions (up to `1,000` attempts) |

### Canonical 11-Feature Request & Response (`POST /predict`)

All predictions strictly validate the 11 leakage-free `ENGINEERED_FEATURES` defined in `src/features.py` (unknown, missing, or leaky fields such as `first_try_correct` return HTTP `422`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "log_opportunity": 2.3979,
    "attempt_count": 1.0,
    "hint_count": 0.0,
    "bottom_hint": 0,
    "hint_ratio": 0.0,
    "user_prev_acc": 0.78,
    "problem_difficulty": 0.71,
    "skill_difficulty": 0.68,
    "is_skill_unmapped": 0,
    "log_response_time": 9.62,
    "is_rapid_guess": 0
  }'
```

```json
{
  "prediction": 1,
  "probability": 0.9476,
  "needs_intervention": false,
  "confidence": 0.9476,
  "label": "Correct",
  "model_version": "random_forest_baseline_v2"
}
```

---

## Validated Benchmark Results & Model Registry

All models are trained and evaluated on the unified 11 engineered features (`log_opportunity`, `attempt_count`, `hint_count`, `bottom_hint`, `hint_ratio`, `user_prev_acc`, `problem_difficulty`, `skill_difficulty`, `is_skill_unmapped`, `log_response_time`, `is_rapid_guess`) and logged to the `student_performance_baseline` experiment on DAGsHub MLflow.

### Full-Split Hold-Out Evaluation (`80,000` train / `20,000` test)

| Model | Artifact | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | Class-0 Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | `models/baseline_model.pkl` | **0.9468** | **0.9305** | **0.9960** | **0.9621** | **0.9362** | **0.9773** | **0.9876** | **0.8785** | **0.0434** | **0.8423** |
| **RBF-kernel SVM** (`30k` train cap) | `models/svm_model.pkl` | 0.9288 | 0.9107 | 0.9924 | 0.9498 | 0.9135 | 0.9313 | 0.9183 | 0.8369 | — | 0.7938 |
| **Logistic Regression** (`balanced`) | `models/logreg_baseline.pkl` | 0.8917 | 0.9188 | 0.9221 | 0.9204 | 0.8754 | 0.9404 | 0.9614 | 0.7509 | 0.0821 | 0.8272 |

### 30k-Capped Multi-Split (`80/20`) & 5-Fold Cross-Validation Suite (`src/experiments.py`)

| Model | 80/20 Accuracy | 80/20 F1 | 80/20 ROC-AUC | 80/20 MCC | 5-Fold CV F1 | 5-Fold CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.9435** | **0.9598** | **0.9742** | **0.8707** | **0.9611 ± 0.0016** | **0.9759 ± 0.0018** |
| **RBF-kernel SVM** | 0.9259 | 0.9478 | 0.9259 | 0.8300 | 0.9491 ± 0.0014 | 0.9291 ± 0.0028 |
| **Logistic Regression** | 0.9085 | 0.9353 | 0.9338 | 0.7869 | 0.9378 ± 0.0021 | 0.9375 ± 0.0031 |

### MLflow Model Registry (`src/register_model.py`)
- **Registered Model:** `student_performance_predictor` (alias `@champion`)
- **Source Run:** `random_forest_baseline_v2` (`160e5351896f46e6b359316808177e86`, `runs:/160e5351896f46e6b359316808177e86/model`)
- Register or verify via CLI:
  ```bash
  python -m src.register_model
  python -m src.register_model --verify-only
  ```

---

## CI/CD Pipeline (`.github/workflows/ci.yml`)

```text
lint-and-test (flake8 + offline quality gate + PyTest)
    └── model-gate-and-registry (DVC pull + live artifact latency/risk gate + full PyTest + MLflow Registry check)
            ├── docker-build (slim multi-stage build + real 11-feature /predict smoke assertions)
            │       └── docker-push (GHCR + optional Docker Hub on main)
            └── metrics-comment (PR champion summary)
```

### Required GitHub Actions Secrets

| Secret | Purpose |
|---|---|
| `DAGSHUB_USERNAME` | DVC remote & MLflow authentication |
| `DAGSHUB_PASSWORD` | DAGsHub access token |
| `MLFLOW_TRACKING_USERNAME` | Optional override (falls back to `DAGSHUB_USERNAME`) |
| `MLFLOW_TRACKING_PASSWORD` | Optional override (falls back to `DAGSHUB_PASSWORD`) |
| `DOCKER_USERNAME` / `DOCKER_PASSWORD` | Optional Docker Hub push (GHCR uses built-in `GITHUB_TOKEN`) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/baseline_model.pkl` | Path to local DVC-tracked model artifact |
| `MLFLOW_MODEL_URI` | _(unset)_ | Optional MLflow Registry URI (e.g. `models:/student_performance_predictor@champion`) with local fallback |
| `MLFLOW_MODEL_NAME` | `student_performance_predictor` | Registered model name on DAGsHub MLflow |
| `GOOGLE_API_KEY` | — | Optional Gemini API key for Phase 3 `/tutor/*` endpoints |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Optional Qdrant host/port for Phase 3 RAG |

---

## Contributors

* **Suhayb** ([@suhaybfr](https://github.com/suhaybfr)) — Project Lead, Modelling, Quality Gates, Registry & Serving
* **Monica** ([@monicaguru10](https://github.com/monicaguru10)) — CI/CD Pipeline & Monitoring Scaffolding
* **Neetu** ([@aekaboteneetu](https://github.com/aekaboteneetu))
* **Akhilesh** ([@akhileshchandaluri](https://github.com/akhileshchandaluri))
