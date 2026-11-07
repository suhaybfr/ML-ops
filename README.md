# AI-Based Digital Learning Assistant — MLOps Project

[![MLOps CI/CD Pipeline](https://github.com/suhaybfr/ML-ops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/suhaybfr/ML-ops/actions)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://dagshub.com/suhaybfr/ML-ops.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Versioned-9cf)](https://dagshub.com/suhaybfr/ML-ops.dvc)

## Overview

An AI-based digital learning assistant that predicts student performance and provides personalized learning support.  The system combines:

- **SVM Prediction Model** — trained on the ASSISTments Skill Builder dataset, served as a REST API.
- **LLM/RAG Tutor** — Gemini-powered conversational tutor with Retrieval-Augmented Generation (LlamaIndex + Qdrant).
- **Full MLOps Pipeline** — data versioning (DVC), experiment tracking (MLflow on DagsHub), monitoring (Evidently + Prometheus + Grafana), containerisation (Docker), and CI/CD (GitHub Actions).

---

## Technology Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML Framework | scikit-learn (SVM, Random Forest, Logistic Regression) |
| Experiment Tracking | MLflow on DagsHub |
| Data Versioning | DVC |
| API | FastAPI + Uvicorn |
| LLM | Google Gemini API (`gemini-1.5-flash`) |
| Vector Store (RAG) | Qdrant |
| RAG Orchestration | LlamaIndex |
| Drift Detection | Evidently AI |
| Metrics | Prometheus (`prometheus_client`) |
| Dashboards | Grafana |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Dataset | ASSISTments Skill Builder |

---

## Project Structure

```text
ML-ops/
├── .dvc/                               # DVC configuration
├── .github/workflows/ci.yml           # CI/CD pipeline (7 jobs)
├── app/
│   ├── main.py                        # FastAPI prediction + Prometheus metrics
│   └── tutor.py                       # LLM/RAG tutor endpoints (Gemini + Qdrant)
├── data/
│   ├── raw/                           # Raw dataset (DVC-tracked)
│   ├── interim/                       # Cleaned intermediate data (DVC-tracked)
│   └── processed/                     # Engineered features & train/test splits (DVC-tracked)
├── docs/
│   ├── MLOPS_FLOW.md
│   └── MODEL_VALIDATION.md
├── models/                            # Trained model artifacts (DVC-tracked)
│   ├── baseline_model.pkl             # Random Forest baseline
│   ├── logreg_baseline.pkl            # Logistic Regression reference
│   └── svm_model.pkl                  # RBF-kernel SVM
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml             # Prometheus scrape config
│   │   └── alert_rules.yml            # Alerting rules
│   └── grafana/
│       ├── provisioning/              # Auto-provisioned datasource + dashboard
│       └── dashboards/
│           └── mlops_dashboard.json   # Grafana dashboard
├── notebooks/
│   ├── 01_baseline_model.ipynb
│   └── 02_svm_model_validation.ipynb
├── reports/
│   ├── baseline_metrics.csv           # Baseline evaluation summary table
│   ├── baseline_metrics.json          # Baseline metrics & slice checks
│   ├── svm_metrics.json               # SVM metrics & slice checks
│   ├── experiment_results.csv         # Multi-split comparison results
│   ├── cv_results.csv                 # Stratified 5-fold CV results
│   ├── prepare_summary.json           # Data split & schema summary
│   ├── drift/                         # Evidently HTML/JSON reports
│   └── figures/                       # ROC curves, confusion matrices, feature importances
├── src/
│   ├── features.py                    # Canonical 11-feature schema & guardrails
│   ├── prepare.py                     # Data preprocessing stage
│   ├── train.py                       # Logistic Regression & Random Forest baselines
│   ├── train_svm.py                   # RBF-kernel SVM (primary model)
│   ├── experiments.py                 # Multi-split model comparison
│   ├── evaluation.py                  # Shared metric & risk functions
│   ├── monitoring.py                  # Evidently drift detection
│   ├── retrain.py                     # Drift-triggered retraining loop
│   └── utils.py                       # MLflow / DagsHub setup
├── tests/
│   ├── conftest.py                    # Test configuration
│   ├── test_pipeline.py               # PyTest pipeline & endpoint suite
│   └── test_phase1_pipeline.py        # Phase 1 schema & invariant tests
├── Dockerfile                         # Multi-stage Docker build
├── docker-compose.yml                 # API + Prometheus + Grafana + Qdrant
├── dvc.yaml                           # DVC pipeline stages
├── params.yaml                        # Hyperparameters
└── requirements.txt
```

---

## Quick Start

### 1 — Clone and pull data

```bash
git clone https://github.com/suhaybfr/ML-ops.git
cd ML-ops
pip install -r requirements.txt
dvc pull          # pulls data + model artifacts from DagsHub
```

### 2 — Run the full pipeline

```bash
dvc repro         # runs prepare → train → train_svm → experiments
```

### 3 — Start all services locally

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Qdrant UI | http://localhost:6333/dashboard |

---

## API Endpoints

### Prediction

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model metadata |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1000) |

**Example single prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"skill_id":7,"attempt_count":2,"ms_first_response":4500,"hint_count":1,"overlap_time":0.05}'
```

```json
{"prediction": 1, "confidence": 0.4823, "label": "✅ Correct", "model_version": "svm_rbf_v1"}
```

### LLM/RAG Tutor

> Requires `GOOGLE_API_KEY` environment variable. Optional — endpoints are skipped if Gemini is not configured.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tutor/status` | Check tutor availability |
| `POST` | `/tutor/ask` | Ask the tutor a question (with RAG) |
| `POST` | `/tutor/explain-prediction` | Explain a model prediction in plain language |
| `POST` | `/tutor/practice-questions` | Generate practice MCQs for a topic |

**Example:**
```bash
curl -X POST http://localhost:8000/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why do I keep getting penalised for hints on skill 7?","use_rag":true}'
```

**Index documents into Qdrant (one-time setup):**
```bash
PYTHONPATH=app python app/tutor.py --index docs/
```

---

## DVC Pipeline

```
prepare → train (Random Forest) → train_svm (RBF SVM) → experiments
```

```bash
dvc repro                    # reproduce all stale stages
dvc repro train_svm          # retrain SVM only
dvc dag                      # visualise stage DAG
```

**Stage outputs:**
- `data/processed/train.csv`, `test.csv`
- `models/baseline_model.pkl`, `models/svm_model.pkl`
- `reports/experiment_results.csv`, `reports/cv_results.csv`

---

## Experiment Tracking

All runs are logged to DagsHub MLflow:

- **Tracking URI:** `https://dagshub.com/suhaybfr/ML-ops.mlflow`
- **Experiment:** `student_performance_baseline`
- **Logged:** hyperparameters, metrics (accuracy, F1, ROC-AUC, MCC, PR-AUC), ROC curves, confusion matrices

**Current best results (80/20 split):**

| Model | F1 | ROC-AUC |
|---|---|---|
| Random Forest | 0.9589 | 0.9783 |
| RBF-kernel SVM | 0.9433 | ~0.97 |
| Logistic Regression | ~0.92 | ~0.95 |

---

## Monitoring

### Data Drift (Evidently)

Run manually:
```bash
PYTHONPATH=src python src/monitoring.py \
  --ref data/processed/train.csv \
  --cur data/processed/test.csv \
  --output reports/drift
```

Reports (HTML + JSON) saved to `reports/drift/`.  Also runs automatically in CI after every DVC pipeline run.

### Prometheus Metrics

The FastAPI app exposes `/metrics` in Prometheus exposition format.  Key metrics:

| Metric | Type | Description |
|---|---|---|
| `prediction_requests_total` | Counter | Total requests by endpoint + result |
| `prediction_latency_seconds` | Histogram | Request latency (p50/p95/p99) |
| `model_loaded` | Gauge | 1 if model is loaded, 0 otherwise |
| `prediction_errors_total` | Counter | Errors by endpoint + type |
| `batch_prediction_size` | Histogram | Distribution of batch sizes |

### Grafana

Visit http://localhost:3000 (admin/admin).  The **MLOps Dashboard** is auto-provisioned with panels for:
- Model status, requests/sec, latency percentiles, error rate, batch size distribution.

---

## Retraining Feedback Loop

`src/retrain.py` implements drift-triggered retraining:

1. **Drift detection** via Evidently (compares reference vs. incoming distribution)
2. **Trigger** `dvc repro train_svm` if drift share ≥ threshold (default 30 %)
3. **Log** drift metrics to MLflow
4. **Register** new model in MLflow Model Registry
5. **Promote** to Production only if new F1 > current champion

```bash
PYTHONPATH=src python src/retrain.py \
  --ref data/processed/train.csv \
  --cur <new_production_data.csv> \
  --drift-threshold 0.3

# Force retrain regardless of drift:
PYTHONPATH=src python src/retrain.py --force
```

In CI, this runs automatically on every push to `main` after the drift-check job.

---

## CI/CD Pipeline (GitHub Actions)

Seven jobs run in sequence on every push / PR:

```
lint-and-test
    └── dvc-pipeline
            ├── drift-check
            │       └── retrain-if-drift (main only)
            └── docker-build
                    └── docker-push (main only)
                            └── metrics-comment (PR only)
```

**Required secrets:**

| Secret | Purpose |
|---|---|
| `DAGSHUB_USERNAME` | DVC remote + MLflow auth |
| `DAGSHUB_PASSWORD` | DVC remote + MLflow auth |
| `MLFLOW_TRACKING_USERNAME` | Same as DAGSHUB_USERNAME |
| `MLFLOW_TRACKING_PASSWORD` | Same as DAGSHUB_PASSWORD |
| `DOCKER_USERNAME` | Docker Hub push |
| `DOCKER_PASSWORD` | Docker Hub access token |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage:
- `TestEvaluationHelpers` — metric functions in `evaluation.py`
- `TestDataPreparation` — train/test split logic (synthetic data, no CSV needed)
- `TestModelArtifacts` — loads and queries `svm_model.pkl` (skipped if not present)
- `TestFastAPIEndpoints` — smoke tests `/health`, `/predict`, `/predict/batch`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `../models/svm_model.pkl` | Path to model artifact |
| `GOOGLE_API_KEY` | — | Gemini API key (tutor) |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `MLFLOW_TRACKING_USERNAME` | — | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | — | DagsHub access token |

---

## Project Status

| Component | Status |
|---|---|
| DVC pipeline (prepare → train → SVM → experiments) | ✅ Done |
| MLflow experiment tracking on DagsHub | ✅ Done |
| FastAPI prediction service | ✅ Done |
| Prometheus metrics (`/metrics` endpoint) | ✅ Done |
| Evidently drift detection | ✅ Done |
| Grafana dashboard (auto-provisioned) | ✅ Done |
| Docker + docker-compose (full stack) | ✅ Done |
| GitHub Actions CI/CD (7 jobs) | ✅ Done |
| PyTest suite | ✅ Done |
| Drift-triggered retraining loop | ✅ Done |
| MLflow Model Registry integration | ✅ Done |
| LLM/RAG Tutor (Gemini + LlamaIndex + Qdrant) | ✅ Done |
| Monitoring alerting rules (Prometheus) | ✅ Done |

---

## Contributors

* **Suhayb** ([@suhaybfr](https://github.com/suhaybfr)) — Project Lead & Serving
* **Monica** ([@monicaguru10](https://github.com/monicaguru10)) — CI/CD Pipeline & MLflow Artifact Logging
* **Neetu** ([@aekaboteneetu](https://github.com/aekaboteneetu))
* **Akhilesh** ([@akhileshchandaluri](https://github.com/akhileshchandaluri))
