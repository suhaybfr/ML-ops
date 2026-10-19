# Baseline Model and MLOps Pipeline Report

**Project Title:** AI-Based Digital Learning Assistant  
**Dataset:** ASSISTments Skill Builder Dataset (525,534 interactions)  
**Task:** Student Correctness Prediction (Binary Classification)  
**Tracking Platform:** DAGsHub (DVC Remote Storage & Hosted MLflow)  

---

## 1. Executive Summary

This report documents the initial machine learning baseline and MLOps data versioning infrastructure developed for the AI-Based Digital Learning Assistant. The predictive component forecasts student learning outcomes—specifically, whether a student successfully solves a problem on their first attempt without requiring intervention.

An end-to-end reproducible pipeline was established using Data Version Control (DVC) and MLflow experiment tracking on DAGsHub. The baseline Random Forest model achieved **94.06% accuracy** and an **ROC-AUC of 0.9810** on the held-out test partition.

---

## 2. Dataset Characterization

The ASSISTments Skill Builder dataset captures fine-grained student problem-solving interaction logs from an online tutoring system.

| Property | Value |
|---|---|
| Total Raw Records | 525,534 rows |
| Raw File Size | 83.2 MB |
| Target Variable | `correct` (1 = solved correctly on first attempt, 0 = incorrect / needed hints) |
| Target Distribution | ~69.3% Correct (1), ~30.7% Incorrect (0) |
| Primary Features Used | `attempt_count`, `ms_first_response`, `hint_count`, `hint_total`, `overlap_time`, `opportunity` |

---

## 3. Data Preprocessing and Partitioning

The automated data preparation stage (`src/prepare.py`) performs the following operations:
1. **Target Validation:** Filters rows to valid binary outcomes (`correct` $\in \{0, 1\}$).
2. **Missing Value Imputation:** Imputes missing numerical entries using feature medians to avoid data leakage.
3. **Partitioning:** Executes an 80/20 stratified split preserving target class ratios:
   - **Training Set (`data/processed/train.csv`):** 80,000 samples
   - **Testing Set (`data/processed/test.csv`):** 20,000 samples

---

## 4. Baseline Model Performance

A Random Forest Classifier was trained to establish an empirical benchmark:

### Model Hyperparameters
- **Estimators (`n_estimators`):** 100
- **Maximum Depth (`max_depth`):** 10
- **Random Seed (`random_state`):** 42
- **Parallel Workers (`n_jobs`):** -1 (all available CPU cores)

### Evaluation Metrics (Held-out Test Set)

| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | **0.9406** | 94.1% overall correct classifications on unseen interactions |
| **Precision** | **0.9221** | 92.2% of predicted correct attempts were true positives |
| **Recall** | **0.9971** | Successfully captures 99.7% of all correct student attempts |
| **F1-Score** | **0.9581** | Harmonic mean reflecting balanced classification power |
| **ROC-AUC** | **0.9810** | High separability between correct and incorrect student actions |

### Feature Importance Summary
1. `attempt_count`: Strongest negative predictor of first-time correctness.
2. `hint_count` & `hint_total`: Explicit indicator of student difficulty.
3. `ms_first_response`: Response latency correlating with cognitive load.
4. `opportunity`: Learning curve progression over repeated skill practice.

---

## 5. MLOps Architecture

```text
Raw Dataset (data/raw/) ──[DVC Tracked]──> DAGsHub Storage
        │
        ▼ (src/prepare.py)
Processed Partitions (data/processed/) ──[DVC Tracked]
        │
        ▼ (src/train.py)
Random Forest Model (models/baseline_model.pkl) ──[DVC Tracked]
        │
        └──> Parameters, Metrics, Artifacts ──> Hosted MLflow (DAGsHub)
```

1. **DVC Pipeline (`dvc.yaml` & `dvc.lock`):**
   - Chains `prepare` and `train` stages into a single reproducible execution graph.
   - Any team member can reproduce the pipeline using `dvc repro`.
2. **DAGsHub Remote Storage:**
   - Raw and processed datasets and model artifacts are versioned in DAGsHub DVC storage.
   - Collaborators sync data via `dvc pull`.
3. **MLflow Tracking:**
   - Experiments, hyperparameter logs, and metric curves are hosted at:  
     `https://dagshub.com/suhaybfr/ML-ops.mlflow`

---

## 6. Next Steps

The core MLOps plumbing and baseline benchmark are established. The model work
(EDA, feature engineering, baseline and SVM models, evaluation and experiment
tracking) is complete and documented in `reports/model_validation_report.md` and
`docs/MODEL_VALIDATION.md`.

The remaining system modules are handled separately by teammates:
1. **Deployment & Serving:** build FastAPI inference endpoints (`src/api.py`) and
   containerize services with Docker.
2. **LLM Tutor (RAG):** connect the predictive model to the conversational tutor.
3. **Monitoring & Feedback Loop:** drift detection and retraining triggers.
