# Baseline Model and MLOps Pipeline Report

**Project Title:** AI-Based Digital Learning Assistant  
**Dataset:** ASSISTments Skill Builder (`data/processed/engineered_features.csv`, 525,534 interactions)  
**Task:** Student Correctness Prediction (Binary Classification)  
**Tracking Platform:** DAGsHub (DVC Remote Storage & Hosted MLflow)  
**Validation Files:** `reports/prepare_summary.json`, `reports/baseline_metrics.csv`, `reports/baseline_metrics.json`

---

## 1. Executive Summary

This report documents the unified machine learning baselines and MLOps versioning pipeline for the AI-Based Digital Learning Assistant. The predictive module estimates whether a student solves an assigned problem correctly (`correct` $\in \{0, 1\}$), providing the intervention signal for the conversational tutor.

Both Phase 1 baseline architectures—a linear **Standardized Logistic Regression** reference (`models/logreg_baseline.pkl`) and a non-linear **Random Forest Classifier** (`models/baseline_model.pkl`)—are trained and evaluated on the versioned 11 hypothesis-driven engineered features. **Random Forest (`models/baseline_model.pkl`) is selected as the accepted baseline/champion model** heading into Phase 2 serving and registry, achieving **94.68% accuracy**, **0.9621 F1**, **0.9773 ROC-AUC**, **0.8785 MCC**, and the lowest probability calibration error (**0.0434 Brier score**).

---

## 2. Dataset & Feature Characterization

| Property | Value |
|---|---|
| Raw / Engineered Records | 525,534 student interactions |
| Target Variable | `correct` (`1` = solved correctly on first attempt, `0` = incorrect / needed hints) |
| Full Target Distribution | **67.95%** Correct (`1`), **32.05%** Incorrect (`0`) |
| Pipeline Modelling Partition | 100,000 stratified sample -> **80,000 train** / **20,000 test** (`src/prepare.py`) |
| Active Predictor Features (11) | `log_opportunity`, `attempt_count`, `hint_count`, `bottom_hint`, `hint_ratio`, `user_prev_acc`, `problem_difficulty`, `skill_difficulty`, `is_skill_unmapped`, `log_response_time`, `is_rapid_guess` |
| Excluded Leaky Column | `first_try_correct` (explicitly incorporates `correct == 1`) |

---

## 3. Baseline Models Compared (`src/train.py`)

| Model | Artifact | Configuration | Role |
|---|---|---|---|
| **Logistic Regression** | `models/logreg_baseline.pkl` | `StandardScaler` + `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")` | Interpretable linear reference |
| **Random Forest (Selected)** | `models/baseline_model.pkl` | `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)` | Primary non-linear benchmark & Phase 2 serving candidate |

### Held-out Test Set Results (20,000 samples)

| Model | Accuracy | Precision | Recall | F1-Score | Macro-F1 | ROC-AUC | PR-AUC | R² | MCC | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8917 | 0.9188 | 0.9221 | 0.9204 | 0.8754 | 0.9404 | 0.9614 | 0.6435 | 0.7509 | 0.0821 |
| **Random Forest** | **0.9468** | **0.9305** | **0.9960** | **0.9621** | **0.9362** | **0.9773** | **0.9876** | **0.8022** | **0.8785** | **0.0434** |

### Struggling-Student (`correct = 0`) & Subgroup Slice Diagnostics
- **Catching struggling students (`class 0`):** Random Forest identifies **84.23%** of struggling attempts with **99.01% precision** (`class 0 F1 = 0.9102`), outperforming balanced Logistic Regression (`class 0 F1 = 0.8304`).
- **Subgroup stability (`reports/baseline_metrics.json`):** Random Forest maintains strong discrimination across both mapped skills (`ROC-AUC = 0.9720`) and unmapped skills (`ROC-AUC = 0.9942`), as well as cold-start first opportunities (`opportunity == 1`, `ROC-AUC = 0.9451`) and practiced attempts (`ROC-AUC = 0.9796`).

### Feature Importance Summary (`reports/figures/baseline_feature_importance.png`)
1. `attempt_count` (`0.4216` Gini importance; `-4.66` standardized log-odds): Strongest negative signal of first-attempt mastery.
2. `bottom_hint` (`0.1437`), `hint_count` (`0.1356`), and `hint_ratio` (`0.1270`): Direct help-seeking indicators.
3. `problem_difficulty` (`0.0907`), `user_prev_acc` (`0.0271`), and `skill_difficulty` (`0.0262`): Item, student, and domain context priors.
4. `log_response_time` (`0.0134`), `log_opportunity` (`0.0092`), and `is_rapid_guess` (`0.0036`): Latency and learning-curve signals.

---

## 4. MLOps Reproducibility & Artifact Tracking

1. **DVC Dataset & Model Versioning:**
   - Raw (`data/raw/skill_builder_data.csv.dvc`), cleaned (`data/interim/cleaned_data.csv.dvc`), engineered (`data/processed/engineered_features.csv.dvc`), split (`train.csv`, `test.csv`), and model binaries (`models/*.pkl`) are versioned in DVC and backed by DAGsHub storage.
2. **Git-Tracked Validation Files (`cache: false` in `dvc.yaml`):**
   - `reports/prepare_summary.json`, `reports/baseline_metrics.csv`, `reports/baseline_metrics.json`, and `reports/figures/baseline_*.png` remain directly visible in Git for review and automated `pytest` quality gates.
3. **MLflow Experiment Tracking on DAGsHub:**
   - Logged to `https://dagshub.com/suhaybfr/ML-ops.mlflow` with standardized tags, full 8-metric + risk/calibration summaries, validation JSON/CSV artifacts, and complete `MLmodel` bundles (`ModelSignature` + `input_example`).
