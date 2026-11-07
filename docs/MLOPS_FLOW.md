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
| `data/raw/skill_builder_data.csv.dvc` | DVC pointer for original 525,534-row ASSISTments dataset. |
| `data/interim/cleaned_data.csv.dvc` | DVC pointer for cleaned 26-column intermediate dataset. |
| `data/processed/engineered_features.csv.dvc` | DVC pointer for 17-column engineered feature matrix. |
| `src/features.py` | Shared 11-feature schema, leakage guardrails, and `select_xy()` helper. |
| `src/prepare.py` | Validation, `attempt_count` outlier capping, and stratified train/test partitioning. |
| `src/train.py` | Logistic Regression & Random Forest baseline training, slice diagnostics, and MLflow logging. |
| `src/train_svm.py` | RBF-kernel SVM training, evaluation, diagnostic plots, and MLflow logging. |
| `src/experiments.py` | Multi-split (`80/20`, `70/30`, `60/40`) and 5-fold CV comparison suite. |
| `src/evaluation.py` | Shared 8-metric definitions plus struggling-student (`class 0`) & subgroup slice checks. |
| `src/utils.py` | DAGsHub MLflow auth, offline fallback, standardized run tagging, and `MLmodel` upload. |
| `tests/test_phase1_pipeline.py` | Automated `pytest` suite checking leakage, schema, model inference, and quality gates. |
| `reports/*.json`, `reports/*.csv` | Git-tracked validation summaries matching DAGsHub MLflow runs. |

## Reproducing and Verifying Locally

```bash
dvc pull
dvc repro
pytest -v
```
