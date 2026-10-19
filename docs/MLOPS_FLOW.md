# MLOps Flow and Project Guide

This guide explains the MLOps work implemented so far in the AI-Based Digital Learning Assistant repository and how teammates can extend the baseline.

## Current flow

```text
ASSISTments raw CSV
   | DVC versioning and DAGsHub remote storage
   v
src/prepare.py
   | filtering, numeric conversion, median imputation, sampling, stratified split
   v
data/processed/train.csv and test.csv
   | DVC dependency graph
   v
src/train.py
   | Random Forest training and evaluation
   +--> models/baseline_model.pkl
   +--> DAGsHub MLflow run: parameters, metrics, model artifact

src/train_svm.py
   | scaled RBF-kernel SVM training and evaluation
   +--> models/svm_model.pkl
   +--> DAGsHub MLflow run: parameters, metrics, ROC curve, confusion matrix, model

src/experiments.py
   | logistic regression vs SVM vs random forest across 80/20, 70/30, 60/40
   | and stratified 5-fold cross-validation
   +--> reports/experiment_results.csv, reports/cv_results.csv, reports/figures/
   +--> DAGsHub MLflow parent run with nested runs per configuration
```

This is a working baseline and reproducibility foundation. It is not yet a complete production serving and monitoring system.

## Tools

### Git and GitHub

Git versions source code, configuration, documentation, reports, and small metadata files. GitHub is the shared remote for collaboration and review. Large data, credentials, virtual environments, caches, and generated local-only files should remain outside normal Git tracking.

Commit messages should be precise and describe one change, for example: `Document the DVC and MLflow baseline workflow`, `Add stratified preprocessing stage`, or `Track ASSISTments dataset with DVC`. Avoid vague or duplicated prefixes such as `chore:` and `docs:` when the message can directly explain the change.

### DVC

DVC versions large files and pipeline outputs without storing their binary contents in Git. The raw dataset is `data/raw/skill_builder_data.csv`; its pointer is `data/raw/skill_builder_data.csv.dvc`. Git versions the pointer and DVC stores or retrieves the actual file from the configured remote. The content hash connects a Git revision to the exact data used by that revision.

Useful commands are `dvc status`, `dvc pull`, `dvc push`, `dvc checkout`, and `dvc repro`. Remote credentials are kept in `.dvc/config.local` and must not be committed.

### DAGsHub

DAGsHub is used as the DVC remote for raw and processed data and model objects, and as the hosted MLflow tracking server for runs, metrics, parameters, and model artifacts. GitHub stores code and reproducibility metadata; DAGsHub stores large data and experiment records.

### DVC pipeline

`dvc.yaml` defines four stages:

1. `prepare` runs `python src/prepare.py`, consumes the raw CSV, preparation code, and `prepare.*` parameters, and produces `data/processed/train.csv` and `data/processed/test.csv`.
2. `train` runs `python src/train.py`, consumes the processed data, training code, MLflow utility code, and `train.*` parameters, and produces `models/baseline_model.pkl`.
3. `train_svm` runs `python src/train_svm.py`, consumes the processed data plus `src/evaluation.py`, and produces the kernel SVM model `models/svm_model.pkl`.
4. `experiments` runs `python src/experiments.py`, compares logistic regression, SVM and random forest across several train/test ratios and with cross-validation, and produces `reports/experiment_results.csv` and `reports/cv_results.csv`.

`dvc.lock` records dependency and output hashes for a successful pipeline state. DVC uses them to skip unchanged work and reproduce the same state.

### Parameters and preparation

`params.yaml` keeps important choices outside Python code. Current values are a 20% test split, seed `42`, a 100,000-row sample, and a Random Forest with 100 trees and maximum depth 10. The `svm` section holds the kernel, `C`, `gamma` and the 30,000-row training cap, and the `experiments` section holds the split ratios and cross-validation fold count. Changing a parameter changes the DVC stage signature, so `dvc repro` reruns the affected stage and downstream stages.

`src/prepare.py` uses Pandas to select six numerical features, retain binary `correct` targets, convert values to numeric, fill missing features with medians, optionally sample rows, perform a stratified split, and write the processed CSVs. Stratification preserves approximate class balance; the fixed seed makes the split repeatable.

scikit-learn provides the baseline `RandomForestClassifier` and evaluation metrics. Joblib serializes the fitted model to `models/baseline_model.pkl`, which future serving code can load.

## MLflow explained

MLflow and DVC solve different problems:

- **DVC:** Which exact data, code, parameters, and outputs belong to this version?
- **MLflow:** What happened during this training run, and how did the model perform?

`src/utils.py` configures MLflow by setting the DAGsHub tracking URI, loading credentials from environment variables or local DVC configuration, and selecting the `student_performance_baseline` experiment.

Inside `mlflow.start_run(run_name="random_forest_baseline")`, `src/train.py` records:

- **Parameters:** model type, tree count, maximum depth, seed, row counts, and feature names.
- **Metrics:** accuracy, precision, recall, F1 score, and ROC-AUC.
- **Artifacts:** the fitted scikit-learn model under the MLflow artifact path `model`; a local copy is saved in `models/`.

The DAGsHub MLflow page lets teammates compare runs while preserving their settings and results. MLflow is currently used for experiment tracking and artifact logging. A formal model registry, promotion stages, and automated deployment are not implemented yet.

## File map

| Path | Role |
|---|---|
| `.dvc/` | DVC configuration and local metadata. |
| `.dvc/config.local` | Local remote credentials; never commit. |
| `data/raw/` | Original DVC-managed input data. |
| `data/processed/` | Generated train/test partitions. |
| `src/prepare.py` | Preprocessing and stratified split stage. |
| `src/train.py` | Training, evaluation, local save, and MLflow logging. |
| `src/train_svm.py` | Kernel SVM training, evaluation, figures, and MLflow logging. |
| `src/experiments.py` | Split-robustness and model-comparison suite. |
| `src/evaluation.py` | Shared metric definitions used by every stage. |
| `src/utils.py` | MLflow/DAGsHub setup and credential loading. |
| `models/` | Generated local model artifacts. |
| `dvc.yaml` | Pipeline stages, dependencies, parameters, and outputs. |
| `dvc.lock` | Locked hashes for a pipeline state. |
| `params.yaml` | Preparation and training configuration. |
| `reports/baseline_model_report.md` | Baseline results and architecture report. |
| `reports/model_validation_report.md` | SVM and model-comparison evaluation report. |
| `docs/MODEL_VALIDATION.md` | Model validation methods, metrics and results. |
| `notebooks/` | Exploratory and baseline modelling work. |
| `tests/` | Test location; coverage still needs expansion. |
| `requirements.txt` | Current Python dependencies. |

## Implemented versus planned

Implemented: Git/GitHub workflow, DVC dataset tracking, DAGsHub DVC remote, reproducible preparation and training stages, parameterization, Random Forest baseline, kernel SVM, a model-comparison and cross-validation suite, shared metric definitions, DAGsHub MLflow tracking, local and MLflow model artifacts, the baseline and model validation reports, and the notebooks.

Planned but not implemented in this repository: FastAPI serving, Docker or Podman packaging, GitHub Actions CI/CD, PostgreSQL, LLM/RAG integration, Evidently drift reports, Prometheus/Grafana monitoring, automated retraining, and MLflow registry or promotion.

## Teammate workflow

1. Change preprocessing in `src/prepare.py`.
2. Add relevant configuration to `params.yaml` and `dvc.yaml`.
3. Update features or model logic in `src/train.py`.
4. Keep held-out test evaluation comparable to the baseline.
5. Log new parameters, metrics, and artifacts to the same MLflow experiment.
6. Run `dvc repro` and inspect the new MLflow run.
7. Update the report when a new model becomes the accepted benchmark.

Use a distinct MLflow run name for each comparison and retain the baseline run as the reference.

## Reproduce the current flow

From the repository root, with dependencies installed and DVC data available, run `dvc pull` followed by `dvc repro`. Then use `dvc status` and `git status` to inspect the result. Do not commit `.dvc/config.local`, credentials, virtual-environment files, notebook checkpoints, or other generated local-only files.

