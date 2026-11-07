# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

### Added
- **SVM Model:** Added an RBF-kernel SVM stage (`src/train_svm.py`) that trains on scaled features, saves `models/svm_model.pkl` and logs metrics and figures to MLflow.
- **Experiment Comparison:** Added `src/experiments.py` to compare logistic regression, SVM and random forest across three train/test splits and 5-fold cross-validation.
- **Shared Metrics:** Added `src/evaluation.py` so every script scores models with the same metric functions.
- **Validation Notebook:** Added `notebooks/02_svm_model_validation.ipynb` with the split and cross-validation results.
- **Validation Docs:** Added `docs/MODEL_VALIDATION.md` and `reports/model_validation_report.md` covering the validation methods and the SVM vs random forest comparison.
- **DVC Stages:** Added the `train_svm` and `experiments` stages to `dvc.yaml`, with their settings in `params.yaml`.
- **Interim Dataset Pipeline:** Exported cleaned base dataset to `data/interim/cleaned_data.csv` and enabled DVC tracking.
- **Feature Engineering Pipeline:** Engineered 9 hypothesis-driven features (learning curves, latency signals, target-encoded difficulties, and help-seeking indicators).
- **Processed Feature Matrix:** Generated and tracked model-ready feature set at `data/processed/engineered_features.csv` via DVC.
- **Data Dictionary:** Added comprehensive column definitions and hypothesis documentation in `reports/data_dictionary.md`.
- Added exploratory and baseline modeling Jupyter notebook (`notebooks/01_baseline_model.ipynb`).
- Added comprehensive baseline model evaluation and MLOps architecture report (`reports/baseline_model_report.md`).
- Created reproducible DVC pipeline (`dvc.yaml`, `dvc.lock`, `params.yaml`) with automated `prepare` and `train` stages.
- Added data preprocessing module (`src/prepare.py`) supporting median imputation, stratified train-test splitting, and processed dataset generation in `data/processed/`.
- Added baseline model training module (`src/train.py`) utilizing Random Forest classifier with performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC).
- Integrated DAGsHub hosted MLflow experiment tracking (`src/utils.py`), logging hyperparameters, evaluation metrics, and model artifacts directly to DAGsHub.
- Added root `.gitignore` covering Python cache files, virtual environments, and local credentials.
- Configured DAGsHub as default DVC remote storage (`origin`) at `https://dagshub.com/suhaybfr/ML-ops.dvc`.
- Uploaded raw and processed datasets and model artifacts to DAGsHub remote storage via `dvc push`.
- Stored remote authentication securely in `.dvc/config.local` (excluded from Git).
- Added DVC documentation and DAGsHub remote workflow instructions to `README.md`.
- Tracked ASSISTments raw dataset (`skill_builder_data.csv`) with DVC under `data/raw/` via `data/raw/skill_builder_data.csv.dvc`.
- Initialized DVC (Data Version Control) repository configuration (`.dvc/`, `.dvcignore`, `.dvc/.gitignore`).

### Fixed
- Fixed `RuntimeWarning: invalid value encountered in log1p` during feature engineering by applying non-negative clipping (`.clip(lower=0)`) to response times and opportunity counts.
- Resolved Jupyter kernel virtual environment mismatch by registering custom kernel (`mlops-venv`) linked directly to `.venv`.

### Changed
- Replaced the hardcoded DAGsHub token in `notebooks/01_baseline_model.ipynb` with the credential loading used by `src/utils.py`.
- Updated `requirements.txt` with the additional packages used by the new scripts.
- Updated `README.md` with the model stages, pipeline steps and current results.
- Updated `README.md` with updated project structure (`data/interim/`, `data/processed/ engineered_features.csv`) and milestone progress.
- Relocated original dataset `skill_builder_data.csv` from repository root to `data/raw/skill_builder_data.csv`.
- Removed `skill_builder_data.csv` from Git tracking in the root folder.
- Removed obsolete placeholders across project directories.
