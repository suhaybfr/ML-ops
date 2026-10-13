# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Interim Dataset Pipeline:** Exported cleaned base dataset to `data/interim/cleaned_data.csv` and enabled DVC tracking.
- **Feature Engineering Pipeline:** Engineered 9 hypothesis-driven features (learning curves, latency signals, target-encoded difficulties, and help-seeking indicators).
- **Processed Feature Matrix:** Generated and tracked model-ready feature set at `data/processed/engineered_features.csv` via DVC.
- **Data Dictionary:** Added comprehensive column definitions and hypothesis documentation in `reports/data_dictionary.md`.
- Added exploratory and baseline modeling Jupyter notebook (`notebooks/01_baseline_model.ipynb`)[cite: 2].
- Added comprehensive baseline model evaluation and MLOps architecture report (`reports/baseline_model_report.md`)[cite: 2].
- Created reproducible DVC pipeline (`dvc.yaml`, `dvc.lock`, `params.yaml`) with automated `prepare` and `train` stages[cite: 2].
- Added data preprocessing module (`src/prepare.py`) supporting median imputation, stratified train-test splitting, and processed dataset generation in `data/processed/`[cite: 2].
- Added baseline model training module (`src/train.py`) utilizing Random Forest classifier with performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)[cite: 2].
- Integrated DAGsHub hosted MLflow experiment tracking (`src/utils.py`), logging hyperparameters, evaluation metrics, and model artifacts directly to DAGsHub[cite: 2].
- Added root `.gitignore` covering Python cache files, virtual environments, and local credentials[cite: 2].
- Configured DAGsHub as default DVC remote storage (`origin`) at `https://dagshub.com/suhaybfr/ML-ops.dvc`[cite: 2].
- Uploaded raw and processed datasets and model artifacts to DAGsHub remote storage via `dvc push`[cite: 2].
- Stored remote authentication securely in `.dvc/config.local` (excluded from Git)[cite: 2].
- Added DVC documentation and DAGsHub remote workflow instructions to `README.md`[cite: 2].
- Tracked ASSISTments raw dataset (`skill_builder_data.csv`) with DVC under `data/raw/` via `data/raw/skill_builder_data.csv.dvc`[cite: 2].
- Initialized DVC (Data Version Control) repository configuration (`.dvc/`, `.dvcignore`, `.dvc/.gitignore`)[cite: 2].

### Fixed
- Fixed `RuntimeWarning: invalid value encountered in log1p` during feature engineering by applying non-negative clipping (`.clip(lower=0)`) to response times and opportunity counts.
- Resolved Jupyter kernel virtual environment mismatch by registering custom kernel (`mlops-venv`) linked directly to `.venv`.

### Changed
- Updated `README.md` with updated project structure (`data/interim/`, `data/processed/ engineered_features.csv`) and milestone progress.
- Relocated original dataset `skill_builder_data.csv` from repository root to `data/raw/skill_builder_data.csv`[cite: 2].
- Removed `skill_builder_data.csv` from Git tracking in the root folder[cite: 2].
- Removed obsolete placeholders across project directories[cite: 2].
