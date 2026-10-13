# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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

### Changed
- Updated `README.md` with pipeline execution instructions, MLflow experiment links, and updated project structure.
- Relocated original dataset `skill_builder_data.csv` from repository root to `data/raw/skill_builder_data.csv`, verifying MD5 checksum preservation (`9410c31fadc7ad6518d296ce254ec4ab`).
- Removed `skill_builder_data.csv` from Git tracking in the root folder.
- Removed obsolete placeholders across project directories.
