# AI-Based Digital Learning Assistant

## Overview

An AI-based digital learning assistant designed to predict student performance and provide personalized learning support. The system combines a machine learning model for student performance prediction with an LLM-powered conversational tutor for answering questions, explaining concepts, and generating practice questions.

The project also focuses on implementing the end-to-end **MLOps workflow**, covering data management, model development, experiment tracking, testing, deployment, monitoring, and continuous improvement using industry-standard tools.

## Objectives

- Develop and evaluate a machine learning model to predict student learning outcomes.
- Build an LLM-powered tutor using Retrieval-Augmented Generation (RAG).
- Implement data and model versioning for reproducibility.
- Track experiments, evaluate models, and manage model versions.
- Automate testing and deployment through CI/CD.
- Monitor data drift, model performance, and application metrics.
- Implement a feedback loop for model retraining and redeployment.

## Technology Stack

| Component | Tools |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | scikit-learn, XGBoost |
| Dataset | ASSISTments Skill Builder |
| Experiment Tracking and Model Registry | MLflow |
| Dataset Versioning | DVC |
| Backend API | FastAPI |
| Database | PostgreSQL |
| LLM and RAG | Gemini API / OpenAI API, LlamaIndex, Qdrant |
| Version Control | Git, GitHub |
| Testing | PyTest |
| Containerization | Docker/Podman |
| CI/CD | GitHub Actions |
| Drift Detection | Evidently |
| Metrics and Monitoring | Prometheus, Grafana |

## MLOps Workflow

1. **Data Acquisition and Exploration:** Obtain student interaction data, perform exploratory data analysis, and identify relevant features.
2. **Feature Engineering:** Prepare training data using historical student interactions.
3. **Model Development:** Train baseline models and compare their performance with XGBoost.
4. **Experiment Tracking:** Log model parameters, evaluation metrics, and artifacts using MLflow.
5. **Data and Model Versioning:** Use DVC and MLflow to support reproducible experiments and model management.
6. **Testing:** Validate data processing, model predictions, and API functionality using PyTest.
7. **Deployment:** Serve the trained model through FastAPI and containerize services using Docker.
8. **CI/CD:** Automate testing and build workflows using GitHub Actions.
9. **Monitoring:** Track data drift, prediction behaviour, application latency, and errors.
10. **Continuous Improvement:** Incorporate new student interactions, evaluate the need for retraining, and manage updated model versions.

## Project Structure

```text
ML-ops/
├── .dvc/                               # DVC configuration and internal metadata
├── .dvcignore                          # Patterns ignored by DVC
├── data/
│   ├── raw/
│   │   ├── .gitignore                  # Excludes raw data file from Git tracking
│   │   ├── skill_builder_data.csv.dvc  # DVC pointer/metadata for raw dataset
│   │   └── skill_builder_data.csv      # Local raw dataset (tracked by DVC, ignored by Git)
│   ├── interim/                        # Cleaned intermediate data
│   │   ├── .gitignore                  # Excludes cleaned CSV from Git tracking
│   │   ├── cleaned_data.csv.dvc        # DVC pointer for cleaned base dataset
│   │   └── cleaned_data.csv            # Cleaned dataset (tracked by DVC)
│   └── processed/                      # Feature engineered & train/test datasets (tracked by DVC)
│       ├── .gitignore                  # Excludes engineered CSV from Git tracking
│       ├── engineered_features.csv.dvc # DVC pointer for feature matrix
│       └── engineered_features.csv     # Final feature matrix (17 columns, tracked by DVC)
├── models/                             # Trained model artifacts (tracked by DVC)
├── notebooks/
│   ├── 01_baseline_model.ipynb         # Logistic regression baseline on the engineered features
│   └── 02_svm_model_validation.ipynb   # SVM modelling, split and cross-validation study
├── reports/
│   ├── baseline_model_report.md        # Evaluation & architecture report
│   ├── data_dictionary.md              # Field definitions for engineered features
│   └── model_validation_report.md      # SVM, split and cross-validation results
├── docs/
│   ├── MLOPS_FLOW.md                   # End-to-end MLOps flow for the project
│   └── MODEL_VALIDATION.md             # Validation methods, results and model comparison
├── src/
│   ├── prepare.py                      # Data preprocessing & feature engineering pipeline
│   ├── train.py                        # Random Forest baseline training & evaluation stage
│   ├── train_svm.py                    # Kernel SVM training, evaluation & MLflow logging
│   ├── experiments.py                  # Split-robustness and model-comparison suite
│   ├── evaluation.py                   # Shared metric definitions
│   └── utils.py                        # MLflow DAGsHub configuration utility
├── tests/
├── params.yaml                         # Pipeline parameters and hyperparameters
├── dvc.yaml                            # DVC reproducible pipeline stages
├── dvc.lock                            # DVC pipeline dependency & output lock
├── CHANGELOG.md
└── README.md
```

## Reproducible Pipeline (DVC)

The pipeline is defined in `dvc.yaml`. DVC runs the steps in order and skips any step whose code, data or parameters haven't changed, so the same results can be reproduced from a clean checkout.

### Pipeline Stages
1. **Prepare (`src/prepare.py`):** Loads the raw CSV, keeps valid binary targets, fills missing feature values with the median, and writes the stratified `data/processed/train.csv` and `data/processed/test.csv`.
2. **Train (`src/train.py`):** Trains the Random Forest baseline, scores it with accuracy, precision, recall, F1 and ROC-AUC, logs the run to DAGsHub MLflow, and saves `models/baseline_model.pkl`.
3. **Train SVM (`src/train_svm.py`):** Trains an RBF-kernel SVM (scaled first, because SVMs care about feature scale), scores it the same way, logs the run plus ROC and confusion-matrix figures to MLflow, and saves `models/svm_model.pkl`.
4. **Experiments (`src/experiments.py`):** Puts logistic regression, SVM and random forest head to head over three train/test splits and 5-fold cross-validation, and writes `reports/experiment_results.csv` and `reports/cv_results.csv`.

### Running the Pipeline
To reproduce the entire pipeline (or only stages whose code/params have changed):
```bash
dvc repro
```

## Experiment Tracking (MLflow + DAGsHub)

All model experiments, hyperparameters, and evaluation metrics are automatically logged to the hosted DAGsHub MLflow server:
- **MLflow Tracking URI:** `https://dagshub.com/suhaybfr/ML-ops.mlflow`
- **Dashboard:** Navigate to the **Experiments** tab on [DAGsHub](https://dagshub.com/suhaybfr/ML-ops/experiments) to compare runs, view metric curves, and inspect logged model artifacts.

## Data Versioning with DVC

Data Version Control (DVC) is used to track and version large dataset files without storing large binaries directly in Git.

- **How it works:**
  - The dataset file (`skill_builder_data.csv`) is placed in `data/raw/`.
  - DVC calculates its content checksum and stores the metadata in `data/raw/skill_builder_data.csv.dvc`.
  - Git tracks the small `.dvc` pointer file and `data/raw/.gitignore`, while the large raw CSV is ignored by Git and managed by DVC.

### Common DVC Commands

- Check status of tracked datasets:
  ```bash
  dvc status
  ```
- Track a new or modified dataset:
  ```bash
  dvc add data/raw/skill_builder_data.csv
  git add data/raw/skill_builder_data.csv.dvc data/raw/.gitignore
  ```
- Retrieve/restore tracked data matching the current Git commit:
  ```bash
  dvc checkout
  ```
- Push tracked data to remote storage:
  ```bash
  dvc push
  ```
- Pull tracked data from remote storage:
  ```bash
  dvc pull
  ```

### Remote Storage (DAGsHub)

- **Remote Endpoint:** Configured to DAGsHub DVC storage at `https://dagshub.com/suhaybfr/ML-ops.dvc` (default remote: `origin`).
- **Data Synchronization:** The raw and processed datasets and model artifacts are backed up to DAGsHub via `dvc push`. Any collaborator can clone the repository and run `dvc pull` to obtain the exact data and model versions.
- **Security:** Remote credentials are kept in `.dvc/config.local` (ignored by Git) to prevent leaking access tokens.

## Current Status

Data versioning with DVC, the pipeline in `dvc.yaml` and MLflow tracking on DAGsHub are all working and verified. The model side is done too. We trained the Random Forest baseline and an RBF-kernel SVM, then compared both against logistic regression across three train/test splits and 5-fold cross-validation, and logged every run to the `student_performance_baseline` experiment on DAGsHub.

So far the Random Forest is the best model (F1 0.9589 and ROC-AUC 0.9783 on the 80/20 split) and the SVM is second (F1 0.9433), but the SVM still beats logistic regression on F1 and MCC. The full write-up is in `docs/MODEL_VALIDATION.md`, with the results report in `reports/model_validation_report.md`.

## Project Goal

The primary goal is to build a functional AI-based learning assistant while gaining practical experience with the complete machine learning lifecycle and the tools used to manage, deploy, monitor, and maintain ML systems.
