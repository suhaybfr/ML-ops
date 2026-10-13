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
│   └── processed/                      # Preprocessed train/test datasets (tracked by DVC)
├── models/                             # Trained model artifacts (tracked by DVC)
├── notebooks/
├── src/
│   ├── prepare.py                      # Data preprocessing & train/test split stage
│   ├── train.py                        # Baseline model training & evaluation stage
│   └── utils.py                        # MLflow DAGsHub configuration utility
├── reports/
├── tests/
├── params.yaml                         # Pipeline parameters and hyperparameters
├── dvc.yaml                            # DVC reproducible pipeline stages
├── dvc.lock                            # DVC pipeline dependency & output lock
├── CHANGELOG.md
└── README.md
```

## Reproducible Pipeline (DVC)

The project uses DVC pipelines to chain data preprocessing and model training into an automated, version-controlled workflow.

### Pipeline Stages
1. **Prepare (`src/prepare.py`):** Loads raw data, filters invalid targets, performs median imputation, and generates stratified `data/processed/train.csv` and `data/processed/test.csv`.
2. **Train (`src/train.py`):** Trains a baseline Random Forest classifier, evaluates accuracy, precision, recall, F1, and ROC-AUC, logs run metadata to DAGsHub MLflow, and saves the model artifact to `models/baseline_model.pkl`.

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

DVC data versioning, the automated reproducible pipeline (`dvc.yaml`), and MLflow experiment tracking on DAGsHub are fully established and verified. A Random Forest baseline model has been trained and evaluated (~94% test accuracy, 0.98 ROC-AUC) with all parameters, metrics, and artifacts logged to DAGsHub.

## Project Goal

The primary goal is to build a functional AI-based learning assistant while gaining practical experience with the complete machine learning lifecycle and the tools used to manage, deploy, monitor, and maintain ML systems.
