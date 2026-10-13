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
├── .dvc/                               # DVC configuration and internal metadata[cite: 3]
├── .dvcignore                          # Patterns ignored by DVC[cite: 3]
├── data/
│   ├── raw/
│   │   ├── .gitignore                  # Excludes raw data file from Git tracking[cite: 3]
│   │   ├── skill_builder_data.csv.dvc  # DVC pointer/metadata for raw dataset[cite: 3]
│   │   └── skill_builder_data.csv      # Local raw dataset (tracked by DVC, ignored by Git)[cite: 3]
│   ├── interim/                        # Cleaned intermediate data[cite: 3]
│   │   ├── .gitignore                  # Excludes cleaned CSV from Git tracking
│   │   ├── cleaned_data.csv.dvc        # DVC pointer for cleaned base dataset
│   │   └── cleaned_data.csv            # Cleaned dataset (tracked by DVC)
│   └── processed/                      # Feature engineered & train/test datasets (tracked by DVC)[cite: 3]
│       ├── .gitignore                  # Excludes engineered CSV from Git tracking
│       ├── engineered_features.csv.dvc # DVC pointer for feature matrix
│       └── engineered_features.csv     # Final feature matrix (17 columns, tracked by DVC)
├── models/                             # Trained model artifacts (tracked by DVC)[cite: 3]
├── notebooks/
├── reports/
│   ├── baseline_model_report.md       # Evaluation & architecture report[cite: 2]
│   └── data_dictionary.md             # Field definitions for engineered features
├── src/
│   ├── prepare.py                      # Data preprocessing & feature engineering pipeline[cite: 3]
│   ├── train.py                        # Model training & evaluation stage[cite: 3]
│   └── utils.py                        # MLflow DAGsHub configuration utility[cite: 3]
├── tests/
├── params.yaml                         # Pipeline parameters and hyperparameters[cite: 3]
├── dvc.yaml                            # DVC reproducible pipeline stages[cite: 3]
├── dvc.lock                            # DVC pipeline dependency & output lock[cite: 3]
├── CHANGELOG.md
└── README.md
```

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
- **Data Synchronization:** The raw dataset is backed up to DAGsHub via `dvc push`. Any collaborator can clone the repository and run `dvc pull` to obtain the exact raw data version.
- **Security:** Remote credentials are kept in `.dvc/config.local` (ignored by Git) to prevent leaking access tokens.

## Current Status

DVC data versioning is set up, verified, and linked to DAGsHub remote storage. The ASSISTments Skill Builder raw dataset is properly stored in `data/raw/` and version-controlled via DVC metadata (`skill_builder_data.csv.dvc`), with the raw data file excluded from Git and safely synced to DAGsHub.

## Project Goal

The primary goal is to build a functional AI-based learning assistant while gaining practical experience with the complete machine learning lifecycle and the tools used to manage, deploy, monitor, and maintain ML systems.
