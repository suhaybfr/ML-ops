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
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
├── tests/
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

### Current Limitations and Next Steps

- **No Remote Storage Configured:** Currently, datasets are cached locally in the `.dvc/cache` directory. No remote storage (e.g. AWS S3, Google Drive, DAGsHub, Azure Blob) has been configured yet.
- **What remains necessary:** To share data with teammates or across machines, an authorized remote storage destination must be configured:
  ```bash
  # Example (configure once authorized):
  dvc remote add -d <remote_name> <remote_url>
  dvc push
  ```

## Current Status

DVC data versioning is set up and verified. The ASSISTments Skill Builder raw dataset is properly placed in `data/raw/` and version-controlled via DVC metadata (`skill_builder_data.csv.dvc`), with the raw data file excluded from Git tracking.

## Project Goal

The primary goal is to build a functional AI-based learning assistant while gaining practical experience with the complete machine learning lifecycle and the tools used to manage, deploy, monitor, and maintain ML systems.
