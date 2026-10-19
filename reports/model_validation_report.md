# Model Evaluation Report: SVM and Baseline Comparison

**Project Title:** AI-Based Digital Learning Assistant
**Dataset:** ASSISTments Skill Builder (`data/raw/skill_builder_data.csv`, 525,534 interactions)
**Task:** Student correctness prediction (binary classification)
**Tracking Platform:** DAGsHub (DVC remote storage and hosted MLflow)
**Companion document:** `docs/MODEL_VALIDATION.md`

---

## 1. Summary

This report covers the modelling and evaluation stage of the project. The
predictive component estimates whether a student's problem attempt is correct,
which is the signal the digital literacy tutor uses to decide when to intervene.

The Random Forest baseline was joined by two comparison models, including the
RBF-kernel SVM requested for the project, and the models were evaluated on
several train/test splits and with stratified cross-validation instead of a
single split. The Random Forest remains the strongest model: on the 80/20 split
it reaches **F1 0.9589** and **ROC-AUC 0.9783**, and it is the most stable model
under 5-fold cross-validation (F1 standard deviation 0.0013). The SVM improves
clearly over logistic regression (F1 0.9433 vs 0.9353 on the same split) and is
robust across splits, but does not overtake the Random Forest.

---

## 2. Models compared

| Model | Configuration | Purpose |
|---|---|---|
| Random Forest | 100 trees, `max_depth=10` | Baseline benchmark |
| Logistic Regression | `StandardScaler`, `max_iter=1000` | Simple linear reference |
| SVM (RBF kernel) | `StandardScaler`, `C=1.0`, `gamma="scale"` | Model requested by the teacher |

Features: `attempt_count`, `ms_first_response`, `hint_count`, `hint_total`,
`overlap_time`, `opportunity`. The SVM is kept close to default settings rather
than exhaustively tuned; training is capped at 30,000 rows because kernel SVM
cost grows with the number of rows, while the test set stays complete.

---

## 3. Validation methods

1. **Hold-out at three train/test ratios** (80/20, 70/30, 60/40), stratified to
   preserve the class ratio (`src/experiments.py`).
2. **Stratified 5-fold cross-validation** on 30,000 rows, reported as mean and
   standard deviation.
3. **Curve and error inspection**, i.e. ROC curves, precision-recall curves and a
   confusion matrix (`reports/figures/`).

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, R² and MCC. The full
metric tables and the R² note are in `docs/MODEL_VALIDATION.md`.

---

## 4. Headline results

**Hold-out, 80/20 split (training capped at 30,000 rows):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | R² |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9062 | 0.8815 | 0.9961 | 0.9353 | 0.9279 | 0.6418 |
| SVM (RBF) | 0.9184 | 0.8952 | 0.9968 | 0.9433 | 0.9017 | 0.4600 |
| **Random Forest** | **0.9419** | **0.9241** | 0.9965 | **0.9589** | **0.9783** | **0.7946** |

**Stratified 5-fold cross-validation (30,000 rows):**

| Model | Accuracy | F1 | ROC-AUC | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.8791 ± 0.0128 | 0.9182 ± 0.0080 | 0.9316 ± 0.0055 | 0.7257 ± 0.0292 |
| SVM (RBF) | 0.8864 ± 0.0164 | 0.9229 ± 0.0104 | 0.9341 ± 0.0167 | 0.7425 ± 0.0371 |
| **Random Forest** | **0.9405 ± 0.0019** | **0.9579 ± 0.0013** | **0.9789 ± 0.0007** | **0.8643 ± 0.0043** |

**Standalone SVM stage** (`src/train_svm.py`, 20,000-row held-out test set):
Accuracy 0.8857, Precision 0.8570, Recall 0.9988, F1 0.9225, ROC-AUC 0.9288,
PR-AUC 0.9136, R² 0.4353, MCC 0.7406.

The full result tables are written to `reports/experiment_results.csv` and
`reports/cv_results.csv`.

---

## 5. Interpretation and limitations

- **Random Forest stays the benchmark.** It wins on every split and shows very
  low cross-validation variance, so it is the model the tutor should use.
- **The SVM is a solid, robust alternative.** It beats logistic regression on
  accuracy, precision, F1 and MCC, and its F1 varies by about 1.2 points across
  the three splits. It still trails the Random Forest on ranking quality
  (ROC-AUC 7.7 points lower, PR-AUC 8.6 points lower on the 80/20 split) and is
  less stable than the forest, whose F1 moves by only 0.07 points.
- **The classes are imbalanced.** About 68% of attempts are correct, and at the
  default threshold all models favour the "correct" class, giving very high
  recall (~0.996) with lower precision (~0.86-0.92). F1 and MCC are therefore
  more honest summaries than accuracy alone. Adjusting the decision threshold or
  class weights is a reasonable future improvement.
- **SVM training is capped at 30,000 rows** for tractability. Comparisons within
  the suite are fair (every model sees the same rows), but the Random Forest
  could use more data if the cap were raised.

---

## 6. Reproducibility and tracking

- DVC stages: `train_svm` and `experiments` in `dvc.yaml`, driven by
  `params.yaml`.
- MLflow: all runs are logged to the `student_performance_baseline` experiment on
  DAGsHub (`https://dagshub.com/suhaybfr/ML-ops/experiments`), including a
  parent suite run with nested runs per split/model and per cross-validation
  summary.
- Notebook: `notebooks/02_svm_model_validation.ipynb`.

---

## 7. Scope

This report covers the model portion of the project: data exploration, feature
selection, baseline and SVM models, evaluation and experiment tracking. The
remaining parts of the system - the FastAPI serving layer, the LLM/RAG tutor,
containerisation, CI/CD and monitoring - are handled separately by teammates.
