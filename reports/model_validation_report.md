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

The Random Forest baseline (`src/train.py`, 80,000 train / 20,000 test) reaches
**Accuracy 0.9468**, **F1 0.9621**, **ROC-AUC 0.9773**, and **MCC 0.8785** on the
unified 11 engineered features (`src/features.py`). In the head-to-head comparison
suite (`src/experiments.py`, 30,000-row training cap for fair kernel-SVM comparison),
the Random Forest remains the strongest model: on the 80/20 split it reaches
**F1 0.9598** and **ROC-AUC 0.9742**, and under 5-fold cross-validation it achieves
**F1 0.9611 ± 0.0016**. The RBF-kernel SVM improves clearly over logistic regression
(F1 0.9478 vs 0.9353 on 80/20; MCC 0.8300 vs 0.7869) and is stable across splits,
but does not overtake the Random Forest.

---

## 2. Models compared

| Model | Configuration | Artifact | Purpose |
|---|---|---|---|
| Random Forest | 100 trees, `max_depth=10` | `models/baseline_model.pkl` | Accepted baseline / Phase 2 champion |
| Logistic Regression | `StandardScaler`, `max_iter=1000` | `models/logreg_baseline.pkl` | Linear reference baseline |
| SVM (RBF kernel) | `StandardScaler`, `C=1.0`, `gamma="scale"` | `models/svm_model.pkl` | Non-linear margin model requested by the teacher |

Features (`ENGINEERED_FEATURES` in `src/features.py`, sourced from
`data/processed/engineered_features.csv`): `log_opportunity`, `attempt_count`,
`hint_count`, `bottom_hint`, `hint_ratio`, `user_prev_acc`, `problem_difficulty`,
`skill_difficulty`, `is_skill_unmapped`, `log_response_time`, `is_rapid_guess`
(`first_try_correct` is excluded to prevent target leakage). Kernel SVM training
is capped at 30,000 rows (`svm.sample_size`) for computational tractability, while
held-out test splits stay complete.

---

## 3. Validation methods

1. **Hold-out at three train/test ratios** (80/20, 70/30, 60/40), stratified to
   preserve the class ratio (`src/experiments.py`).
2. **Stratified 5-fold cross-validation** on 30,000 rows, reported as mean and
   standard deviation.
3. **Curve, calibration, and subgroup slice inspection**, i.e. ROC curves,
   precision-recall curves, confusion matrices (`reports/figures/`), Brier
   calibration, struggling-student (class 0) recall, and slice diagnostics
   (`reports/baseline_metrics.json`, `reports/svm_metrics.json`).

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, R² and MCC. The full
metric tables and the R² note are in `docs/MODEL_VALIDATION.md`.

---

## 4. Headline results

**Hold-out, 80/20 split (`src/experiments.py`, training capped at 30,000 rows):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | R² | MCC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9085 | 0.8994 | 0.9742 | 0.9353 | 0.9338 | 0.9489 | 0.6647 | 0.7869 |
| SVM (RBF) | 0.9259 | 0.9080 | 0.9912 | 0.9478 | 0.9259 | 0.9137 | 0.5176 | 0.8300 |
| **Random Forest** | **0.9435** | **0.9281** | **0.9939** | **0.9598** | **0.9742** | **0.9859** | **0.7881** | **0.8707** |

**Stratified 5-fold cross-validation (30,000 rows):**

| Model | Accuracy | F1 | ROC-AUC | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.9118 ± 0.0029 | 0.9378 ± 0.0021 | 0.9375 ± 0.0031 | 0.7953 ± 0.0072 |
| SVM (RBF) | 0.9277 ± 0.0021 | 0.9491 ± 0.0014 | 0.9291 ± 0.0028 | 0.8346 ± 0.0049 |
| **Random Forest** | **0.9453 ± 0.0022** | **0.9611 ± 0.0016** | **0.9759 ± 0.0018** | **0.8749 ± 0.0052** |

**Full-split baselines (`src/train.py`, 80,000 train / 20,000 test):**
- **Random Forest (`models/baseline_model.pkl`):** Accuracy 0.9468, Precision 0.9305,
  Recall 0.9960, F1 0.9621, Macro-F1 0.9362, ROC-AUC 0.9773, PR-AUC 0.9876,
  R² 0.8022, MCC 0.8785, Brier score 0.0434, struggling-student (class 0) recall 0.8423.
- **Logistic Regression (`models/logreg_baseline.pkl`, `class_weight="balanced"`):**
  Accuracy 0.8917, Precision 0.9188, Recall 0.9221, F1 0.9204, Macro-F1 0.8754,
  ROC-AUC 0.9404, PR-AUC 0.9614, R² 0.6435, MCC 0.7509, Brier score 0.0821.

**Standalone SVM stage** (`src/train_svm.py`, 30,000 train / 20,000 held-out test set):
Accuracy 0.9288, Precision 0.9107, Recall 0.9924, F1 0.9498, Macro-F1 0.9135,
ROC-AUC 0.9313, PR-AUC 0.9183, R² 0.5196, MCC 0.8369.

The full result tables and JSON validation artifacts are tracked in Git under
`reports/baseline_metrics.csv`, `reports/baseline_metrics.json`,
`reports/svm_metrics.json`, `reports/experiment_results.csv`, and
`reports/cv_results.csv`.

---

## 5. Interpretation and limitations

- **Random Forest stays the benchmark.** It wins on every split, achieves the
  best probability calibration (Brier 0.0434), detects 84.2% of struggling attempts
  with 99.0% class-0 precision, and shows very low cross-validation variance, so it
  is the model promoted toward Phase 2 serving.
- **The SVM improved substantially on the cleaned 11-feature matrix.** Moving from
  raw uncleaned timers to `engineered_features.csv` boosted standalone SVM MCC from
  0.7406 to 0.8369 (+9.6 points) and accuracy from 88.57% to 92.88%. It beats
  logistic regression on accuracy, precision, F1 and MCC across every split, while
  trailing the Random Forest on ranking quality (ROC-AUC ~4.8 points lower on 80/20).
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
