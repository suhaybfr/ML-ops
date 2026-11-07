# Model Validation

This document explains the predictive model in the AI-Based Digital Learning
Assistant, how it is evaluated, and why the evaluation is set up this way. The
model estimates how well a student is doing on the ASSISTments Skill Builder
data: given a problem attempt, does it predict that the attempt is correct
(`correct` in {0, 1})? That signal is what a digital literacy tutor needs in
order to decide when to give a hint or move on.

## What is trained

| Model | Role | Artifact | Configuration |
|---|---|---|---|
| Random Forest | Accepted baseline / Phase 2 champion | `models/baseline_model.pkl` | 100 trees, `max_depth=10` |
| Logistic Regression | Simple linear reference | `models/logreg_baseline.pkl` | `StandardScaler`, `max_iter=1000` |
| SVM (RBF kernel) | The model the teacher asked for | `models/svm_model.pkl` | `StandardScaler`, `C=1.0`, `gamma="scale"` |

The SVM and logistic regression are wrapped in a `StandardScaler` pipeline
because both are distance/margin based and are sensitive to feature scale. All
three models share the exact 11 leakage-free engineered features defined in
`src/features.py` (`ENGINEERED_FEATURES` from `data/processed/engineered_features.csv`):
`log_opportunity`, `attempt_count`, `hint_count`, `bottom_hint`, `hint_ratio`,
`user_prev_acc`, `problem_difficulty`, `skill_difficulty`, `is_skill_unmapped`,
`log_response_time`, and `is_rapid_guess` (`first_try_correct` is excluded to
avoid target leakage).

The SVM is intentionally left at sane defaults rather than heavily tuned. Kernel
SVM training cost grows with the number of rows, so the training split is capped
at `svm.sample_size` (30,000 rows) while the test set stays complete.

## How it is validated

Three complementary techniques are used, so the result does not depend on one
lucky split:

1. **Hold-out evaluation at several train/test ratios** (`src/experiments.py`).
   The pooled data is re-split at 80/20, 70/30 and 60/40 with stratification so
   the class ratio is preserved, and every model is scored on each split.
2. **Stratified 5-fold cross-validation** on 30,000 rows. Cross-validation
   averages over five folds and reports a standard deviation, which is a much
   more reliable model comparison than a single split.
3. **Curve and error inspection**: ROC curves, precision-recall curves and a
   confusion matrix, saved to `reports/figures/` and logged to MLflow.

## Metrics

| Metric | What it tells us |
|---|---|
| Accuracy | Overall share of correct predictions |
| Precision / Recall | How clean the "correct" predictions are, and how many correct attempts are found |
| F1 score | Balance of precision and recall (primary metric here, because the classes are imbalanced) |
| ROC-AUC | Ranking quality across all thresholds |
| PR-AUC | Ranking quality that is more informative when classes are imbalanced |
| R² | Variance in the label explained by the model score (see note) |
| MCC | Single balanced correlation between predictions and labels |

**Note on R² for classification.** The textbook R² (`1 - SS_res / SS_tot`) is
not meaningful for a classifier whose score is a raw SVM decision function: the
score is not on the same 0-to-1 scale as the label, so the residual sum of
squares explodes and R² goes strongly negative even for an accurate model. We
therefore report the scale-invariant version, the squared Pearson correlation
between the continuous score and the label. It is computed in
`src/evaluation.py` so every stage reports the same value.

## Results

### Hold-out evaluation (`src/experiments.py`, training capped at 30,000 rows)

| Split | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | R² | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 80/20 | Logistic Regression | 0.9085 | 0.8994 | 0.9742 | 0.9353 | 0.9338 | 0.9489 | 0.6647 | 0.7869 |
| 80/20 | **SVM (RBF)** | 0.9259 | 0.9080 | 0.9912 | 0.9478 | 0.9259 | 0.9137 | 0.5176 | 0.8300 |
| 80/20 | **Random Forest** | **0.9435** | **0.9281** | **0.9939** | **0.9598** | **0.9742** | **0.9859** | **0.7881** | **0.8707** |
| 70/30 | Logistic Regression | 0.9099 | 0.8994 | 0.9767 | 0.9364 | 0.9356 | 0.9522 | 0.6688 | 0.7906 |
| 70/30 | **SVM (RBF)** | 0.9277 | 0.9095 | 0.9923 | 0.9491 | 0.9281 | 0.9165 | 0.5261 | 0.8344 |
| 70/30 | **Random Forest** | **0.9444** | **0.9293** | **0.9937** | **0.9604** | **0.9744** | **0.9859** | **0.7917** | **0.8727** |
| 60/40 | Logistic Regression | 0.9117 | 0.9012 | 0.9771 | 0.9376 | 0.9341 | 0.9493 | 0.6710 | 0.7948 |
| 60/40 | **SVM (RBF)** | 0.9276 | 0.9083 | 0.9936 | 0.9491 | 0.9305 | 0.9191 | 0.5222 | 0.8345 |
| 60/40 | **Random Forest** | **0.9436** | **0.9275** | **0.9947** | **0.9599** | **0.9741** | **0.9857** | **0.7902** | **0.8710** |

### Stratified 5-fold cross-validation (30,000 rows)

| Model | Accuracy | F1 | ROC-AUC | R² | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9118 ± 0.0029 | 0.9378 ± 0.0021 | 0.9375 ± 0.0031 | 0.6752 ± 0.0111 | 0.7953 ± 0.0072 |
| SVM (RBF) | 0.9277 ± 0.0021 | 0.9491 ± 0.0014 | 0.9291 ± 0.0028 | 0.5404 ± 0.0095 | 0.8346 ± 0.0049 |
| **Random Forest** | **0.9453 ± 0.0022** | **0.9611 ± 0.0016** | **0.9759 ± 0.0018** | **0.7983 ± 0.0078** | **0.8749 ± 0.0052** |

### Full-split baselines (`src/train.py`, 80,000 train / 20,000 test)

| Model | Artifact | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC | R² | MCC | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | `models/baseline_model.pkl` | **0.9468** | **0.9305** | **0.9960** | **0.9621** | **0.9362** | **0.9773** | **0.9876** | **0.8022** | **0.8785** | **0.0434** |
| Logistic Regression (`balanced`) | `models/logreg_baseline.pkl` | 0.8917 | 0.9188 | 0.9221 | 0.9204 | 0.8754 | 0.9404 | 0.9614 | 0.6435 | 0.7509 | 0.0821 |

### Standalone SVM stage (`src/train_svm.py`)

Trained on 30,000 rows and evaluated on the full 20,000-row held-out test set
from the DVC `prepare` stage:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | R² | MCC |
|---|---|---|---|---|---|---|---|
| 0.9288 | 0.9107 | 0.9924 | 0.9498 | 0.9313 | 0.9183 | 0.5196 | 0.8369 |

## Reading the results

* The Random Forest is the strongest model on every split, achieves well-calibrated
  probabilities (`Brier = 0.0434`), and is very stable under cross-validation
  (F1 standard deviation of `0.0016`). It remains the accepted benchmark and Phase 2
  champion model (`models/baseline_model.pkl`).
* Using the cleaned 11-feature matrix boosted the standalone SVM from `0.7406` MCC
  (on raw uncleaned timers) to `0.8369` MCC. Across the three splits its F1 moves
  by only `0.13` points (`0.9478`–`0.9491`), beating logistic regression on accuracy,
  precision, F1 and MCC, while trailing the Random Forest on threshold-independent
  ranking (ROC-AUC, PR-AUC) and R².

### How far behind is the SVM?

Difference against the Random Forest, in percentage points (negative = worse):

| Metric | 80/20 hold-out | Stratified 5-fold CV |
|---|---|---|
| Accuracy | −1.76 | −1.76 |
| Precision | −2.00 | −2.11 |
| Recall | −0.26 | −0.13 |
| F1 | −1.20 | −1.20 |
| ROC-AUC | −4.83 | −4.68 |
| PR-AUC | −7.22 | −7.03 |
| R² | −27.05 | −25.78 |
| MCC | −4.07 | −4.03 |

Against logistic regression on the 80/20 split, the SVM is ahead on accuracy
(+1.74), precision (+0.87), F1 (+1.25) and MCC (+4.32), while slightly behind on
ROC-AUC (−0.79).
* Every model has a very high recall and a much lower precision. The data is
  imbalanced (about 68% of attempts are correct), and at the default threshold
  the models lean towards predicting "correct". Accuracy and F1 are therefore
  the honest headline numbers, not accuracy alone.
* Results are stable across the 80/20, 70/30 and 60/40 splits, which is the
  point of running them.

## SVM versus Random Forest

On this dataset the Random Forest wins, but that is a property of the data, not
a verdict on the SVM. The two models fail and succeed in different regimes.

**Where an SVM can be the better choice**

* **Small datasets.** The max-margin objective is well behaved when there are few
  rows, while a Random Forest needs enough rows per split for its trees to be
  stable and will happily memorise noise in small samples.
* **High-dimensional or sparse features** (for example text-style features). A
  margin-based model with regularisation handles many features well.
* **Smooth, well-separated decision boundaries.** The RBF kernel can model a
  curved boundary with `C` and `gamma`; trees can only build axis-aligned
  staircase boundaries and need many splits to approximate a diagonal one.
* **A single deterministic solution.** SVM training is a convex problem with one
  global optimum, so the same data always gives the same model. A Random Forest
  is a greedy ensemble whose quality depends on the number of trees, depth and
  the random feature subsets.
* **Fewer knobs to tune.** The SVM here uses `C` and `gamma`; a Random Forest has
  tree count, depth, minimum samples per leaf, feature fraction and more.
* **Compact, fast inference.** An SVM depends only on its support vectors, so it
  can be small and quick to evaluate, whereas a forest of deep trees is a large
  object to load and score.

**Where a Random Forest wins (which is this dataset)**

* **Lots of rows of tabular data.** With tens of thousands of interactions, the
  forest's variance averaging pays off, which is exactly what the numbers show.
* **Non-linear feature interactions.** Trees split on thresholds and combine
  features natively (for example "many hints **and** long response time"), so
  they capture this without any kernel engineering.
* **Mixed scales and outliers.** Trees do not care about feature scale, so they
  need no `StandardScaler` and are less sensitive to odd values in the logs.
* **Native probability estimates** without needing extra calibration.

**Verdict for this project.** Keep the Random Forest as the model the tutor uses,
because it has the best and most stable scores. The SVM is still worth having: it
is the model the teacher asked for, it clearly beats logistic regression on F1
and MCC, its performance barely changes between splits, and it would become the
stronger option if the feature set grew much wider or the training data shrank.

## Experiment tracking

All runs are logged to the `student_performance_baseline` experiment on DAGsHub
MLflow, so the SVM and the comparison suite sit next to the original baseline:

- `src/train_svm.py` logs parameters, the eight metrics, the ROC curve and the
  confusion matrix, plus the model artifact.
- `src/experiments.py` logs a parent run `svm_validation_suite` with one nested
  run per (split, model) pair and one per cross-validation summary, and uploads
  `reports/experiment_results.csv`, `reports/cv_results.csv` and the figures.

Dashboard: `https://dagshub.com/suhaybfr/ML-ops/experiments`

## Reproducing

```bash
dvc repro train_svm     # SVM stage
dvc repro experiments   # split + cross-validation suite
```

Or run the scripts directly (`python src/train_svm.py`, `python src/experiments.py`),
or open `notebooks/02_svm_model_validation.ipynb` for the interactive version.
