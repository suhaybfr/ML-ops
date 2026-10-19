# Model Validation

This document explains the predictive model in the AI-Based Digital Learning
Assistant, how it is evaluated, and why the evaluation is set up this way. The
model estimates how well a student is doing on the ASSISTments Skill Builder
data: given a problem attempt, does it predict that the attempt is correct
(`correct` in {0, 1})? That signal is what a digital literacy tutor needs in
order to decide when to give a hint or move on.

## What is trained

| Model | Role | Configuration |
|---|---|---|
| Random Forest | Baseline (already in the pipeline) | 100 trees, `max_depth=10` |
| Logistic Regression | Simple linear reference | `StandardScaler`, `max_iter=1000` |
| SVM (RBF kernel) | The model the teacher asked for | `StandardScaler`, `C=1.0`, `gamma="scale"` |

The SVM and logistic regression are wrapped in a `StandardScaler` pipeline
because both are distance/margin based and are sensitive to feature scale. The
features are the same six used by the baseline: `attempt_count`,
`ms_first_response`, `hint_count`, `hint_total`, `overlap_time` and
`opportunity`.

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

### Hold-out evaluation (training capped at 30,000 rows)

| Split | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | R² | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 80/20 | Logistic Regression | 0.9062 | 0.8815 | 0.9961 | 0.9353 | 0.9279 | 0.9239 | 0.6418 | 0.7861 |
| 80/20 | **SVM (RBF)** | 0.9184 | 0.8952 | 0.9968 | 0.9433 | 0.9017 | 0.9022 | 0.4600 | 0.8140 |
| 80/20 | **Random Forest** | 0.9419 | 0.9241 | 0.9965 | 0.9589 | 0.9783 | 0.9885 | 0.7946 | 0.8672 |
| 70/30 | Logistic Regression | 0.8863 | 0.8583 | 0.9978 | 0.9228 | 0.9346 | 0.9521 | 0.6162 | 0.7416 |
| 70/30 | **SVM (RBF)** | 0.9002 | 0.8727 | 0.9992 | 0.9317 | 0.9380 | 0.9227 | 0.4626 | 0.7737 |
| 70/30 | **Random Forest** | 0.9409 | 0.9225 | 0.9969 | 0.9583 | 0.9795 | 0.9892 | 0.7955 | 0.8650 |
| 60/40 | Logistic Regression | 0.8860 | 0.8581 | 0.9975 | 0.9226 | 0.9270 | 0.9332 | 0.6150 | 0.7406 |
| 60/40 | **SVM (RBF)** | 0.9168 | 0.8931 | 0.9972 | 0.9423 | 0.9224 | 0.9146 | 0.4505 | 0.8107 |
| 60/40 | **Random Forest** | 0.9411 | 0.9231 | 0.9965 | 0.9584 | 0.9792 | 0.9888 | 0.7962 | 0.8654 |

### Stratified 5-fold cross-validation (30,000 rows)

| Model | Accuracy | F1 | ROC-AUC | R² | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8791 ± 0.0128 | 0.9182 ± 0.0080 | 0.9316 ± 0.0055 | 0.5933 ± 0.0285 | 0.7257 ± 0.0292 |
| SVM (RBF) | 0.8864 ± 0.0164 | 0.9229 ± 0.0104 | 0.9341 ± 0.0167 | 0.4643 ± 0.0272 | 0.7425 ± 0.0371 |
| Random Forest | 0.9405 ± 0.0019 | 0.9579 ± 0.0013 | 0.9789 ± 0.0007 | 0.7950 ± 0.0043 | 0.8643 ± 0.0043 |

### Standalone SVM stage (`src/train_svm.py`)

Trained on 30,000 rows and evaluated on the full 20,000-row held-out test set
from the DVC `prepare` stage:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | R² | MCC |
|---|---|---|---|---|---|---|---|
| 0.8857 | 0.8570 | 0.9988 | 0.9225 | 0.9288 | 0.9136 | 0.4353 | 0.7406 |

## Reading the results

* The Random Forest is the strongest model on every split and is very stable
  under cross-validation (F1 standard deviation of 0.0013). It remains the
  accepted benchmark.
* The SVM is a clear improvement over logistic regression on F1 and MCC. Across
  the three splits its F1 moves by about 1.2 points, while the Random Forest
  moves by only 0.07 points, so the forest is the more stable model. The SVM
  trails the forest mainly on threshold-independent ranking (ROC-AUC, PR-AUC)
  and on R².

### How far behind is the SVM?

Difference against the Random Forest, in percentage points (negative = worse):

| Metric | 80/20 hold-out | Stratified 5-fold CV |
|---|---|---|
| Accuracy | −2.35 | −5.41 |
| Precision | −2.90 | −6.42 |
| Recall | +0.04 | +0.21 |
| F1 | −1.57 | −3.50 |
| ROC-AUC | −7.66 | −4.47 |
| PR-AUC | −8.63 | −6.39 |
| R² | −33.46 | −33.06 |
| MCC | −5.32 | −12.17 |

The SVM only wins on recall, and by a negligible amount. Against logistic
regression the SVM is ahead on accuracy (+1.22), precision (+1.37), F1 (+0.80)
and MCC (+2.79) but behind on ROC-AUC (−2.62) on the 80/20 split.
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
