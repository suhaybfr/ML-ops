"""Metric helpers shared by the training scripts and the notebook.

R2 for classification is the squared correlation between the score and the
label. The usual R2 formula is not usable here because an SVM decision
function is not on the same scale as a 0/1 label, which pushes R2 strongly
negative even for an accurate model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_scores(model, X):
    """Return a continuous score for the positive class.

    Uses ``predict_proba`` when the estimator exposes it and falls back to
    ``decision_function``, which is required for plain SVM classifiers.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def squared_correlation(y_true, y_score):
    """Scale-invariant R^2: the squared Pearson correlation of score and label."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if np.std(y_true) == 0 or np.std(y_score) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_score)[0, 1] ** 2)


def classification_metrics(y_true, y_pred, y_score):
    """Compute the full classification metric set for a hold-out evaluation."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "r2_score": squared_correlation(y_true, y_score),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
