"""Shared classification metrics, model-risk diagnostics, and JSON persistence.

R2 for classification is reported as the scale-invariant squared Pearson
correlation between the continuous model score and the binary label, because an
uncalibrated SVM decision function is not on the 0-to-1 label scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "r2_score",
    "mcc",
]


def positive_scores(model, X):
    """Return a continuous score for the positive class.

    Uses ``predict_proba`` when the estimator exposes it and falls back to
    ``decision_function`` for margin-based classifiers such as ``SVC``.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def squared_correlation(y_true, y_score) -> float:
    """Scale-invariant R^2: squared Pearson correlation of score and label."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if np.std(y_true) == 0 or np.std(y_score) == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_score)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr**2)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute the standard 8-metric classification evaluation dictionary."""
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


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return explicit confusion-matrix counts (tn, fp, fn, tp)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def model_risk_diagnostics(
    X_df: pd.DataFrame,
    y_true,
    y_pred,
    y_score,
    is_probability: bool = True,
) -> dict:
    """Compute Phase-2-ready model-risk, per-class, calibration, and slice metrics.

    In a digital learning tutor, ``correct == 0`` represents a struggling student
    who needs intervention; false positives on ``correct == 1`` represent missed
    interventions. Tracking ``class_0_recall`` (specificity) and slice stability
    prevents majority-class bias from hiding tutor failures.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    cm = confusion_counts(y_true_arr, y_pred_arr)
    neg_total = max(cm["tn"] + cm["fp"], 1)

    class_0_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    class_0_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    class_0_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    missed_intervention_rate = float(cm["fp"] / neg_total)

    diagnostics: dict = {
        "confusion_matrix": cm,
        "struggling_student_class0": {
            "precision": round(class_0_prec, 4),
            "recall_specificity": round(class_0_rec, 4),
            "f1_score": round(class_0_f1, 4),
            "missed_intervention_rate_fpr": round(missed_intervention_rate, 4),
        },
        "macro_f1": round(macro_f1, 4),
    }

    if is_probability and np.all((y_score_arr >= 0.0) & (y_score_arr <= 1.0)):
        diagnostics["brier_score"] = round(float(brier_score_loss(y_true_arr, y_score_arr)), 4)
        # Find probability threshold maximizing macro-F1 across candidate grid
        best_thr, best_mf1 = 0.5, macro_f1
        for thr in np.linspace(0.30, 0.85, 23):
            pred_t = (y_score_arr >= thr).astype(int)
            mf1_t = float(f1_score(y_true_arr, pred_t, average="macro", zero_division=0))
            if mf1_t > best_mf1:
                best_mf1 = mf1_t
                best_thr = float(thr)
        diagnostics["recommended_macro_f1_threshold"] = round(best_thr, 4)
        diagnostics["macro_f1_at_recommended_threshold"] = round(best_mf1, 4)

    # Subgroup / slice quality checks
    slices = {}
    if "skill_name" in X_df.columns:
        unmapped = X_df["skill_name"].astype(str) == "Unknown"
        slices["mapped_skills"] = ~unmapped
        slices["unmapped_skills"] = unmapped
    if "is_rapid_guess" in X_df.columns:
        rapid = pd.to_numeric(X_df["is_rapid_guess"], errors="coerce").fillna(0) == 1
        slices["normal_latency_ge_3s"] = ~rapid
        slices["rapid_guess_lt_3s"] = rapid
    elif "ms_first_response" in X_df.columns:
        rapid = pd.to_numeric(X_df["ms_first_response"], errors="coerce").fillna(0) < 3000
        slices["normal_latency_ge_3s"] = ~rapid
        slices["rapid_guess_lt_3s"] = rapid
    if "opportunity" in X_df.columns:
        first_opp = pd.to_numeric(X_df["opportunity"], errors="coerce").fillna(1) <= 1
        slices["cold_start_opportunity_1"] = first_opp
        slices["practiced_opportunity_gt_1"] = ~first_opp

    slice_report = {}
    for s_name, mask in slices.items():
        mask_arr = np.asarray(mask, dtype=bool)
        n_sub = int(mask_arr.sum())
        if n_sub >= 20 and len(np.unique(y_true_arr[mask_arr])) == 2:
            slice_report[s_name] = {
                "samples": n_sub,
                "accuracy": round(float(accuracy_score(y_true_arr[mask_arr], y_pred_arr[mask_arr])), 4),
                "f1_score": round(
                    float(f1_score(y_true_arr[mask_arr], y_pred_arr[mask_arr], zero_division=0)), 4
                ),
                "macro_f1": round(
                    float(
                        f1_score(
                            y_true_arr[mask_arr],
                            y_pred_arr[mask_arr],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "roc_auc": round(float(roc_auc_score(y_true_arr[mask_arr], y_score_arr[mask_arr])), 4),
            }
    diagnostics["slices"] = slice_report
    return diagnostics


def save_validation_json(payload: dict, out_path: str | Path) -> Path:
    """Write a deterministic, human-readable JSON validation file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return out_path
