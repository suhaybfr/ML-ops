"""Phase 2 Model-Risk & Quality Gate validator.

Verifies that the accepted baseline / champion model (Random Forest,
``models/baseline_model.pkl``) meets strict classification, probability
calibration (Brier score), struggling-student (``class 0``) intervention recall,
leakage-free 11-feature schema, subgroup slice floors, and optional live
artifact inference + latency checks before registry promotion or container build.

Usage:
    python -m src.validate_model
    python -m src.validate_model --require-artifact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.features import ENGINEERED_FEATURES, LEAKY_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = REPO_ROOT / "reports" / "baseline_metrics.json"
DEFAULT_CHAMPION_KEY = "random_forest_baseline"
DEFAULT_REFERENCE_KEY = "logistic_regression_baseline"

DEFAULT_THRESHOLDS: dict[str, float] = {
    "accuracy": 0.92,
    "f1_score": 0.94,
    "roc_auc": 0.96,
    "mcc": 0.84,
    "macro_f1": 0.90,
    "max_brier_score": 0.08,
    "class0_recall": 0.80,
    "class0_precision": 0.85,
    "slice_min_f1": 0.88,
    "slice_min_macro_f1": 0.85,
    "slice_min_roc_auc": 0.90,
    "max_slice_roc_auc_gap": 0.10,
    "max_single_latency_ms": 100.0,
}

REQUIRED_SLICES = [
    "mapped_skills",
    "unmapped_skills",
    "normal_latency_ge_3s",
    "rapid_guess_lt_3s",
    "cold_start_opportunity_1",
    "practiced_opportunity_gt_1",
]


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    actual: Any,
    expected: str,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
        }
    )


def validate_baseline_metrics(
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    model_key: str = DEFAULT_CHAMPION_KEY,
    reference_key: str | None = DEFAULT_REFERENCE_KEY,
    thresholds: dict[str, float] | None = None,
    require_artifact: bool = False,
) -> dict[str, Any]:
    """Run all Phase 2 model-risk and quality-gate checks.

    Returns a structured report dictionary with ``passed`` (bool), ``checks``,
    and ``failures``.
    """
    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    metrics_path = Path(metrics_path)
    checks: list[dict[str, Any]] = []

    if not metrics_path.exists():
        raise FileNotFoundError(f"Validation metrics file not found: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        payload = json.load(f)

    # 1. Feature contract & leakage guardrails
    reported_features = payload.get("features", [])
    _check(
        checks,
        "feature_set_name",
        payload.get("feature_set") == "engineered_11",
        payload.get("feature_set"),
        "== 'engineered_11'",
    )
    _check(
        checks,
        "exact_11_engineered_features",
        reported_features == list(ENGINEERED_FEATURES),
        reported_features,
        f"exact 11 features ({len(ENGINEERED_FEATURES)})",
    )
    leaky_present = [c for c in LEAKY_COLS if c in reported_features]
    _check(
        checks,
        "no_target_leakage_columns",
        len(leaky_present) == 0,
        leaky_present,
        "[] (no leaky columns)",
    )

    models_section = payload.get("models", {})
    if model_key not in models_section:
        raise KeyError(f"Champion model key {model_key!r} missing from {metrics_path}")

    champ = models_section[model_key]
    m = champ.get("metrics", {})
    risk = champ.get("risk_and_slice_diagnostics", {})
    class0 = risk.get("struggling_student_class0", {})
    slices = risk.get("slices", {})

    # 2. Headline classification thresholds
    for metric_key in ("accuracy", "f1_score", "roc_auc", "mcc"):
        val = float(m.get(metric_key, 0.0))
        floor = cfg[metric_key]
        _check(checks, f"metric_{metric_key}", val >= floor, round(val, 6), f">= {floor}")

    # 3. Macro-F1, calibration (Brier), and struggling-student (class 0) risk gates
    macro_f1 = float(risk.get("macro_f1", 0.0))
    _check(
        checks,
        "risk_macro_f1",
        macro_f1 >= cfg["macro_f1"],
        round(macro_f1, 4),
        f">= {cfg['macro_f1']}",
    )

    brier = float(risk.get("brier_score", 1.0))
    _check(
        checks,
        "calibration_brier_score",
        0.0 <= brier <= cfg["max_brier_score"],
        round(brier, 4),
        f"<= {cfg['max_brier_score']}",
    )

    c0_recall = float(class0.get("recall_specificity", 0.0))
    _check(
        checks,
        "struggling_student_class0_recall",
        c0_recall >= cfg["class0_recall"],
        round(c0_recall, 4),
        f">= {cfg['class0_recall']}",
    )

    c0_prec = float(class0.get("precision", 0.0))
    _check(
        checks,
        "struggling_student_class0_precision",
        c0_prec >= cfg["class0_precision"],
        round(c0_prec, 4),
        f">= {cfg['class0_precision']}",
    )

    # 4. Subgroup slice quality gates
    slice_aucs: list[float] = []
    for slice_name in REQUIRED_SLICES:
        s_metrics = slices.get(slice_name)
        _check(
            checks,
            f"slice_present_{slice_name}",
            s_metrics is not None,
            s_metrics is not None,
            "True",
        )
        if s_metrics is not None:
            s_f1 = float(s_metrics.get("f1_score", 0.0))
            s_mf1 = float(s_metrics.get("macro_f1", 0.0))
            s_auc = float(s_metrics.get("roc_auc", 0.0))
            slice_aucs.append(s_auc)
            _check(
                checks,
                f"slice_{slice_name}_f1",
                s_f1 >= cfg["slice_min_f1"],
                round(s_f1, 4),
                f">= {cfg['slice_min_f1']}",
            )
            _check(
                checks,
                f"slice_{slice_name}_macro_f1",
                s_mf1 >= cfg["slice_min_macro_f1"],
                round(s_mf1, 4),
                f">= {cfg['slice_min_macro_f1']}",
            )
            _check(
                checks,
                f"slice_{slice_name}_roc_auc",
                s_auc >= cfg["slice_min_roc_auc"],
                round(s_auc, 4),
                f">= {cfg['slice_min_roc_auc']}",
            )

    if len(slice_aucs) >= 2:
        auc_gap = max(slice_aucs) - min(slice_aucs)
        _check(
            checks,
            "slice_max_roc_auc_disparity",
            auc_gap <= cfg["max_slice_roc_auc_gap"],
            round(auc_gap, 4),
            f"<= {cfg['max_slice_roc_auc_gap']}",
        )

    # 5. Champion vs linear baseline superiority check
    if reference_key and reference_key in models_section:
        ref = models_section[reference_key]
        ref_m = ref.get("metrics", {})
        ref_risk = ref.get("risk_and_slice_diagnostics", {})
        beats_ref = (
            float(m.get("f1_score", 0.0)) > float(ref_m.get("f1_score", 0.0))
            and float(m.get("roc_auc", 0.0)) > float(ref_m.get("roc_auc", 0.0))
            and float(m.get("mcc", 0.0)) > float(ref_m.get("mcc", 0.0))
            and float(risk.get("brier_score", 1.0)) < float(ref_risk.get("brier_score", 1.0))
        )
        _check(
            checks,
            f"beats_reference_{reference_key}",
            beats_ref,
            {
                "rf_f1": round(float(m.get("f1_score", 0.0)), 4),
                "lr_f1": round(float(ref_m.get("f1_score", 0.0)), 4),
                "rf_brier": round(float(risk.get("brier_score", 1.0)), 4),
                "lr_brier": round(float(ref_risk.get("brier_score", 1.0)), 4),
            },
            "champion exceeds reference on F1, ROC-AUC, MCC, and Brier",
        )

    # 6. Live serialized artifact & latency gate (when materialized or required)
    rel_artifact = champ.get("artifact_path", "models/baseline_model.pkl")
    artifact_path = Path(rel_artifact)
    if not artifact_path.is_absolute():
        artifact_path = REPO_ROOT / artifact_path

    if require_artifact or artifact_path.exists():
        exists = artifact_path.exists() and artifact_path.stat().st_size > 0
        _check(
            checks,
            "artifact_exists_and_nonempty",
            exists,
            str(artifact_path),
            "non-empty .pkl file on disk",
        )
        if exists:
            model = joblib.load(artifact_path)
            fitted_cols = list(getattr(model, "feature_names_in_", []))
            _check(
                checks,
                "artifact_feature_names_contract",
                fitted_cols == list(ENGINEERED_FEATURES),
                fitted_cols,
                "exact 11 ENGINEERED_FEATURES",
            )
            sample_df = pd.DataFrame(
                [
                    {
                        "log_opportunity": 2.3979,
                        "attempt_count": 1.0,
                        "hint_count": 0.0,
                        "bottom_hint": 0.0,
                        "hint_ratio": 0.0,
                        "user_prev_acc": 0.85,
                        "problem_difficulty": 0.78,
                        "skill_difficulty": 0.72,
                        "is_skill_unmapped": 0.0,
                        "log_response_time": 9.21,
                        "is_rapid_guess": 0.0,
                    },
                    {
                        "log_opportunity": 0.6931,
                        "attempt_count": 5.0,
                        "hint_count": 3.0,
                        "bottom_hint": 1.0,
                        "hint_ratio": 1.0,
                        "user_prev_acc": 0.25,
                        "problem_difficulty": 0.30,
                        "skill_difficulty": 0.40,
                        "is_skill_unmapped": 0.0,
                        "log_response_time": 10.50,
                        "is_rapid_guess": 0.0,
                    },
                ]
            )[list(ENGINEERED_FEATURES)]

            # Warmup call
            _ = model.predict_proba(sample_df.iloc[:1])

            latencies_ms: list[float] = []
            for _ in range(15):
                t0 = time.perf_counter()
                probs = model.predict_proba(sample_df.iloc[:1])
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            median_ms = float(np.median(latencies_ms))
            probs_two = model.predict_proba(sample_df)
            valid_probs = bool(
                probs_two.shape == (2, 2)
                and np.all(np.isfinite(probs_two))
                and np.all((probs_two >= 0.0) & (probs_two <= 1.0))
                and np.allclose(probs_two.sum(axis=1), 1.0, atol=1e-6)
            )
            _check(
                checks,
                "artifact_calibrated_probabilities",
                valid_probs,
                probs_two[:, 1].round(4).tolist(),
                "probabilities in [0, 1] summing to 1.0",
            )
            _check(
                checks,
                "artifact_directional_sanity",
                float(probs_two[0, 1]) > float(probs_two[1, 1]),
                {
                    "mastery_prob": round(float(probs_two[0, 1]), 4),
                    "struggling_prob": round(float(probs_two[1, 1]), 4),
                },
                "mastery student P(correct=1) > struggling student P(correct=1)",
            )
            _check(
                checks,
                "artifact_single_prediction_latency_ms",
                median_ms <= cfg["max_single_latency_ms"],
                round(median_ms, 2),
                f"<= {cfg['max_single_latency_ms']} ms",
            )

    failures = [c for c in checks if not c["passed"]]
    return {
        "model_key": model_key,
        "metrics_path": str(metrics_path),
        "passed": len(failures) == 0,
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failed_checks": len(failures),
        "checks": checks,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 Model-Risk and Quality Gate checks."
    )
    parser.add_argument(
        "--metrics-path",
        default=str(DEFAULT_METRICS_PATH),
        help="Path to baseline_metrics.json",
    )
    parser.add_argument(
        "--model-key",
        default=DEFAULT_CHAMPION_KEY,
        help="Champion model key inside baseline_metrics.json",
    )
    parser.add_argument(
        "--require-artifact",
        action="store_true",
        help="Require models/baseline_model.pkl on disk and run live inference/latency checks",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the quality-gate result JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_baseline_metrics(
        metrics_path=args.metrics_path,
        model_key=args.model_key,
        require_artifact=args.require_artifact,
    )

    print(
        f"[QualityGate] Model={report['model_key']} | "
        f"{report['passed_checks']}/{report['total_checks']} checks passed"
    )
    for item in report["checks"]:
        icon = "PASS" if item["passed"] else "FAIL"
        print(
            f"  [{icon}] {item['check']:<40} actual={item['actual']} "
            f"(expected {item['expected']})"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    if not report["passed"]:
        print(
            f"[QualityGate] FAILED ({report['failed_checks']} check(s) violated).",
            file=sys.stderr,
        )
        return 1

    print("[QualityGate] ALL QUALITY GATES PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
