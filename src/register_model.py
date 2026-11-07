"""Phase 2 MLflow Model Registry management for the Champion model.

Registers, aliases/promotes, and verifies the accepted Random Forest champion
model (``random_forest_baseline_v2`` / ``runs:/<run_id>/model``) in the MLflow
Model Registry on DAGsHub (or an explicit local SQLite tracking URI for tests),
enforcing the Phase 2 Model-Risk & Quality Gate beforehand.

Usage:
    # Idempotently register & promote champion Random Forest on DAGsHub
    python -m src.register_model

    # Register a specific MLflow run ID
    python -m src.register_model --run-id 160e5351896f46e6b359316808177e86

    # Verify existing registered model without creating new versions (CI gate)
    python -m src.register_model --verify-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.utils import DEFAULT_EXPERIMENT, setup_mlflow
from src.validate_model import DEFAULT_METRICS_PATH, validate_baseline_metrics

REGISTRY_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "student_performance_predictor")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def find_champion_run(
    client: MlflowClient,
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_id: str | None = None,
    model_tag: str = "random_forest",
):
    """Locate the target training run while ignoring drift-check or nested runs."""
    if run_id:
        return client.get_run(run_id)

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise LookupError(f"MLflow experiment {experiment_name!r} not found.")

    # Primary lookup: tagged Phase 1/2 champion candidate
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.registry_candidate = 'true' and tags.model = '{model_tag}'",
        order_by=["start_time DESC"],
        max_results=5,
    )
    if not runs:
        # Secondary lookup: stage='train' for requested model family
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.stage = 'train' and tags.model = '{model_tag}'",
            order_by=["start_time DESC"],
            max_results=5,
        )
    if not runs:
        # Fallback by canonical run name
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.run_name = 'random_forest_baseline_v2'",
            order_by=["start_time DESC"],
            max_results=5,
        )
    if not runs:
        raise LookupError(
            f"No eligible {model_tag!r} training run found in experiment {experiment_name!r}."
        )
    return runs[0]


def get_current_champion_version(client: MlflowClient, model_name: str):
    """Return the active champion ModelVersion via alias or version tags, or None."""
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None

    # Check explicit deployment_status tag first, then legacy stage
    tagged = [
        mv
        for mv in versions
        if getattr(mv, "tags", {}).get("deployment_status") == CHAMPION_ALIAS
    ]
    if tagged:
        return sorted(tagged, key=lambda v: int(v.version), reverse=True)[0]

    prod = [mv for mv in versions if getattr(mv, "current_stage", "") == "Production"]
    if prod:
        return sorted(prod, key=lambda v: int(v.version), reverse=True)[0]

    return None


def get_champion_f1(client: MlflowClient, model_name: str) -> tuple[float, str | None]:
    """Return ``(f1_score, version_str)`` of the current registered champion."""
    mv = get_current_champion_version(client, model_name)
    if mv is None:
        return 0.0, None

    tag_f1 = getattr(mv, "tags", {}).get("f1_score")
    if tag_f1 is not None:
        try:
            return float(tag_f1), str(mv.version)
        except ValueError:
            pass

    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
            return float(run.data.metrics.get("f1_score", 0.0)), str(mv.version)
        except Exception:
            pass
    return 0.0, str(mv.version)


def _set_alias_and_tags(
    client: MlflowClient,
    model_name: str,
    version: str,
    alias: str,
    tags: dict[str, str],
) -> bool:
    """Apply MLflow 3 model alias (with graceful fallback on older servers) + version tags."""
    alias_applied = False
    try:
        client.set_registered_model_alias(model_name, alias, str(version))
        alias_applied = True
    except MlflowException as exc:
        print(
            f"[Registry] Notice: alias API unavailable on remote ({exc}); "
            "relying on version tags."
        )

    for key, val in tags.items():
        try:
            client.set_model_version_tag(model_name, str(version), str(key), str(val))
        except MlflowException as exc:
            print(f"[Registry] Notice: could not set version tag {key}={val}: {exc}")

    return alias_applied


def register_and_promote_champion(
    model_name: str = REGISTRY_MODEL_NAME,
    run_id: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
    enforce_gate: bool = True,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    artifact_subpath: str = "model",
) -> dict[str, Any]:
    """Idempotently register a validated model run and promote it if it beats/matches champion."""
    if enforce_gate:
        gate = validate_baseline_metrics(metrics_path=metrics_path, require_artifact=False)
        if not gate["passed"]:
            raise ValueError(
                f"Quality gate failed ({gate['failed_checks']} check(s)); "
                "refusing to register model."
            )

    active_uri = tracking_uri or setup_mlflow(experiment_name=experiment_name)
    mlflow.set_tracking_uri(active_uri)
    client = MlflowClient(tracking_uri=active_uri)

    run = find_champion_run(client, experiment_name=experiment_name, run_id=run_id)
    target_run_id = run.info.run_id
    candidate_f1 = float(run.data.metrics.get("f1_score", 0.0))
    candidate_auc = float(run.data.metrics.get("roc_auc", 0.0))
    candidate_brier = float(run.data.metrics.get("brier_score", 0.0))
    model_family = run.data.tags.get("model", "random_forest")

    prev_f1, prev_version = get_champion_f1(client, model_name)

    # Idempotency check: reuse existing version if this run_id is already registered
    existing_mv = None
    try:
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.run_id == target_run_id:
                existing_mv = mv
                break
    except MlflowException:
        existing_mv = None

    created_new_version = False
    if existing_mv is not None:
        mv = existing_mv
        print(
            f"[Registry] Reusing existing registered version v{mv.version} "
            f"for run {target_run_id} ({run.info.run_name})"
        )
    else:
        model_uri = f"runs:/{target_run_id}/{artifact_subpath}"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        created_new_version = True
        print(
            f"[Registry] Registered {model_name} v{mv.version} from {model_uri} "
            f"(F1={candidate_f1:.4f}, ROC-AUC={candidate_auc:.4f})"
        )

    # Promote if first champion, re-confirming current champion run, or strictly improving F1
    same_as_champion = prev_version is not None and str(mv.version) == str(prev_version)
    should_promote = prev_version is None or same_as_champion or (candidate_f1 > prev_f1)

    common_tags = {
        "model_family": model_family,
        "run_name": run.info.run_name or "",
        "feature_set": run.data.tags.get("feature_set", "engineered_11"),
        "f1_score": f"{candidate_f1:.6f}",
        "roc_auc": f"{candidate_auc:.6f}",
        "brier_score": f"{candidate_brier:.6f}",
    }

    if should_promote:
        # Demote previous champion tag if a newer version superseded it
        if prev_version is not None and str(prev_version) != str(mv.version):
            try:
                client.set_model_version_tag(
                    model_name, str(prev_version), "deployment_status", "archived_champion"
                )
            except MlflowException:
                pass

        alias_applied = _set_alias_and_tags(
            client,
            model_name=model_name,
            version=str(mv.version),
            alias=CHAMPION_ALIAS,
            tags={**common_tags, "deployment_status": CHAMPION_ALIAS},
        )
        promoted = True
        print(
            f"[Registry] Champion active: {model_name} v{mv.version} "
            f"(F1={candidate_f1:.4f} vs prior={prev_f1:.4f}, alias_set={alias_applied})"
        )
    else:
        alias_applied = _set_alias_and_tags(
            client,
            model_name=model_name,
            version=str(mv.version),
            alias=CHALLENGER_ALIAS,
            tags={**common_tags, "deployment_status": CHALLENGER_ALIAS},
        )
        promoted = False
        print(
            f"[Registry] Kept {model_name} v{mv.version} as challenger "
            f"(candidate F1={candidate_f1:.4f} <= champion v{prev_version} F1={prev_f1:.4f})"
        )

    return {
        "model_name": model_name,
        "version": str(mv.version),
        "run_id": target_run_id,
        "run_name": run.info.run_name,
        "f1_score": candidate_f1,
        "roc_auc": candidate_auc,
        "brier_score": candidate_brier,
        "created_new_version": created_new_version,
        "promoted": promoted,
        "alias_applied": alias_applied,
        "tracking_uri": active_uri,
    }


def verify_registered_model(
    model_name: str = REGISTRY_MODEL_NAME,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
    min_f1: float = 0.94,
) -> dict[str, Any]:
    """Verify that the Model Registry contains a valid champion version meeting F1 floor."""
    active_uri = tracking_uri or setup_mlflow(experiment_name=experiment_name)
    mlflow.set_tracking_uri(active_uri)
    client = MlflowClient(tracking_uri=active_uri)

    mv = get_current_champion_version(client, model_name)
    if mv is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise LookupError(f"No registered versions found for model {model_name!r}.")
        mv = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    f1, _ = get_champion_f1(client, model_name)
    if f1 == 0.0 and mv.run_id:
        run = client.get_run(mv.run_id)
        f1 = float(run.data.metrics.get("f1_score", 0.0))

    if f1 < min_f1:
        raise ValueError(
            f"Registered model {model_name} v{mv.version} F1={f1:.4f} is below floor {min_f1:.4f}."
        )

    info = {
        "model_name": model_name,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "status": mv.status,
        "source": mv.source,
        "f1_score": f1,
        "tracking_uri": active_uri,
    }
    print(
        f"[Registry] Verified {model_name} v{mv.version} "
        f"(run_id={mv.run_id}, status={mv.status}, F1={f1:.4f})"
    )
    return info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register or verify the Phase 2 champion model in MLflow Model Registry."
    )
    parser.add_argument(
        "--model-name",
        default=REGISTRY_MODEL_NAME,
        help=f"Registered model name (default: {REGISTRY_MODEL_NAME})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit MLflow run ID to register (defaults to tagged Random Forest champion)",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing registered champion without creating new versions",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip local baseline_metrics.json quality gate check before registration",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        verify_registered_model(
            model_name=args.model_name,
            experiment_name=args.experiment,
        )
        return 0

    register_and_promote_champion(
        model_name=args.model_name,
        run_id=args.run_id,
        experiment_name=args.experiment,
        enforce_gate=not args.skip_gate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
