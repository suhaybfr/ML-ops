"""Out-of-band async retraining worker & CLI job (Milestone 3).

Decouples model retraining from the FastAPI inference request thread:
  1. As an in-process background job triggered by ``POST /model/retrain``
     (``async_mode=True``), returning ``job_id`` immediately while running
     retraining off the request thread.
  2. As a standalone CLI / Azure Container Apps Job / GitHub Actions worker:
     ``python -m src.worker_retrain --from-live --shadow --api-url http://...``

Uploads versioned ``.pkl`` artifacts to Azure Blob Storage (with local fallback),
respects ``ADMIN_API_KEY`` when notifying the serving API, and supports deploying
newly trained candidates into **Shadow Challenger Mode** before 100% promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from src.artifact_store import ModelArtifactStore  # noqa: E402
from src.retrain import (  # noqa: E402
    DEFAULT_CHALLENGER_PATH,
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_PROMOTED_PATH,
    retrain_cycle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _admin_headers() -> dict[str, str]:
    key = os.environ.get("ADMIN_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


def _notify_api_shadow_deploy(
    api_url: str,
    model_path: str,
    version_label: str,
    blob_uri: Optional[str] = None,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/model/shadow/deploy"
    resp = requests.post(
        url,
        headers=_admin_headers(),
        json={
            "model_path": model_path,
            "version_label": version_label,
            "artifact_uri": blob_uri,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _notify_api_reload(
    api_url: str,
    model_path: str,
    version_label: str,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/model/reload"
    resp = requests.post(
        url,
        headers=_admin_headers(),
        json={"model_path": model_path, "version_label": version_label},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def execute_retrain_job(
    *,
    from_live: bool = True,
    force: bool = False,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    deploy_as_shadow: bool = True,
    hot_reload: bool = False,
    output_dir: Optional[str | Path] = None,
    api_url: Optional[str] = None,
    on_shadow_ready: Optional[Callable[[str, str, Optional[str]], dict[str, Any]]] = None,
    on_hot_reload: Optional[Callable[[str, str], dict[str, Any]]] = None,
    on_complete: Optional[Callable[[], None]] = None,
    artifact_store: Optional[ModelArtifactStore] = None,
) -> dict[str, Any]:
    """Run drift-gated retraining, upload versioned artifact, and deploy shadow/champion."""
    drift_dir = Path(
        output_dir or os.environ.get("DRIFT_REPORTS_DIR", str(REPO_ROOT / "reports" / "drift"))
    )
    result = retrain_cycle(
        from_live=from_live,
        force=force,
        drift_threshold=drift_threshold,
        output_dir=drift_dir,
        status_path=drift_dir / "latest_status.json",
    )

    store = artifact_store or ModelArtifactStore()
    artifact_upload: Optional[dict[str, Any]] = None
    candidate_pkl = (
        Path(result["active_artifact_path"])
        if result.get("promoted") and result.get("active_artifact_path")
        else DEFAULT_CHALLENGER_PATH
    )
    if result.get("retrained") and candidate_pkl.exists():
        version_tag = str(result.get("active_model_version") or "challenger")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        blob_name = f"{version_tag}_{stamp}.pkl"
        try:
            artifact_upload = store.upload_model(
                candidate_pkl,
                blob_name=blob_name,
                metadata={
                    "model_version": version_tag,
                    "candidate_f1": str(result.get("candidate_f1", 0.0)),
                    "gate_passed": str(result.get("quality_gate_passed", False)),
                },
            )
        except Exception as exc:
            artifact_upload = {"backend": store.backend_name, "error": str(exc)}
    result["artifact_upload"] = artifact_upload

    blob_uri = (artifact_upload or {}).get("uri")
    version_label = str(result.get("active_model_version") or "random_forest_challenger")

    if result.get("retrained") and result.get("quality_gate_passed"):
        if deploy_as_shadow:
            shadow_pkl = str(
                DEFAULT_PROMOTED_PATH
                if result.get("promoted") and DEFAULT_PROMOTED_PATH.exists()
                else candidate_pkl
            )
            if on_shadow_ready is not None:
                result["shadow_result"] = on_shadow_ready(shadow_pkl, version_label, blob_uri)
            elif api_url:
                try:
                    result["shadow_result"] = _notify_api_shadow_deploy(
                        api_url, shadow_pkl, version_label, blob_uri
                    )
                except Exception as exc:
                    result["shadow_result"] = {"error": str(exc)}
        elif hot_reload and result.get("promoted") and result.get("active_artifact_path"):
            if on_hot_reload is not None:
                result["reload_result"] = on_hot_reload(
                    result["active_artifact_path"], version_label
                )
            elif api_url:
                try:
                    result["reload_result"] = _notify_api_reload(
                        api_url, result["active_artifact_path"], version_label
                    )
                except Exception as exc:
                    result["reload_result"] = {"error": str(exc)}

    if on_complete is not None:
        try:
            on_complete()
        except Exception:
            pass

    return result


def submit_async_retrain_job(**kwargs: Any) -> dict[str, Any]:
    """Queue an out-of-band background retraining job and return its ``job_id`` immediately."""
    job_id = f"retrain_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now_iso,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    with _jobs_lock:
        if len(_jobs) > 100:
            for old_id in list(_jobs.keys())[:25]:
                _jobs.pop(old_id, None)
        _jobs[job_id] = record

    def _runner() -> None:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            res = execute_retrain_job(**kwargs)
            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["result"] = res
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_runner, name=f"worker-{job_id}", daemon=True)
    thread.start()
    return dict(record)


def get_retrain_job(job_id: str) -> Optional[dict[str, Any]]:
    with _jobs_lock:
        found = _jobs.get(job_id)
        return dict(found) if found is not None else None


def list_retrain_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return [dict(v) for v in reversed(list(_jobs.values()))]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-band retraining worker for Azure Container Apps Jobs / CI."
    )
    parser.add_argument("--from-live", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    parser.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Deploy passing candidate into Shadow Challenger Mode instead of immediate swap",
    )
    parser.add_argument(
        "--hot-reload",
        action="store_true",
        help="Immediately hot-reload promoted model if --shadow is not set",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("SERVING_API_URL", "").strip() or None,
        help="Serving API base URL to notify for shadow deploy or hot-reload",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = execute_retrain_job(
        from_live=args.from_live,
        force=args.force,
        drift_threshold=args.drift_threshold,
        deploy_as_shadow=args.shadow,
        hot_reload=args.hot_reload,
        api_url=args.api_url,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
