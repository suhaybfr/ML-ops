"""Offline tests for Milestone 3: Security, Rate Limiting, Async Worker, Azure Blob & Shadow Mode."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from src.artifact_store import ModelArtifactStore
from src.security import reset_rate_limiters
from src.storage import reset_storage


def _valid_features() -> dict:
    return {
        "log_opportunity": 2.3979,
        "attempt_count": 1.0,
        "hint_count": 0.0,
        "bottom_hint": 0,
        "hint_ratio": 0.0,
        "user_prev_acc": 0.85,
        "problem_difficulty": 0.78,
        "skill_difficulty": 0.74,
        "is_skill_unmapped": 0,
        "log_response_time": 9.4,
        "is_rapid_guess": 0,
    }


def test_admin_api_key_protects_mutating_endpoints_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod

    client = TestClient(main_mod.app)

    # 1. When ADMIN_API_KEY is unset, endpoints are open for local dev / offline CI
    monkeypatch.setenv("ADMIN_API_KEY", "")
    assert client.get("/metrics").status_code == 200
    assert client.get("/health").json()["admin_auth_enabled"] is False

    # 2. When ADMIN_API_KEY is set, mutating endpoints + /metrics require auth
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret-admin-key")
    assert client.get("/health").json()["admin_auth_enabled"] is True

    for method, path in (
        ("GET", "/metrics"),
        ("POST", "/model/reload"),
        ("POST", "/telemetry/reset"),
        ("POST", "/model/shadow/clear"),
    ):
        r_unauth = client.request(method, path)
        assert r_unauth.status_code == 401, f"{method} {path} should require auth"

        r_wrong = client.request(method, path, headers={"X-API-Key": "wrong"})
        assert r_wrong.status_code == 401

    # 3. Both X-API-Key and Authorization: Bearer succeed
    assert client.get("/metrics", headers={"X-API-Key": "super-secret-admin-key"}).status_code == 200
    assert (
        client.get(
            "/metrics", headers={"Authorization": "Bearer super-secret-admin-key"}
        ).status_code
        == 200
    )
    assert (
        client.post("/telemetry/reset", headers={"X-API-Key": "super-secret-admin-key"}).status_code
        == 200
    )

    monkeypatch.setenv("ADMIN_API_KEY", "")
    reset_storage()


def test_tutor_rate_limiting_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TUTOR_ASK_RATE_LIMIT", "2")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    reset_rate_limiters()

    from fastapi import FastAPI
    from app.tutor import router as tutor_router

    app = FastAPI()
    app.include_router(tutor_router)
    client = TestClient(app)

    # First 2 requests pass rate limiter (hit missing API key -> 500)
    assert client.post("/tutor/ask", json={"question": "What is phishing?"}).status_code == 500
    assert client.post("/tutor/ask", json={"question": "What is MFA?"}).status_code == 500

    # 3rd request within 60s window is rejected with HTTP 429 + Retry-After
    r3 = client.post("/tutor/ask", json={"question": "What is SIFT?"})
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers

    reset_rate_limiters()


def test_model_artifact_store_local_and_mocked_azure_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src_pkl = tmp_path / "cand.pkl"
    src_pkl.write_bytes(b"fake-pickle-bytes")

    # Local fallback mode
    local_store = ModelArtifactStore(connection_string="", local_dir=tmp_path / "artifacts")
    assert local_store.backend_name == "local"
    up = local_store.upload_model(src_pkl, blob_name="v_test.pkl")
    assert up["backend"] == "local"
    assert "v_test.pkl" in local_store.list_models()
    dl = local_store.download_model("v_test.pkl", tmp_path / "downloaded.pkl")
    assert dl.read_bytes() == b"fake-pickle-bytes"

    # Mocked Azure Blob mode
    fake_blob_client = MagicMock()
    fake_blob_client.download_blob.return_value.readall.return_value = b"azure-bytes"
    fake_container = MagicMock()
    fake_container.get_blob_client.return_value = fake_blob_client

    az_store = ModelArtifactStore(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=abc==",
        container_name="models",
    )
    assert az_store.backend_name == "azure_blob"
    monkeypatch.setattr(az_store, "_get_container_client", lambda: fake_container)

    up_az = az_store.upload_model(src_pkl, blob_name="v_az.pkl", metadata={"f1": "0.96"})
    assert up_az["backend"] == "azure_blob"
    assert up_az["uri"] == "az://models/v_az.pkl"
    fake_blob_client.upload_blob.assert_called_once()

    dl_az = az_store.download_model("az://models/v_az.pkl", tmp_path / "from_az.pkl")
    assert dl_az.read_bytes() == b"azure-bytes"


def test_shadow_challenger_scoring_metrics_and_blue_green_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("ADMIN_API_KEY", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod

    # Create a deterministic DummyClassifier shadow model that always predicts 0 (disagrees with mastery)
    dummy = DummyClassifier(strategy="constant", constant=0)
    import numpy as np

    dummy.fit(np.zeros((2, 11)), [0, 1])
    chall_path = main_mod.REPO_ROOT / "models" / "challenger_test_shadow.pkl"
    joblib.dump(dummy, chall_path)
    baseline_bytes_before = (main_mod.REPO_ROOT / "models" / "baseline_model.pkl").read_bytes()

    try:
        client = TestClient(main_mod.app)
        main_mod.clear_shadow_model()

        # Deploy challenger into Shadow Mode
        dep = client.post(
            "/model/shadow/deploy",
            json={
                "model_path": str(chall_path),
                "version_label": "dummy_shadow_v1",
            },
        )
        assert dep.status_code == 200
        assert dep.json()["active"] is True
        assert dep.json()["challenger_version"] == "dummy_shadow_v1"

        # Score live traffic -> Champion still returns 1, shadow disagrees (predicts 0)
        pred_res = client.post("/predict/explain", json=_valid_features())
        assert pred_res.status_code == 200
        pbody = pred_res.json()
        assert pbody["prediction"] == 1  # Champion unaffected!

        # Submit delayed feedback so live F1 metrics populate
        fb = client.post(
            "/telemetry/feedback",
            json={
                "features": _valid_features(),
                "correct": 1,
                "session_id": pbody["interaction_id"],
            },
        )
        assert fb.status_code == 200

        st = client.get("/model/shadow/status").json()
        assert st["active"] is True
        assert st["predictions_scored"] >= 1
        assert st["shadow_disagreement_rate"] == 1.0
        assert st["labeled_samples"] == 1
        assert st["champion_live_f1"] == 1.0
        assert st["challenger_live_f1"] == 0.0

        # Verify Prometheus exposition includes Milestone 3 shadow metrics
        metrics_txt = client.get("/metrics").text
        assert "shadow_disagreement_rate" in metrics_txt
        assert "challenger_live_f1" in metrics_txt
        assert "shadow_model_active 1.0" in metrics_txt

        # Clear shadow mode cleanly
        clr = client.post("/model/shadow/clear")
        assert clr.status_code == 200
        assert clr.json()["active"] is False

        # Verify baseline_model.pkl was NEVER modified
        assert (main_mod.REPO_ROOT / "models" / "baseline_model.pkl").read_bytes() == (
            baseline_bytes_before
        )
    finally:
        if chall_path.exists():
            chall_path.unlink()
        main_mod.clear_shadow_model()
        reset_storage()


def test_async_retrain_job_returns_immediately_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("ADMIN_API_KEY", "")
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod
    import src.worker_retrain as worker_mod

    # Patch retrain_cycle inside worker_retrain so the async job completes fast & deterministically
    monkeypatch.setattr(
        worker_mod,
        "retrain_cycle",
        lambda **kwargs: {
            "status": "promoted",
            "retrained": True,
            "promoted": True,
            "quality_gate_passed": True,
            "candidate_f1": 0.965,
            "active_model_version": "rf_async_v99",
            "active_artifact_path": str(main_mod.REPO_ROOT / "models" / "baseline_model.pkl"),
        },
    )

    client = TestClient(main_mod.app)
    resp = client.post(
        "/model/retrain",
        json={"async_mode": True, "deploy_as_shadow": True, "force": True},
    )
    assert resp.status_code == 200
    job = resp.json()
    assert job["job_id"].startswith("retrain_")
    assert job["status"] in {"queued", "running", "completed"}

    # Poll until completed (should take <1s with mocked retrain_cycle)
    final = job
    for _ in range(40):
        final = client.get(f"/model/retrain/jobs/{job['job_id']}").json()
        if final["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert final["status"] == "completed"
    assert final["result"]["active_model_version"] == "rf_async_v99"
    assert client.get("/model/shadow/status").json()["active"] is True

    main_mod.clear_shadow_model()
    reset_storage()
