"""Offline unit tests for Milestone 5: Discord/Webhook Alerts, Azure CD & Cloud Config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from src.alerts import build_alert_payload, send_alert_webhook
from src.storage import reset_storage

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_alert_webhook_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    res = send_alert_webhook("Test", "No webhook set")
    assert res["sent"] is False
    assert res["reason"] == "no_webhook_configured"


def test_alert_webhook_formats_discord_and_generic_payloads(monkeypatch: pytest.MonkeyPatch):
    discord_url = "https://discord.com/api/webhooks/12345/abcdef"
    p_discord = build_alert_payload(
        "Drift Detected",
        "64% features drifted",
        severity="critical",
        fields={"Drift": "64%"},
        webhook_url=discord_url,
    )
    assert "embeds" in p_discord
    assert p_discord["embeds"][0]["title"] == "[CRITICAL] Drift Detected"
    assert p_discord["embeds"][0]["fields"][0]["name"] == "Drift"

    generic_url = "https://hooks.example.com/alert"
    p_generic = build_alert_payload(
        "Retrain Promoted",
        "F1=0.965",
        severity="info",
        webhook_url=generic_url,
    )
    assert p_generic["severity"] == "info"
    assert p_generic["service"] == "student-performance-predictor"

    fake_resp = MagicMock()
    fake_resp.status_code = 204
    fake_resp.raise_for_status.return_value = None
    post_mock = MagicMock(return_value=fake_resp)
    monkeypatch.setattr("src.alerts.requests.post", post_mock)

    out = send_alert_webhook("Test", "Ping", severity="info", webhook_url=discord_url)
    assert out["sent"] is True
    assert out["discord"] is True
    post_mock.assert_called_once()


def test_alerts_test_endpoint_respects_admin_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    reset_storage()

    import app.main as main_mod

    client = TestClient(main_mod.app)
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    assert client.post("/alerts/test").status_code == 401

    ok = client.post("/alerts/test", headers={"X-API-Key": "secret-key"})
    assert ok.status_code == 200
    assert ok.json()["sent"] is False

    monkeypatch.setenv("ADMIN_API_KEY", "")
    reset_storage()


def test_ci_workflow_preserves_all_5_jobs_and_adds_azure_cd():
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    doc = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    jobs = set(doc["jobs"].keys())
    expected = {
        "lint-and-test",
        "model-gate-and-registry",
        "docker-build",
        "docker-push",
        "metrics-comment",
        "deploy-azure",
    }
    assert expected.issubset(jobs)
    assert (REPO_ROOT / "infra" / "deploy_azure.sh").exists()
    assert (REPO_ROOT / "monitoring" / "grafana-cloud" / "alloy-config.river").exists()
    assert (REPO_ROOT / "docs" / "CLOUD_DEPLOYMENT_GUIDE.md").exists()
