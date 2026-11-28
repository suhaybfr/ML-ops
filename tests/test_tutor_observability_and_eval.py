"""Offline unit tests for Milestone 4: RAG Observability, Relevance Floor & Adversarial Eval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.artifact_store import ModelArtifactStore
from src.evaluate_tutor import evaluate_tutor
from src.storage import reset_storage


def test_metrics_exposes_milestone4_rag_observability_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("ADMIN_API_KEY", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod

    client = TestClient(main_mod.app)
    txt = client.get("/metrics").text
    assert "tutor_tokens_total" in txt
    assert "tutor_retrieval_top_similarity" in txt
    assert "tutor_chunks_filtered_total" in txt
    reset_storage()


def test_relevance_floor_and_token_usage_on_tutor_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import FastAPI
    import app.tutor as tutor_mod

    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("TUTOR_MIN_SIMILARITY", "0.99")
    tutor_mod.reset_store()

    from src.evaluate_tutor import _DeterministicOfflineEmbedding

    monkeypatch.setattr(tutor_mod, "_get_embed_model", lambda: _DeterministicOfflineEmbedding())
    monkeypatch.setattr(
        tutor_mod,
        "_get_llm",
        lambda api_key: SimpleNamespace(
            chat=lambda msgs: SimpleNamespace(
                message=SimpleNamespace(content="Let's think step by step — what clue do you see?"),
                raw={"usage_metadata": {"prompt_token_count": 88, "candidates_token_count": 19}},
            )
        ),
    )

    app = FastAPI()
    app.include_router(tutor_mod.router)
    client = TestClient(app)

    resp = client.post("/tutor/ask", json={"question": "totally unrelated banana query 999"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["chunks_filtered"] >= 1
    assert body["token_usage"] == {"prompt": 88, "completion": 19, "total": 107}
    assert body["top_similarity"] is not None
    tutor_mod.reset_store()


def test_evaluate_tutor_offline_gate_passes_all_suites(tmp_path: Path):
    out_file = tmp_path / "tutor_eval_report.json"
    report = evaluate_tutor(live=False, max_p95_ms=5000.0, output_path=out_file)
    assert out_file.exists()
    assert report["mode"] == "offline"
    assert report["passed"] is True
    assert report["passed_checks"] == report["total_checks"]
    assert report["total_checks"] == 9  # 5 jailbreak + 2 citation/floor + 1 hard-500 + 1 SLA


def test_delete_non_baseline_models_preserves_baseline(tmp_path: Path):
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "baseline_model.pkl").write_bytes(b"baseline")
    (art_dir / "retrained_v5.pkl").write_bytes(b"demo")
    store = ModelArtifactStore(connection_string="", local_dir=art_dir)
    deleted = store.delete_non_baseline_models()
    assert deleted == ["retrained_v5.pkl"]
    assert (art_dir / "baseline_model.pkl").exists()
