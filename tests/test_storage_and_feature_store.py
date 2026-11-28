"""Offline unit tests for Milestone 1: Cloud Persistence & Online Feature Store.

Every test runs 100% offline (DATABASE_URL="", GOOGLE_API_KEY="", MLFLOW_OFFLINE="1")
using local CSV/JSON fallback and mocked psycopg/requests objects for the Postgres,
pgvector, and Supabase Storage code paths.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from llama_index.core.schema import Document

from app.pgvector_store import PgVectorTutorStore, format_pgvector
from src.features import ENGINEERED_FEATURES
from src.storage import (
    DEFAULT_PRIOR_ACCURACY,
    SCHEMA_SQL_PATH,
    LocalStorageBackend,
    PostgresStorageBackend,
    SupabaseObjectStorage,
    get_storage,
    reset_storage,
)


def _sample_interaction(session_id: str = "sess-1", correct: Any = None) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "source": "api_predict",
        "skill_name": "phishing_cyber_hygiene",
        "log_opportunity": 1.3863,
        "attempt_count": 2.0,
        "hint_count": 1.0,
        "bottom_hint": 0,
        "hint_ratio": 0.33,
        "user_prev_acc": 0.72,
        "problem_difficulty": 0.65,
        "skill_difficulty": 0.68,
        "is_skill_unmapped": 0,
        "log_response_time": 9.4,
        "is_rapid_guess": 0,
        "prediction": 1,
        "probability": 0.81,
        "needs_intervention": False,
        "model_version": "random_forest_baseline_v2",
        "correct": correct,
    }


def test_schema_sql_contains_all_milestone1_tables_and_pgvector():
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    for table in (
        "students",
        "assessment_items",
        "interaction_events",
        "drift_snapshots",
        "retrain_runs",
        "rag_documents",
        "rag_chunks",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "vector(384)" in sql


def test_local_storage_interactions_drift_retrain_and_feature_store(tmp_path: Path):
    store = LocalStorageBackend(
        buffer_path=tmp_path / "interactions.csv",
        drift_dir=tmp_path / "drift",
        feature_store_path=tmp_path / "feature_store.json",
    )

    # 1. Interactions + in-place outcome upsert
    assert store.append_interactions([_sample_interaction("s1"), _sample_interaction("s2")]) == 2
    assert store.upsert_interaction_outcome("s1", correct=0) is True
    assert store.upsert_interaction_outcome(
        "s3", correct=1, fallback_record=_sample_interaction("s3")
    ) is False

    all_df = store.load_interactions(labeled_only=False)
    labeled_df = store.load_interactions(labeled_only=True)
    assert len(all_df) == 3
    assert len(labeled_df) == 2
    assert list(ENGINEERED_FEATURES) == [c for c in ENGINEERED_FEATURES if c in all_df.columns]

    stats = store.get_interaction_stats(watermark_rows=1)
    assert stats["total_rows"] == 3
    assert stats["labeled_rows"] == 2
    assert stats["window_rows"] == 2

    # 2. Drift snapshot & retrain run persistence
    store.save_drift_snapshot({"drift_share": 0.45, "dataset_drifted": True})
    assert store.get_latest_drift_snapshot()["drift_share"] == 0.45

    store.save_retrain_run({"status": "promoted", "retrain_runs_total": 1, "watermark_rows": 3})
    assert store.get_latest_retrain_run()["status"] == "promoted"

    # 3. Online Feature Store: students expanding user_prev_acc & per-skill opportunity
    ctx0 = store.get_student_feature_context("stu_alice", skill_name="phishing_cyber_hygiene")
    assert ctx0["total_attempts"] == 0
    assert ctx0["user_prev_acc"] == DEFAULT_PRIOR_ACCURACY
    assert ctx0["opportunity"] == 1
    assert ctx0["log_opportunity"] == round(math.log1p(1), 6)

    ctx1 = store.record_student_outcome("stu_alice", "phishing_cyber_hygiene", correct=1)
    assert ctx1["total_attempts"] == 1
    assert ctx1["correct_attempts"] == 1
    assert ctx1["user_prev_acc"] == 1.0
    assert ctx1["opportunity"] == 2

    ctx2 = store.record_student_outcome("stu_alice", "phishing_cyber_hygiene", correct=0)
    assert ctx2["total_attempts"] == 2
    assert ctx2["correct_attempts"] == 1
    assert ctx2["user_prev_acc"] == 0.5
    assert ctx2["opportunity"] == 3

    ctx3 = store.record_student_outcome("stu_alice", "digital_privacy", correct=1)
    assert ctx3["total_attempts"] == 3
    assert ctx3["correct_attempts"] == 2
    assert abs(ctx3["user_prev_acc"] - (2 / 3)) < 1e-5
    assert ctx3["opportunity"] == 2  # digital_privacy had 1 completed, so next is 2

    # 4. Assessment items CRUD
    item = store.upsert_assessment_item(
        {
            "item_id": "dl_phish_01",
            "skill_name": "phishing_cyber_hygiene",
            "prompt": "Which sender detail is a phishing red flag?",
            "options": ["Official domain", "Misspelled look-alike domain"],
            "correct_answer": "Misspelled look-alike domain",
            "hints": ["Check the domain after @."],
            "bottom_hint_text": "Look-alike domains imitate trusted senders.",
            "problem_difficulty": 0.62,
            "skill_difficulty": 0.66,
        }
    )
    assert item["item_id"] == "dl_phish_01"
    assert store.get_assessment_item("dl_phish_01")["problem_difficulty"] == 0.62
    assert len(store.list_assessment_items(skill_name="phishing_cyber_hygiene")) == 1
    assert store.list_assessment_items(skill_name="other_skill") == []

    # 5. Reset clears telemetry, drift files, and optionally students
    reset_out = store.reset_telemetry(include_students=True)
    assert reset_out["reset"] is True
    assert reset_out["cleared_students"] is True
    assert store.get_interaction_stats()["total_rows"] == 0
    assert store.get_student_feature_context("stu_alice")["total_attempts"] == 0


def test_api_health_and_student_feature_store_endpoints_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod

    client = TestClient(main_mod.app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["storage_backend"] == "local"
    assert body["vector_backend"] == "chroma"

    r0 = client.get("/features/student/stu_bob", params={"skill_name": "source_credibility"})
    assert r0.status_code == 200
    assert r0.json()["opportunity"] == 1
    assert r0.json()["user_prev_acc"] == DEFAULT_PRIOR_ACCURACY

    r1 = client.post(
        "/features/student/stu_bob/outcome",
        json={"skill_name": "source_credibility", "correct": 0, "display_name": "Bob"},
    )
    assert r1.status_code == 200
    assert r1.json()["total_attempts"] == 1
    assert r1.json()["user_prev_acc"] == 0.0
    assert r1.json()["opportunity"] == 2

    reset_storage()


def _make_fake_pg_connection(fetchone_seq=None, fetchall_seq=None, rowcount: int = 1):
    cursor = MagicMock()
    cursor.rowcount = rowcount
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    if fetchone_seq is not None:
        cursor.fetchone.side_effect = list(fetchone_seq)
    if fetchall_seq is not None:
        cursor.fetchall.side_effect = list(fetchall_seq)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    return conn, cursor


def test_postgres_storage_backend_sql_and_factory(monkeypatch: pytest.MonkeyPatch):
    conn, cursor = _make_fake_pg_connection(
        fetchone_seq=[
            (1,),  # health_check SELECT 1
            (json.dumps({"drift_share": 0.55, "dataset_drifted": True}),),  # latest drift
            (json.dumps({"status": "promoted", "retrain_runs_total": 2}),),  # latest retrain
            (1, 1, {"phishing": 1}),  # record_student_outcome FOR UPDATE
            ("stu_1", "Alice", 2, 1, 0.5, {"phishing": 2}),  # get_or_create_student
        ]
    )
    monkeypatch.setattr(PostgresStorageBackend, "_connect", lambda self: conn)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.supabase.co:5432/postgres")
    reset_storage()

    backend = get_storage()
    assert isinstance(backend, PostgresStorageBackend)
    assert backend.health_check()["connected"] is True

    assert backend.append_interactions([_sample_interaction("pg-1", correct=1)]) == 1
    assert backend.upsert_interaction_outcome("pg-1", correct=0) is True
    backend.save_drift_snapshot({"tag": "t1", "drift_share": 0.55, "dataset_drifted": True})
    assert backend.get_latest_drift_snapshot()["drift_share"] == 0.55

    backend.save_retrain_run({"status": "promoted", "retrain_runs_total": 2})
    assert backend.get_latest_retrain_run()["status"] == "promoted"

    ctx = backend.record_student_outcome("stu_1", "phishing", correct=0, display_name="Alice")
    assert ctx["user_prev_acc"] == 0.5
    assert ctx["opportunity"] == 3

    # Clearing DATABASE_URL switches factory cleanly back to LocalStorageBackend
    monkeypatch.setenv("DATABASE_URL", "")
    reset_storage()
    assert isinstance(get_storage(), LocalStorageBackend)
    reset_storage()


def test_supabase_object_storage_rest_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert SupabaseObjectStorage.from_env() is None

    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc-key-123")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "course-pdfs")
    client = SupabaseObjectStorage.from_env()
    assert client is not None

    fake_resp = MagicMock()
    fake_resp.content = b"%PDF-1.4 test"
    fake_resp.raise_for_status.return_value = None
    post_mock = MagicMock(return_value=fake_resp)
    get_mock = MagicMock(return_value=fake_resp)
    monkeypatch.setattr("src.storage.requests.post", post_mock)
    monkeypatch.setattr("src.storage.requests.get", get_mock)

    path = client.upload_pdf("abc123", "module 1.pdf", b"%PDF-1.4 test")
    assert path == "course-pdfs/abc123/module 1.pdf"
    assert post_mock.call_args.kwargs["headers"]["x-upsert"] == "true"
    assert client.download_pdf(path) == b"%PDF-1.4 test"


def test_pgvector_store_upsert_query_and_hard_500_on_failure(monkeypatch: pytest.MonkeyPatch):
    assert format_pgvector([0.1, -0.25, 1.0]) == "[0.100000,-0.250000,1.000000]"

    conn, cursor = _make_fake_pg_connection(
        fetchone_seq=[(2,)],
        fetchall_seq=[
            [("builtin://digital_literacy/phishing", None, "Check sender domain", 0.9123)],
        ],
    )
    monkeypatch.setattr(PgVectorTutorStore, "_connect", lambda self: conn)
    store = PgVectorTutorStore(dsn="postgresql://fake", auto_init_schema=True)

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.01] * 384 for _ in texts]

    docs = [
        Document(
            text="Check the sender domain before clicking links.",
            id_="seed::phishing",
            metadata={"source": "builtin://digital_literacy/phishing"},
        )
    ]
    n_chunks = store.upsert_documents("digital_literacy_tutor", docs, _fake_embed)
    assert n_chunks >= 1
    assert store.count("digital_literacy_tutor") == 2

    hits = store.query("digital_literacy_tutor", "how to spot phishing?", _fake_embed, top_k=3)
    assert len(hits) == 1
    assert hits[0].source == "builtin://digital_literacy/phishing"
    assert hits[0].score == 0.9123

    # Hard HTTP 500 contract on embedding failure during PDF upload
    monkeypatch.setattr(
        "app.pgvector_store.read_pages",
        lambda data, name, digest: (
            1,
            [Document(text="Course handout text.", id_=f"{digest}/1", metadata={"page": 1})],
        ),
    )
    conn2, _ = _make_fake_pg_connection(fetchone_seq=[None])
    monkeypatch.setattr(PgVectorTutorStore, "_connect", lambda self: conn2)

    def _boom_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding exploded")

    with pytest.raises(HTTPException) as exc_info:
        store.upload_pdf("digital_literacy_tutor", "notes.pdf", b"%PDF-1.4 body", _boom_embed)
    assert exc_info.value.status_code == 500
    assert "PDF indexing failed" in exc_info.value.detail


def test_run_full_monitoring_handles_nan_targets_without_crashing(tmp_path: Path):
    """Unlabeled rows (NaN correct) mixed into reference/current must never crash retrain."""
    import numpy as np
    from src.induce_drift import generate_synthetic_baseline
    from src.monitoring import run_full_monitoring

    ref = generate_synthetic_baseline(n_rows=40, seed=1)
    cur = generate_synthetic_baseline(n_rows=20, seed=2)
    ref.loc[:10, "correct"] = np.nan
    cur.loc[:5, "correct"] = np.nan

    res = run_full_monitoring(
        reference_df=ref,
        current_df=cur,
        model_path=str(Path("models/baseline_model.pkl")),
        output_dir=str(tmp_path),
        tag="nan_target_safety",
        log_to_mlflow=False,
    )
    assert "drift_share" in res
    assert res["model_drift"] is not None
