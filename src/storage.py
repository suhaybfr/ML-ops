"""Cloud Persistence & Online Feature Store abstraction (Milestone 1).

Provides a unified repository interface backed by:
  - **Supabase Postgres + pgvector** (``PostgresStorageBackend``) when ``DATABASE_URL`` is set,
  - **Local CSV / JSON** (``LocalStorageBackend``) when ``DATABASE_URL`` is unset (100% offline CI).

Also provides ``SupabaseObjectStorage`` for persisting uploaded course PDFs in a
Supabase Storage bucket when ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY``
(or ``SUPABASE_KEY``) are configured.
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

import pandas as pd
import requests

from src.features import ENGINEERED_FEATURES, TARGET_COL

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_BUFFER_PATH = REPO_ROOT / "data" / "live" / "interactions.csv"
DEFAULT_DRIFT_DIR = REPO_ROOT / "reports" / "drift"
SCHEMA_SQL_PATH = REPO_ROOT / "sql" / "001_init_supabase.sql"

DEFAULT_PRIOR_ACCURACY = 0.70

TELEMETRY_COLUMNS: list[str] = [
    "timestamp",
    "session_id",
    "source",
    "skill_name",
    "opportunity",
    *ENGINEERED_FEATURES,
    "prediction",
    "probability",
    "needs_intervention",
    "model_version",
    TARGET_COL,
]


def _derive_opportunity(log_opportunity: float) -> int:
    try:
        return max(1, int(round(math.expm1(float(log_opportunity)))))
    except (OverflowError, ValueError, TypeError):
        return 1


def _normalize_interaction_row(rec: dict[str, Any], now_iso: str) -> dict[str, Any]:
    skill = rec.get("skill_name")
    if not skill:
        unmapped = int(float(rec.get("is_skill_unmapped", 0))) == 1
        skill = "Unknown" if unmapped else "Digital Literacy"
    raw_target = rec.get(TARGET_COL)
    target_val = None if raw_target is None or raw_target == "" else int(raw_target)
    opp_val = int(rec.get("opportunity") or _derive_opportunity(rec.get("log_opportunity", 0.0)))
    row: dict[str, Any] = {
        "timestamp": rec.get("timestamp") or now_iso,
        "session_id": rec.get("session_id") or "",
        "student_id": rec.get("student_id") or None,
        "item_id": rec.get("item_id") or None,
        "source": rec.get("source") or "api_predict",
        "skill_name": str(skill),
        "opportunity": opp_val,
        "prediction": int(rec["prediction"]),
        "probability": round(float(rec["probability"]), 6),
        "needs_intervention": int(bool(rec["needs_intervention"])),
        "model_version": str(rec.get("model_version") or "unknown"),
        TARGET_COL: target_val,
    }
    for col in ENGINEERED_FEATURES:
        row[col] = float(rec[col])
    return row


def is_postgres_configured() -> bool:
    """Return True when ``DATABASE_URL`` is set to a non-empty connection string."""
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _get_live_buffer_path() -> Path:
    env_path = os.environ.get("LIVE_INTERACTIONS_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_LIVE_BUFFER_PATH


def _get_drift_dir() -> Path:
    env_dir = os.environ.get("DRIFT_REPORTS_DIR", "").strip()
    return Path(env_dir) if env_dir else DEFAULT_DRIFT_DIR


# ---------------------------------------------------------------------------
# Local CSV / JSON fallback backend (offline CI & local dev)
# ---------------------------------------------------------------------------


class LocalStorageBackend:
    """Thread-safe local CSV + JSON storage backend used when ``DATABASE_URL`` is unset."""

    backend_name: str = "local"

    def __init__(
        self,
        buffer_path: Optional[Path] = None,
        drift_dir: Optional[Path] = None,
        feature_store_path: Optional[Path] = None,
    ) -> None:
        self._buffer_path = buffer_path
        self._drift_dir = drift_dir
        self._feature_store_path = feature_store_path
        self._lock = threading.RLock()

    @property
    def buffer_path(self) -> Path:
        return self._buffer_path if self._buffer_path is not None else _get_live_buffer_path()

    @property
    def drift_dir(self) -> Path:
        return self._drift_dir if self._drift_dir is not None else _get_drift_dir()

    @property
    def feature_store_path(self) -> Path:
        if self._feature_store_path is not None:
            return self._feature_store_path
        env_fs = os.environ.get("FEATURE_STORE_PATH", "").strip()
        if env_fs:
            return Path(env_fs)
        return self.buffer_path.parent / "feature_store.json"

    # -- Interactions --------------------------------------------------------

    def append_interactions(self, rows: Iterable[dict[str, Any]]) -> int:
        records = list(rows)
        if not records:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        formatted = []
        for rec in records:
            norm = _normalize_interaction_row(rec, now_iso)
            csv_row = {col: norm.get(col, "") for col in TELEMETRY_COLUMNS}
            if csv_row[TARGET_COL] is None:
                csv_row[TARGET_COL] = ""
            formatted.append(csv_row)

        target = self.buffer_path
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_header = not target.exists() or target.stat().st_size == 0
            with open(target, "a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerows(formatted)
        return len(formatted)

    def upsert_interaction_outcome(
        self,
        interaction_id: str,
        correct: int,
        fallback_record: Optional[dict[str, Any]] = None,
    ) -> bool:
        target = self.buffer_path
        with self._lock:
            if target.exists() and target.stat().st_size > 0 and interaction_id:
                with open(target, "r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                matched = False
                for r in reversed(rows):
                    if r.get("session_id") == interaction_id:
                        r[TARGET_COL] = str(int(correct))
                        matched = True
                        break
                if matched:
                    with open(target, "w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
                        writer.writeheader()
                        writer.writerows(rows)
                    return True

        if fallback_record is not None:
            rec = dict(fallback_record)
            rec["session_id"] = interaction_id
            rec[TARGET_COL] = int(correct)
            self.append_interactions([rec])
        return False

    def load_interactions(
        self,
        labeled_only: bool = False,
        since_row: int = 0,
    ) -> pd.DataFrame:
        target = self.buffer_path
        if not target.exists() or target.stat().st_size == 0:
            return pd.DataFrame(columns=TELEMETRY_COLUMNS)

        with self._lock:
            df = pd.read_csv(target)

        if since_row > 0:
            df = df.iloc[since_row:].copy()

        for col in ENGINEERED_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if TARGET_COL in df.columns:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

        if labeled_only and TARGET_COL in df.columns:
            df = df[df[TARGET_COL].isin([0, 1])].copy()
            df[TARGET_COL] = df[TARGET_COL].astype(int)

        return df.reset_index(drop=True)

    def get_interaction_stats(self, watermark_rows: int = 0) -> dict[str, Any]:
        df = self.load_interactions(labeled_only=False)
        total = int(len(df))
        if total == 0:
            return {
                "total_rows": 0,
                "labeled_rows": 0,
                "window_rows": 0,
                "intervention_rate": 0.0,
                "buffer_path": str(self.buffer_path),
            }

        labeled = int(df[TARGET_COL].isin([0, 1]).sum()) if TARGET_COL in df.columns else 0
        window_df = df.iloc[watermark_rows:] if 0 < watermark_rows < total else df
        interventions = (
            float(pd.to_numeric(window_df["needs_intervention"], errors="coerce").fillna(0).mean())
            if "needs_intervention" in window_df.columns and len(window_df) > 0
            else 0.0
        )
        return {
            "total_rows": total,
            "labeled_rows": labeled,
            "window_rows": int(len(df) - min(watermark_rows, total)),
            "intervention_rate": round(interventions, 4),
            "buffer_path": str(self.buffer_path),
        }

    def reset_telemetry(
        self, drift_dir: Optional[Path] = None, include_students: bool = False
    ) -> dict[str, Any]:
        target = self.buffer_path
        with self._lock:
            if target.exists():
                target.unlink()
            if include_students and self.feature_store_path.exists():
                doc = self._read_fs_doc()
                doc["students"] = {}
                self._write_fs_doc(doc)

        removed_files: list[str] = []
        dpath = drift_dir if drift_dir is not None else self.drift_dir
        if dpath.exists():
            for item in dpath.iterdir():
                if item.is_file() and item.suffix in {".json", ".html", ".csv"}:
                    item.unlink()
                    removed_files.append(item.name)

        return {
            "reset": True,
            "buffer_path": str(target),
            "cleared_drift_files": removed_files,
            "cleared_students": include_students,
        }

    # -- Drift & Retrain snapshots ------------------------------------------

    def save_drift_snapshot(
        self, payload: dict[str, Any], drift_dir: Optional[Path] = None
    ) -> dict[str, Any]:
        dpath = drift_dir if drift_dir is not None else self.drift_dir
        dpath.mkdir(parents=True, exist_ok=True)
        latest = dpath / "latest_drift_summary.json"
        latest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def get_latest_drift_snapshot(self, drift_dir: Optional[Path] = None) -> dict[str, Any]:
        dpath = drift_dir if drift_dir is not None else self.drift_dir
        latest = dpath / "latest_drift_summary.json"
        if not latest.exists():
            return {}
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_retrain_run(
        self, payload: dict[str, Any], status_path: Optional[Path] = None
    ) -> dict[str, Any]:
        target = status_path if status_path is not None else (self.drift_dir / "latest_status.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return payload

    def get_latest_retrain_run(self, status_path: Optional[Path] = None) -> dict[str, Any]:
        target = status_path if status_path is not None else (self.drift_dir / "latest_status.json")
        if not target.exists():
            return {}
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # -- Online Feature Store (students & assessment_items) -----------------

    def _read_fs_doc(self) -> dict[str, Any]:
        path = self.feature_store_path
        if not path.exists():
            return {"students": {}, "assessment_items": {}}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc.setdefault("students", {})
            doc.setdefault("assessment_items", {})
            return doc
        except Exception:
            return {"students": {}, "assessment_items": {}}

    def _write_fs_doc(self, doc: dict[str, Any]) -> None:
        path = self.feature_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")

    def get_or_create_student(
        self, student_id: str, display_name: Optional[str] = None
    ) -> dict[str, Any]:
        sid = student_id.strip()
        if not sid:
            raise ValueError("student_id must be non-empty")
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            doc = self._read_fs_doc()
            student = doc["students"].get(sid)
            if student is None:
                student = {
                    "student_id": sid,
                    "display_name": display_name or sid,
                    "total_attempts": 0,
                    "correct_attempts": 0,
                    "user_prev_acc": DEFAULT_PRIOR_ACCURACY,
                    "skill_opportunities": {},
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
                doc["students"][sid] = student
                self._write_fs_doc(doc)
            elif display_name and not student.get("display_name"):
                student["display_name"] = display_name
                student["updated_at"] = now_iso
                self._write_fs_doc(doc)
            return dict(student)

    def get_student_feature_context(
        self, student_id: str, skill_name: Optional[str] = None
    ) -> dict[str, Any]:
        student = self.get_or_create_student(student_id)
        skill_opps = dict(student.get("skill_opportunities") or {})
        prior_opp = int(skill_opps.get(skill_name, 0)) if skill_name else 0
        next_opportunity = prior_opp + 1
        return {
            "student_id": student["student_id"],
            "display_name": student.get("display_name") or student["student_id"],
            "skill_name": skill_name,
            "total_attempts": int(student["total_attempts"]),
            "correct_attempts": int(student["correct_attempts"]),
            "user_prev_acc": round(float(student["user_prev_acc"]), 6),
            "opportunity": next_opportunity,
            "log_opportunity": round(math.log1p(next_opportunity), 6),
            "skill_opportunities": skill_opps,
        }

    def record_student_outcome(
        self,
        student_id: str,
        skill_name: str,
        correct: int,
        display_name: Optional[str] = None,
    ) -> dict[str, Any]:
        sid = student_id.strip()
        if not sid:
            raise ValueError("student_id must be non-empty")
        corr_int = 1 if int(correct) else 0
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            doc = self._read_fs_doc()
            student = doc["students"].get(sid)
            if student is None:
                student = {
                    "student_id": sid,
                    "display_name": display_name or sid,
                    "total_attempts": 0,
                    "correct_attempts": 0,
                    "user_prev_acc": DEFAULT_PRIOR_ACCURACY,
                    "skill_opportunities": {},
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            student["total_attempts"] = int(student["total_attempts"]) + 1
            student["correct_attempts"] = int(student["correct_attempts"]) + corr_int
            student["user_prev_acc"] = round(
                student["correct_attempts"] / max(student["total_attempts"], 1), 6
            )
            skill_opps = dict(student.get("skill_opportunities") or {})
            skill_key = (skill_name or "Digital Literacy").strip() or "Digital Literacy"
            skill_opps[skill_key] = int(skill_opps.get(skill_key, 0)) + 1
            student["skill_opportunities"] = skill_opps
            student["updated_at"] = now_iso
            doc["students"][sid] = student
            self._write_fs_doc(doc)
        return self.get_student_feature_context(sid, skill_name=skill_key)

    def upsert_assessment_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item["item_id"]).strip()
        if not item_id:
            raise ValueError("item_id must be non-empty")
        record = {
            "item_id": item_id,
            "skill_name": str(item["skill_name"]),
            "prompt": str(item["prompt"]),
            "options": list(item.get("options") or []),
            "correct_answer": str(item["correct_answer"]),
            "hints": list(item.get("hints") or []),
            "bottom_hint_text": item.get("bottom_hint_text"),
            "problem_difficulty": float(item.get("problem_difficulty", 0.70)),
            "skill_difficulty": float(item.get("skill_difficulty", 0.69)),
            "is_skill_unmapped": int(item.get("is_skill_unmapped", 0)),
            "metadata": dict(item.get("metadata") or {}),
        }
        with self._lock:
            doc = self._read_fs_doc()
            doc["assessment_items"][item_id] = record
            self._write_fs_doc(doc)
        return dict(record)

    def get_assessment_item(self, item_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            doc = self._read_fs_doc()
            found = doc["assessment_items"].get(item_id)
            return dict(found) if found is not None else None

    def list_assessment_items(self, skill_name: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            doc = self._read_fs_doc()
            items = list(doc["assessment_items"].values())
        if skill_name:
            items = [it for it in items if it.get("skill_name") == skill_name]
        return sorted(items, key=lambda it: str(it.get("item_id", "")))

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "connected": True,
            "buffer_path": str(self.buffer_path),
        }


# ---------------------------------------------------------------------------
# Supabase Postgres + pgvector backend
# ---------------------------------------------------------------------------


class PostgresStorageBackend:
    """Supabase Postgres repository & Online Feature Store backed by ``psycopg`` v3."""

    backend_name: str = "postgres"

    def __init__(self, dsn: str, auto_init_schema: bool = True) -> None:
        self.dsn = dsn
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        if auto_init_schema:
            self.ensure_schema()

    def _connect(self):
        import psycopg

        # prepare_threshold=None ensures compatibility with Supabase PgBouncer transaction pooler
        return psycopg.connect(self.dsn, prepare_threshold=None)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            sql_text = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_text)
                conn.commit()
            self._schema_ready = True

    # -- Interactions --------------------------------------------------------

    def append_interactions(self, rows: Iterable[dict[str, Any]]) -> int:
        records = list(rows)
        if not records:
            return 0
        self.ensure_schema()
        now_iso = datetime.now(timezone.utc).isoformat()
        insert_sql = """
            INSERT INTO interaction_events (
                timestamp, session_id, student_id, item_id, source, skill_name, opportunity,
                log_opportunity, attempt_count, hint_count, bottom_hint, hint_ratio,
                user_prev_acc, problem_difficulty, skill_difficulty, is_skill_unmapped,
                log_response_time, is_rapid_guess,
                prediction, probability, needs_intervention, model_version, correct
            ) VALUES (
                %(timestamp)s, %(session_id)s, %(student_id)s, %(item_id)s, %(source)s,
                %(skill_name)s, %(opportunity)s,
                %(log_opportunity)s, %(attempt_count)s, %(hint_count)s, %(bottom_hint)s,
                %(hint_ratio)s, %(user_prev_acc)s, %(problem_difficulty)s, %(skill_difficulty)s,
                %(is_skill_unmapped)s, %(log_response_time)s, %(is_rapid_guess)s,
                %(prediction)s, %(probability)s, %(needs_intervention)s,
                %(model_version)s, %(correct)s
            )
        """
        formatted = [_normalize_interaction_row(rec, now_iso) for rec in records]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, formatted)
            conn.commit()
        return len(formatted)

    def upsert_interaction_outcome(
        self,
        interaction_id: str,
        correct: int,
        fallback_record: Optional[dict[str, Any]] = None,
    ) -> bool:
        self.ensure_schema()
        if interaction_id:
            update_sql = """
                UPDATE interaction_events
                SET correct = %s
                WHERE id = (
                    SELECT id FROM interaction_events
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT 1
                )
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(update_sql, (int(correct), interaction_id))
                    updated = cur.rowcount > 0
                conn.commit()
            if updated:
                return True

        if fallback_record is not None:
            rec = dict(fallback_record)
            rec["session_id"] = interaction_id
            rec[TARGET_COL] = int(correct)
            self.append_interactions([rec])
        return False

    def load_interactions(
        self,
        labeled_only: bool = False,
        since_row: int = 0,
    ) -> pd.DataFrame:
        self.ensure_schema()
        cols_sql = ", ".join(TELEMETRY_COLUMNS)
        offset = max(int(since_row), 0)
        query = (
            f"SELECT {cols_sql} FROM interaction_events "
            "ORDER BY id ASC OFFSET %s"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (offset,))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=TELEMETRY_COLUMNS)

        df = pd.DataFrame(rows, columns=TELEMETRY_COLUMNS)
        for col in ENGINEERED_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if labeled_only:
            df = df[df[TARGET_COL].isin([0, 1])].copy()
            df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df.reset_index(drop=True)

    def get_interaction_stats(self, watermark_rows: int = 0) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::bigint AS total_rows,
                        COUNT(*) FILTER (WHERE correct IN (0, 1))::bigint AS labeled_rows
                    FROM interaction_events
                    """
                )
                total, labeled = cur.fetchone()
                total = int(total or 0)
                labeled = int(labeled or 0)

                if total == 0:
                    return {
                        "total_rows": 0,
                        "labeled_rows": 0,
                        "window_rows": 0,
                        "intervention_rate": 0.0,
                        "buffer_path": "postgres://interaction_events",
                    }

                offset = watermark_rows if 0 < watermark_rows < total else 0
                cur.execute(
                    """
                    SELECT AVG(needs_intervention)::double precision
                    FROM (
                        SELECT needs_intervention FROM interaction_events
                        ORDER BY id ASC
                        OFFSET %s
                    ) sub
                    """,
                    (offset,),
                )
                avg_int = cur.fetchone()[0]
        return {
            "total_rows": total,
            "labeled_rows": labeled,
            "window_rows": int(total - min(watermark_rows, total)),
            "intervention_rate": round(float(avg_int or 0.0), 4),
            "buffer_path": "postgres://interaction_events",
        }

    def reset_telemetry(
        self, drift_dir: Optional[Path] = None, include_students: bool = False
    ) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE interaction_events RESTART IDENTITY")
                cur.execute("TRUNCATE TABLE drift_snapshots RESTART IDENTITY")
                cur.execute("TRUNCATE TABLE retrain_runs RESTART IDENTITY")
                if include_students:
                    cur.execute("DELETE FROM students")
            conn.commit()
        removed_files: list[str] = []
        if drift_dir is not None and drift_dir.exists():
            for item in drift_dir.iterdir():
                if item.is_file() and item.suffix in {".json", ".html", ".csv"}:
                    item.unlink()
                    removed_files.append(item.name)
        return {
            "reset": True,
            "buffer_path": "postgres://interaction_events",
            "cleared_drift_files": removed_files,
            "cleared_students": include_students,
        }

    # -- Drift & Retrain snapshots ------------------------------------------

    def save_drift_snapshot(
        self, payload: dict[str, Any], drift_dir: Optional[Path] = None
    ) -> dict[str, Any]:
        self.ensure_schema()
        md = payload.get("model_drift") or {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO drift_snapshots (
                        tag, engine, drift_share, dataset_drifted,
                        drifted_features_count, model_drift_detected, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        payload.get("tag"),
                        payload.get("engine"),
                        float(payload.get("drift_share", 0.0)),
                        bool(payload.get("dataset_drifted", False)),
                        int(payload.get("drifted_features_count", 0)),
                        bool(md.get("model_drift_detected", False)),
                        json.dumps(payload, default=str),
                    ),
                )
            conn.commit()
        return payload

    def get_latest_drift_snapshot(self, drift_dir: Optional[Path] = None) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM drift_snapshots ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
        if not row:
            return {}
        val = row[0]
        return val if isinstance(val, dict) else json.loads(val)

    def save_retrain_run(
        self, payload: dict[str, Any], status_path: Optional[Path] = None
    ) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO retrain_runs (
                        status, retrained, promoted, retrain_runs_total,
                        retrain_promotions_total, retrain_gate_failures_total,
                        watermark_rows, candidate_f1, active_model_version,
                        active_artifact_path, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        str(payload.get("status", "idle")),
                        bool(payload.get("retrained", False)),
                        bool(payload.get("promoted", False)),
                        int(payload.get("retrain_runs_total", 0)),
                        int(payload.get("retrain_promotions_total", 0)),
                        int(payload.get("retrain_gate_failures_total", 0)),
                        int(payload.get("watermark_rows", 0)),
                        float(payload.get("candidate_f1", 0.0)),
                        payload.get("active_model_version"),
                        payload.get("active_artifact_path"),
                        json.dumps(payload, default=str),
                    ),
                )
            conn.commit()
        return payload

    def get_latest_retrain_run(self, status_path: Optional[Path] = None) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM retrain_runs ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
        if not row:
            return {}
        val = row[0]
        return val if isinstance(val, dict) else json.loads(val)

    # -- Online Feature Store (students & assessment_items) -----------------

    def get_or_create_student(
        self, student_id: str, display_name: Optional[str] = None
    ) -> dict[str, Any]:
        sid = student_id.strip()
        if not sid:
            raise ValueError("student_id must be non-empty")
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO students (student_id, display_name, user_prev_acc)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (student_id) DO NOTHING
                    """,
                    (sid, display_name or sid, DEFAULT_PRIOR_ACCURACY),
                )
                cur.execute(
                    """
                    SELECT student_id, display_name, total_attempts, correct_attempts,
                           user_prev_acc, skill_opportunities
                    FROM students WHERE student_id = %s
                    """,
                    (sid,),
                )
                row = cur.fetchone()
            conn.commit()
        skill_opps = row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}")
        return {
            "student_id": row[0],
            "display_name": row[1] or row[0],
            "total_attempts": int(row[2]),
            "correct_attempts": int(row[3]),
            "user_prev_acc": round(float(row[4]), 6),
            "skill_opportunities": skill_opps,
        }

    def get_student_feature_context(
        self, student_id: str, skill_name: Optional[str] = None
    ) -> dict[str, Any]:
        student = self.get_or_create_student(student_id)
        skill_opps = dict(student.get("skill_opportunities") or {})
        prior_opp = int(skill_opps.get(skill_name, 0)) if skill_name else 0
        next_opportunity = prior_opp + 1
        return {
            "student_id": student["student_id"],
            "display_name": student["display_name"],
            "skill_name": skill_name,
            "total_attempts": student["total_attempts"],
            "correct_attempts": student["correct_attempts"],
            "user_prev_acc": student["user_prev_acc"],
            "opportunity": next_opportunity,
            "log_opportunity": round(math.log1p(next_opportunity), 6),
            "skill_opportunities": skill_opps,
        }

    def record_student_outcome(
        self,
        student_id: str,
        skill_name: str,
        correct: int,
        display_name: Optional[str] = None,
    ) -> dict[str, Any]:
        sid = student_id.strip()
        if not sid:
            raise ValueError("student_id must be non-empty")
        corr_int = 1 if int(correct) else 0
        skill_key = (skill_name or "Digital Literacy").strip() or "Digital Literacy"
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO students (student_id, display_name, user_prev_acc)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (student_id) DO NOTHING
                    """,
                    (sid, display_name or sid, DEFAULT_PRIOR_ACCURACY),
                )
                cur.execute(
                    """
                    SELECT total_attempts, correct_attempts, skill_opportunities
                    FROM students WHERE student_id = %s FOR UPDATE
                    """,
                    (sid,),
                )
                total_att, corr_att, raw_opps = cur.fetchone()
                new_total = int(total_att) + 1
                new_corr = int(corr_att) + corr_int
                new_acc = round(new_corr / max(new_total, 1), 6)
                opps = raw_opps if isinstance(raw_opps, dict) else json.loads(raw_opps or "{}")
                opps[skill_key] = int(opps.get(skill_key, 0)) + 1
                cur.execute(
                    """
                    UPDATE students
                    SET total_attempts = %s,
                        correct_attempts = %s,
                        user_prev_acc = %s,
                        skill_opportunities = %s::jsonb,
                        updated_at = NOW()
                    WHERE student_id = %s
                    """,
                    (new_total, new_corr, new_acc, json.dumps(opps), sid),
                )
            conn.commit()
        return self.get_student_feature_context(sid, skill_name=skill_key)

    def upsert_assessment_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item["item_id"]).strip()
        if not item_id:
            raise ValueError("item_id must be non-empty")
        self.ensure_schema()
        record = {
            "item_id": item_id,
            "skill_name": str(item["skill_name"]),
            "prompt": str(item["prompt"]),
            "options": list(item.get("options") or []),
            "correct_answer": str(item["correct_answer"]),
            "hints": list(item.get("hints") or []),
            "bottom_hint_text": item.get("bottom_hint_text"),
            "problem_difficulty": float(item.get("problem_difficulty", 0.70)),
            "skill_difficulty": float(item.get("skill_difficulty", 0.69)),
            "is_skill_unmapped": int(item.get("is_skill_unmapped", 0)),
            "metadata": dict(item.get("metadata") or {}),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO assessment_items (
                        item_id, skill_name, prompt, options, correct_answer,
                        hints, bottom_hint_text, problem_difficulty, skill_difficulty,
                        is_skill_unmapped, metadata
                    ) VALUES (
                        %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (item_id) DO UPDATE SET
                        skill_name = EXCLUDED.skill_name,
                        prompt = EXCLUDED.prompt,
                        options = EXCLUDED.options,
                        correct_answer = EXCLUDED.correct_answer,
                        hints = EXCLUDED.hints,
                        bottom_hint_text = EXCLUDED.bottom_hint_text,
                        problem_difficulty = EXCLUDED.problem_difficulty,
                        skill_difficulty = EXCLUDED.skill_difficulty,
                        is_skill_unmapped = EXCLUDED.is_skill_unmapped,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        record["item_id"],
                        record["skill_name"],
                        record["prompt"],
                        json.dumps(record["options"]),
                        record["correct_answer"],
                        json.dumps(record["hints"]),
                        record["bottom_hint_text"],
                        record["problem_difficulty"],
                        record["skill_difficulty"],
                        record["is_skill_unmapped"],
                        json.dumps(record["metadata"]),
                    ),
                )
            conn.commit()
        return record

    def get_assessment_item(self, item_id: str) -> Optional[dict[str, Any]]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT item_id, skill_name, prompt, options, correct_answer,
                           hints, bottom_hint_text, problem_difficulty, skill_difficulty,
                           is_skill_unmapped, metadata
                    FROM assessment_items WHERE item_id = %s
                    """,
                    (item_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def list_assessment_items(self, skill_name: Optional[str] = None) -> list[dict[str, Any]]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                if skill_name:
                    cur.execute(
                        """
                        SELECT item_id, skill_name, prompt, options, correct_answer,
                               hints, bottom_hint_text, problem_difficulty, skill_difficulty,
                               is_skill_unmapped, metadata
                        FROM assessment_items
                        WHERE skill_name = %s
                        ORDER BY item_id ASC
                        """,
                        (skill_name,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT item_id, skill_name, prompt, options, correct_answer,
                               hints, bottom_hint_text, problem_difficulty, skill_difficulty,
                               is_skill_unmapped, metadata
                        FROM assessment_items
                        ORDER BY item_id ASC
                        """
                    )
                rows = cur.fetchall()
        return [self._row_to_item(r) for r in rows]

    @staticmethod
    def _row_to_item(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "item_id": row[0],
            "skill_name": row[1],
            "prompt": row[2],
            "options": row[3] if isinstance(row[3], list) else json.loads(row[3] or "[]"),
            "correct_answer": row[4],
            "hints": row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]"),
            "bottom_hint_text": row[6],
            "problem_difficulty": float(row[7]),
            "skill_difficulty": float(row[8]),
            "is_skill_unmapped": int(row[9]),
            "metadata": row[10] if isinstance(row[10], dict) else json.loads(row[10] or "{}"),
        }

    def health_check(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return {"backend": self.backend_name, "connected": True}
        except Exception as exc:
            return {"backend": self.backend_name, "connected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Supabase Storage REST helper (course PDF object persistence)
# ---------------------------------------------------------------------------


class SupabaseObjectStorage:
    """Zero-dependency Supabase Storage REST client for course PDF persistence."""

    def __init__(
        self,
        supabase_url: str,
        api_key: str,
        bucket: str = "course-pdfs",
        timeout: int = 30,
    ) -> None:
        self.base_url = supabase_url.rstrip("/")
        self.api_key = api_key
        self.bucket = bucket
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> Optional["SupabaseObjectStorage"]:
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.environ.get("SUPABASE_KEY", "").strip()
        )
        if not url or not key:
            return None
        bucket = (
            os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
            or os.environ.get("SUPABASE_PDF_BUCKET", "").strip()
            or "course-pdfs"
        )
        return cls(supabase_url=url, api_key=key, bucket=bucket)

    def _headers(self, content_type: Optional[str] = None, upsert: bool = False) -> dict[str, str]:
        headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if upsert:
            headers["x-upsert"] = "true"
        return headers

    def upload_pdf(self, digest: str, filename: str, data: bytes) -> str:
        object_path = f"{digest}/{filename}"
        encoded = quote(object_path, safe="/")
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{encoded}"
        resp = requests.post(
            url,
            headers=self._headers(content_type="application/pdf", upsert=True),
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return f"{self.bucket}/{object_path}"

    def download_pdf(self, storage_path: str) -> bytes:
        path = storage_path.lstrip("/")
        if not path.startswith(f"{self.bucket}/"):
            path = f"{self.bucket}/{path}"
        encoded = quote(path, safe="/")
        url = f"{self.base_url}/storage/v1/object/{encoded}"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def delete_pdf(self, storage_path: str) -> None:
        if not storage_path:
            return
        path = storage_path.lstrip("/")
        if not path.startswith(f"{self.bucket}/"):
            path = f"{self.bucket}/{path}"
        encoded = quote(path, safe="/")
        url = f"{self.base_url}/storage/v1/object/{encoded}"
        resp = requests.delete(url, headers=self._headers(), timeout=self.timeout)
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()


# ---------------------------------------------------------------------------
# Singleton factory (cache-keyed by env vars for clean test isolation)
# ---------------------------------------------------------------------------

_storage_lock = threading.Lock()
_storage_instance: Optional[LocalStorageBackend | PostgresStorageBackend] = None
_storage_cache_key: Optional[tuple[str, str, str]] = None


def get_storage() -> LocalStorageBackend | PostgresStorageBackend:
    """Return the active storage backend (Postgres if ``DATABASE_URL`` is set, else local)."""
    global _storage_instance, _storage_cache_key
    dsn = os.environ.get("DATABASE_URL", "").strip()
    live_env = os.environ.get("LIVE_INTERACTIONS_PATH", "").strip()
    drift_env = os.environ.get("DRIFT_REPORTS_DIR", "").strip()
    key = (dsn, live_env, drift_env)

    if _storage_instance is not None and _storage_cache_key == key:
        return _storage_instance

    with _storage_lock:
        if _storage_instance is not None and _storage_cache_key == key:
            return _storage_instance
        if dsn:
            _storage_instance = PostgresStorageBackend(dsn=dsn, auto_init_schema=True)
        else:
            _storage_instance = LocalStorageBackend()
        _storage_cache_key = key
        return _storage_instance


def reset_storage() -> None:
    """Clear the cached storage backend singleton (used by tests)."""
    global _storage_instance, _storage_cache_key
    with _storage_lock:
        _storage_instance = None
        _storage_cache_key = None
