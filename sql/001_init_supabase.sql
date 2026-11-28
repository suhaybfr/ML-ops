-- =============================================================================
--  Milestone 1: Cloud Persistence & Online Feature Store (Supabase Postgres)
-- =============================================================================
--  Idempotent schema for:
--    1. students            — Online Feature Store (expanding user_prev_acc & skill opportunity)
--    2. assessment_items    — Graded Digital Literacy items (Milestone 2 ready)
--    3. interaction_events  — Live 11-feature inference & outcome telemetry
--    4. drift_snapshots     — Covariate & concept drift history
--    5. retrain_runs        — Drift-gated retraining & promotion history
--    6. rag_documents       — Content-addressed course PDFs & Supabase Storage paths
--    7. rag_chunks          — pgvector(384) chunks for ONNX all-MiniLM-L6-v2 RAG
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS students (
    student_id          TEXT PRIMARY KEY,
    display_name        TEXT,
    total_attempts      INTEGER NOT NULL DEFAULT 0,
    correct_attempts    INTEGER NOT NULL DEFAULT 0,
    user_prev_acc       DOUBLE PRECISION NOT NULL DEFAULT 0.70,
    skill_opportunities JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assessment_items (
    item_id            TEXT PRIMARY KEY,
    skill_name         TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    options            JSONB NOT NULL DEFAULT '[]'::jsonb,
    correct_answer     TEXT NOT NULL,
    hints              JSONB NOT NULL DEFAULT '[]'::jsonb,
    bottom_hint_text   TEXT,
    problem_difficulty DOUBLE PRECISION NOT NULL DEFAULT 0.70,
    skill_difficulty   DOUBLE PRECISION NOT NULL DEFAULT 0.69,
    is_skill_unmapped  INTEGER NOT NULL DEFAULT 0,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interaction_events (
    id                 BIGSERIAL PRIMARY KEY,
    timestamp          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id         TEXT,
    student_id         TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    item_id            TEXT REFERENCES assessment_items(item_id) ON DELETE SET NULL,
    source             TEXT NOT NULL DEFAULT 'api_predict',
    skill_name         TEXT NOT NULL DEFAULT 'Digital Literacy',
    opportunity        INTEGER NOT NULL DEFAULT 1,
    log_opportunity    DOUBLE PRECISION NOT NULL,
    attempt_count      DOUBLE PRECISION NOT NULL,
    hint_count         DOUBLE PRECISION NOT NULL,
    bottom_hint        DOUBLE PRECISION NOT NULL,
    hint_ratio         DOUBLE PRECISION NOT NULL,
    user_prev_acc      DOUBLE PRECISION NOT NULL,
    problem_difficulty DOUBLE PRECISION NOT NULL,
    skill_difficulty   DOUBLE PRECISION NOT NULL,
    is_skill_unmapped  DOUBLE PRECISION NOT NULL,
    log_response_time  DOUBLE PRECISION NOT NULL,
    is_rapid_guess     DOUBLE PRECISION NOT NULL,
    prediction         INTEGER NOT NULL,
    probability        DOUBLE PRECISION NOT NULL,
    needs_intervention INTEGER NOT NULL,
    model_version      TEXT NOT NULL DEFAULT 'unknown',
    correct            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_interaction_events_session_id
    ON interaction_events (session_id);
CREATE INDEX IF NOT EXISTS idx_interaction_events_student_id
    ON interaction_events (student_id);
CREATE INDEX IF NOT EXISTS idx_interaction_events_timestamp
    ON interaction_events (timestamp DESC);

CREATE TABLE IF NOT EXISTS drift_snapshots (
    id                     BIGSERIAL PRIMARY KEY,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tag                    TEXT,
    engine                 TEXT,
    drift_share            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    dataset_drifted        BOOLEAN NOT NULL DEFAULT FALSE,
    drifted_features_count INTEGER NOT NULL DEFAULT 0,
    model_drift_detected   BOOLEAN NOT NULL DEFAULT FALSE,
    payload                JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_drift_snapshots_created_at
    ON drift_snapshots (created_at DESC);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                          BIGSERIAL PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status                      TEXT NOT NULL,
    retrained                   BOOLEAN NOT NULL DEFAULT FALSE,
    promoted                    BOOLEAN NOT NULL DEFAULT FALSE,
    retrain_runs_total          INTEGER NOT NULL DEFAULT 0,
    retrain_promotions_total    INTEGER NOT NULL DEFAULT 0,
    retrain_gate_failures_total INTEGER NOT NULL DEFAULT 0,
    watermark_rows              INTEGER NOT NULL DEFAULT 0,
    candidate_f1                DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    active_model_version        TEXT,
    active_artifact_path        TEXT,
    payload                     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrain_runs_created_at
    ON retrain_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS rag_documents (
    document_id   TEXT PRIMARY KEY,
    collection    TEXT NOT NULL,
    filename      TEXT NOT NULL,
    pages         INTEGER NOT NULL,
    pages_indexed INTEGER NOT NULL,
    chunks        INTEGER NOT NULL,
    ready         BOOLEAN NOT NULL DEFAULT FALSE,
    storage_path  TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_collection
    ON rag_documents (collection);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id    TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source      TEXT NOT NULL,
    page        INTEGER,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_collection
    ON rag_chunks (collection);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
    ON rag_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_cosine
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
