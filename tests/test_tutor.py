"""Offline unit tests for the Phase 3 digital literacy tutor (``app/tutor.py``).

No network access or real ``GOOGLE_API_KEY`` is required:
  * Gemini is replaced by an in-process fake LLM (``tutor._get_llm`` is patched).
  * Embeddings use LlamaIndex ``MockEmbedding`` (no ONNX model download).
  * ChromaDB persists to a per-test ``tmp_path`` directory.
"""

from __future__ import annotations

from io import BytesIO

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.vector_stores.chroma")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole  # noqa: E402

import app.tutor as tutor  # noqa: E402


class FakeLLM:
    """Records prompts and returns a deterministic Socratic reply."""

    def __init__(self, reply: str = "What clues in the sender address look unusual? [1]"):
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.reply))


class ExplodingLLM:
    def chat(self, messages, **kwargs):
        raise RuntimeError("upstream 503")


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def tutor_client(tmp_path, monkeypatch, fake_llm):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("TUTOR_COLLECTION", "test_digital_literacy")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    monkeypatch.setattr(tutor, "_get_embed_model", lambda: MockEmbedding(embed_dim=384))
    monkeypatch.setattr(tutor, "_get_llm", lambda api_key: fake_llm)
    tutor.reset_store()

    app = FastAPI()
    app.include_router(tutor.router)
    with TestClient(app) as client:
        yield client
    tutor.reset_store()


STRUGGLING_CONTEXT = {
    "skill_name": "phishing_cyber_hygiene",
    "attempt_count": 5,
    "hint_count": 3,
    "bottom_hint": 1,
    "needs_intervention": True,
}


def test_ask_returns_grounded_socratic_answer(tutor_client, fake_llm):
    resp = tutor_client.post(
        "/tutor/ask",
        json={
            "question": "How do I spot a phishing email?",
            "student_context": STRUGGLING_CONTEXT,
            "top_k": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == fake_llm.reply
    assert body["model_version"].startswith(f"{tutor.TUTOR_VERSION}/")
    assert len(body["sources"]) == 2
    assert all(s["source"].startswith("builtin://digital_literacy/") for s in body["sources"])

    system_msg, user_msg = fake_llm.calls[0]
    assert system_msg.role == MessageRole.SYSTEM
    assert "NEVER give away the direct answer" in system_msg.content
    assert "COURSE CONTEXT" in user_msg.content
    assert "STRUGGLING" in user_msg.content  # intervention-aware scaffolding
    assert "skill_name: phishing_cyber_hygiene" in user_msg.content


def test_auto_seeds_empty_collection(tutor_client):
    store = tutor._get_store()
    assert store.collection.count() > 0
    first = store.collection.count()
    tutor.reset_store()
    # Re-opening a populated collection must not duplicate the seed curriculum.
    assert tutor._get_store().collection.count() == first


def test_missing_api_key_is_hard_500(tutor_client, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resp = tutor_client.post("/tutor/ask", json={"question": "What is MFA?"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "GOOGLE_API_KEY environment variable is not configured"


def test_gemini_failure_is_hard_500(tutor_client, monkeypatch):
    monkeypatch.setattr(tutor, "_get_llm", lambda api_key: ExplodingLLM())
    resp = tutor_client.post("/tutor/ask", json={"question": "What is MFA?"})
    assert resp.status_code == 500
    assert "Gemini generation failed" in resp.json()["detail"]


def test_empty_gemini_answer_is_hard_500(tutor_client, monkeypatch):
    monkeypatch.setattr(tutor, "_get_llm", lambda api_key: FakeLLM(reply="   "))
    resp = tutor_client.post("/tutor/ask", json={"question": "What is MFA?"})
    assert resp.status_code == 500


def test_vector_store_failure_is_hard_500(tutor_client, monkeypatch):
    def broken_store():
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(tutor, "_get_store", broken_store)
    resp = tutor_client.post("/tutor/ask", json={"question": "What is MFA?"})
    assert resp.status_code == 500
    assert "Vector store retrieval failed" in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 11}, {}],
)
def test_ask_validation(tutor_client, payload):
    assert tutor_client.post("/tutor/ask", json=payload).status_code == 422


def test_index_directory_is_idempotent(tutor_client, tmp_path, monkeypatch):
    docs = tmp_path / "course_docs"
    (docs / "nested").mkdir(parents=True)
    (docs / "passwords.md").write_text("# Passphrases\nUse long unique passphrases and MFA.")
    (docs / "nested" / "search.txt").write_text("Use quotation marks for exact phrase search.")
    (docs / "ignored.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(tutor, "REPO_ROOT", tmp_path)

    first = tutor_client.post("/tutor/index", json={"directory": str(docs)})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["documents_indexed"] == 2
    assert sorted(body["files"]) == ["course_docs/nested/search.txt", "course_docs/passwords.md"]

    second = tutor_client.post("/tutor/index", json={"directory": str(docs)}).json()
    assert second["collection_size"] == body["collection_size"]


def test_index_rejects_paths_outside_project(tutor_client, tmp_path, monkeypatch):
    monkeypatch.setattr(tutor, "REPO_ROOT", tmp_path / "project")
    resp = tutor_client.post("/tutor/index", json={"directory": "/etc"})
    assert resp.status_code == 400


def test_index_missing_directory_404(tutor_client, tmp_path, monkeypatch):
    monkeypatch.setattr(tutor, "REPO_ROOT", tmp_path)
    resp = tutor_client.post("/tutor/index", json={"directory": str(tmp_path / "nope")})
    assert resp.status_code == 404


def test_cli_index(tutor_client, tmp_path, monkeypatch, capsys):
    docs = tmp_path / "cli_docs"
    docs.mkdir()
    (docs / "privacy.md").write_text("# Privacy\nReview app permissions regularly.")
    monkeypatch.setattr(tutor, "REPO_ROOT", tmp_path)
    assert tutor.main(["--index", str(docs)]) == 0
    assert "Indexed 1 document(s)" in capsys.readouterr().out


def test_instructional_plan_adapts_to_telemetry():
    assert "STRUGGLING" in tutor._instructional_plan(STRUGGLING_CONTEXT)
    progressing = tutor._instructional_plan({"attempt_count": 1, "hint_count": 0})
    assert "progressing" in progressing
    assert "No telemetry" in tutor._instructional_plan({})


def course_pdf(*pages, password=None):
    """Generate a real PDF, including blank pages, without a fixture binary."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=400, height=400)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 20 350 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def upload(client, data, filename="lecture.pdf"):
    return client.post("/tutor/documents", params={"filename": filename}, content=data,
                       headers={"content-type": "application/pdf"})


def test_pdf_library_to_grounded_answer(tutor_client, fake_llm):
    data = course_pdf("", "Chlorophyll absorbs light during photosynthesis.")
    first = upload(tutor_client, data)
    assert first.status_code == 200, first.text
    book = first.json()
    assert (book["pages"], book["pages_indexed"], book["chunks"]) == (2, 1, 1)
    assert upload(tutor_client, data, "renamed.pdf").json() == book
    other = upload(tutor_client, course_pdf("A separate geometry lecture."))
    assert other.json()["id"] != book["id"]  # same filename, different content
    tutor._client.cache_clear()
    assert len(tutor_client.get("/tutor/documents").json()) == 2
    res = tutor_client.post("/tutor/ask", json={"document_id": book["id"],
        "question": "What absorbs light?", "history": [{"question": "What is a pigment?",
                                                        "answer": "Think about leaf colour."}]})
    assert res.status_code == 200, res.text
    src, = res.json()["sources"]
    assert (src["page"], src["source"]) == (2, "lecture.pdf")
    assert "Chlorophyll" in src["text"]
    prompt = fake_llm.calls[-1][-1].content
    assert "Chlorophyll" in prompt and "page 2" in prompt and "geometry" not in prompt
    assert fake_llm.calls[-1][1].content == "What is a pigment?"
    assert tutor_client.post("/tutor/ask", json={"question": "Test", "document_id": "a"*64}).status_code == 404


def test_upload_rejects_unusable_files(tutor_client, monkeypatch):
    assert upload(tutor_client, b"hello").status_code == 415
    assert upload(tutor_client, b"%PDF-unreadable").status_code == 422
    assert upload(tutor_client, course_pdf("")).status_code == 422
    assert upload(tutor_client, course_pdf("Text", password="test")).status_code == 422
    monkeypatch.setattr(tutor, "MAX_BYTES", 20)
    assert upload(tutor_client, b"%PDF-" + b"x"*20).status_code == 413
    assert tutor_client.get("/tutor/documents").json() == []


def test_upload_retry_after_embedding_failure(tutor_client, monkeypatch):
    with monkeypatch.context() as patch:
        def failed_embedding():
            raise RuntimeError("Embedding weights unavailable")
        patch.setattr(tutor, "_get_embed_model", failed_embedding)
        data = course_pdf("Course notes about acceleration.")
        assert upload(tutor_client, data).status_code == 503
    assert tutor_client.get("/tutor/documents").json() == []
    assert upload(tutor_client, data).status_code == 200


def test_tutor_ui_assets(tutor_client):
    assert "Study room" in tutor_client.get("/tutor").text
    assert tutor_client.get("/tutor/assets/study.css").status_code == 200
    assert tutor_client.get("/tutor/assets/study.js").status_code == 200
