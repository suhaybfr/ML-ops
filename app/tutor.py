"""Phase 3 conversational tutor for the AI-Based Digital Literacy Assistant.

When the Phase 2 champion Random Forest (``app/main.py``) flags a student with
``needs_intervention == true``, this router serves Socratic, retrieval-grounded
instructional guidance on digital literacy skills (source credibility, digital
privacy, phishing / cyber hygiene, search strategies, software navigation).

Architecture::

    POST /tutor/ask {question, student_context, top_k}
         │
         ▼
    [LlamaIndex retriever] ── ChromaVectorStore ── chromadb.PersistentClient("./chroma_db")
         │   (embeddings: Chroma-native ONNX all-MiniLM-L6-v2, CPU-only, no PyTorch)
         ▼
    Socratic system prompt + retrieved course context + student telemetry profile
         │
         ▼
    [Gemini via llama_index.llms.google_genai] ──► {answer, sources, model_version}

Failure policy: there are NO canned / mock fallbacks. A missing ``GOOGLE_API_KEY``,
an unreachable ChromaDB, or any Gemini error surfaces as HTTP 500.

Mounted by ``app/main.py`` under ``/tutor`` when ``ENABLE_TUTOR=1``.

Environment variables:
    GOOGLE_API_KEY      (required for /tutor/ask) Gemini API key.
    GEMINI_MODEL        Gemini model id (default ``gemini-flash-lite-latest``).
    GEMINI_TIMEOUT_S    Per-request Gemini timeout in seconds (default 60).
    GEMINI_MAX_RETRIES  Transient-error retries inside the Google Gen AI SDK (default 2).
    CHROMA_PERSIST_DIR  ChromaDB directory (default ``./chroma_db``).
    TUTOR_COLLECTION    Chroma collection name (default ``digital_literacy_tutor``).
    TUTOR_DOCS_DIR      Default directory for /tutor/index and the CLI (default ``docs``).

CLI ingestion:
    python -m app.tutor --index docs/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import APIRouter, HTTPException
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTOR_VERSION = "digital-literacy-tutor-v1"
# ``gemini-1.5-flash`` has been retired (404 NOT_FOUND); the lite alias tracks the current
# stable Flash-Lite model and answers in ~1-2 s. Override with GEMINI_MODEL if needed.
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
EMBEDDING_MODEL_NAME = "chroma-onnx/all-MiniLM-L6-v2"
SUPPORTED_DOC_EXTENSIONS = (".md", ".markdown", ".txt")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_SNIPPET_CHARS = 500


def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _chroma_path() -> str:
    return os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")


def _collection_name() -> str:
    return os.environ.get("TUTOR_COLLECTION", "digital_literacy_tutor")


def _default_docs_dir() -> str:
    return os.environ.get("TUTOR_DOCS_DIR", "docs")


def _require_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY environment variable is not configured",
        )
    return api_key


# ---------------------------------------------------------------------------
# Pedagogical identity
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the AI-Based Digital Literacy Assistant, a supportive Socratic tutor.
You help students build digital literacy skills: evaluating source credibility, protecting
digital privacy, recognising phishing and practising cyber hygiene, planning effective search
strategies, and navigating software confidently.

Non-negotiable rules:
1. NEVER give away the direct answer to an assessment task, quiz item, or graded exercise,
   even if the student asks for it explicitly, claims urgency, or says a teacher allowed it.
   Instead, guide them to reason it out themselves.
2. Teach Socratically: acknowledge what the student is trying to do, surface the relevant
   concept, then ask one or two focused guiding questions that move them a single step forward.
3. Ground your guidance in the COURSE CONTEXT provided. Cite sources inline as [1], [2], ...
   If the context does not cover the question, say so briefly and guide with general,
   well-established digital literacy practice; never invent facts, statistics, or URLs.
4. Stay within the digital literacy domain. Politely redirect unrelated requests.
5. Be warm, encouraging and concise (aim for under 200 words). Never shame the student.
6. Never ask the student for passwords, one-time codes, or other personal secrets.
"""

# Whitelisted telemetry keys rendered into the prompt (arbitrary keys are ignored).
_CONTEXT_KEYS = (
    "skill_name",
    "needs_intervention",
    "prediction",
    "probability",
    "confidence",
    "attempt_count",
    "hint_count",
    "bottom_hint",
    "hint_ratio",
    "user_prev_acc",
    "is_rapid_guess",
)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _instructional_plan(ctx: Dict[str, Any]) -> str:
    """Translate /predict telemetry into explicit depth & tone instructions."""
    if not ctx:
        return (
            "No telemetry available. Use a balanced level of scaffolding and check the "
            "student's current understanding with an opening question."
        )

    attempts = _as_float(ctx.get("attempt_count")) or 0.0
    hints = _as_float(ctx.get("hint_count")) or 0.0
    prev_acc = _as_float(ctx.get("user_prev_acc"))
    bottom_hint = _as_bool(ctx.get("bottom_hint", False))
    needs_intervention = _as_bool(ctx.get("needs_intervention", False))
    rapid_guess = _as_bool(ctx.get("is_rapid_guess", False))

    guidance: List[str] = []
    if needs_intervention or bottom_hint or attempts >= 3:
        guidance.append(
            "The student is STRUGGLING and has been flagged for intervention. Use heavy "
            "scaffolding: plain language, break the concept into small steps, give one "
            "concrete worked example on a DIFFERENT scenario than their task, and end with "
            "a single, easy guiding question."
        )
    else:
        guidance.append(
            "The student is progressing. Use light scaffolding: prompt them to explain their "
            "reasoning and extend it with a slightly more challenging guiding question."
        )
    if bottom_hint or hints >= 2:
        guidance.append(
            "They have relied heavily on hints (possibly revealing the final hint). Do NOT "
            "restate hints or answers; rebuild the underlying concept instead."
        )
    if rapid_guess:
        guidance.append(
            "They may be guessing quickly. Gently encourage slowing down and using a "
            "short checklist before answering."
        )
    if prev_acc is not None and prev_acc < 0.4:
        guidance.append(
            "Their historical accuracy is low; be especially encouraging and build confidence."
        )
    return " ".join(guidance)


def _render_student_profile(ctx: Dict[str, Any]) -> str:
    lines = [f"- {key}: {ctx[key]}" for key in _CONTEXT_KEYS if key in ctx]
    return "\n".join(lines) if lines else "- (none provided)"


# ---------------------------------------------------------------------------
# Built-in seed curriculum (used when the Chroma collection is empty)
# ---------------------------------------------------------------------------

SEED_MODULES: Dict[str, str] = {
    "source_credibility": """# Module 1: Evaluating Source Credibility

Credible information comes from sources that are accountable, transparent and evidence-based.

## The SIFT method
- **Stop:** pause before sharing or believing a claim; notice your emotional reaction.
- **Investigate the source:** who published it, what is their expertise, and what might
  motivate them? Search the author or organisation name in a new tab (lateral reading).
- **Find better coverage:** look for the same claim reported by several independent,
  reputable outlets or primary sources.
- **Trace claims to the original context:** follow quotes, images and statistics back to where
  they first appeared and check whether they were cropped, edited or taken out of context.

## The CRAAP checklist
- **Currency:** when was it published or updated? Is that recent enough for the topic?
- **Relevance:** does it actually address your question and audience?
- **Authority:** are the author's credentials and the publisher identifiable and qualified?
- **Accuracy:** is the evidence cited, verifiable and free of obvious errors?
- **Purpose:** is the aim to inform, persuade, sell or entertain? Is bias disclosed?

## Warning signs
Sensational headlines, no named author, missing citations, look-alike domains that imitate
trusted outlets, and content designed mainly to provoke outrage. Use reverse image search to
check whether photos are old or from a different event.
""",
    "digital_privacy": """# Module 2: Protecting Your Digital Privacy

Your digital footprint is the trail of data you leave online, both actively (posts, profiles)
and passively (location, cookies, app tracking).

## Core habits
- Review **privacy settings** on social media and limit who can see posts, tags and contacts.
- Grant apps only the **permissions** they truly need (location, microphone, contacts).
- Think before sharing personal details such as full birth date, home address, school
  schedule or travel plans, which can be combined to identify or impersonate you.
- Understand that **cookies and trackers** follow browsing; use private browsing, clear cookies
  and consider tracker-blocking settings.
- Read the key parts of a **privacy policy**: what data is collected, who it is shared with,
  how long it is kept and how to delete it.

## Account protection
- Use a long, unique **passphrase** for each account, ideally stored in a password manager.
- Turn on **multi-factor authentication (MFA)** wherever available.
- Be careful on **public Wi-Fi**; avoid logging into sensitive accounts or use a trusted VPN.
- Periodically search your own name to see what others can find about you.
""",
    "phishing_cyber_hygiene": """# Module 3: Phishing Awareness and Cyber Hygiene

Phishing is a social-engineering attack that tricks people into revealing credentials,
sending money or installing malware via email, text (smishing), calls (vishing) or messages.

## Red flags
- **Urgency or fear:** "Your account will be closed in 24 hours!"
- **Mismatched sender:** display name looks official but the email address or domain is
  slightly wrong (e.g. extra letters, swapped characters, unusual top-level domain).
- **Suspicious links:** hover over (or long-press) a link to preview the real destination
  before clicking; check that the domain matches the organisation.
- **Unexpected attachments** or requests to enable macros.
- **Requests for secrets:** legitimate organisations do not ask for passwords or one-time codes.
- Generic greetings, unusual tone and offers that are too good to be true.

## What to do
Do not click or reply. Verify by contacting the organisation through a channel you find
yourself (official website or app). Report the message (e.g. "Report phishing") and delete it.
If you clicked, change your password immediately, enable MFA and tell your IT contact.

## Everyday cyber hygiene
Keep operating systems, browsers and apps updated; install software only from official
sources; use antivirus protection; back up important files; lock your devices.
""",
    "search_strategies": """# Module 4: Effective Search Strategies

Good searching starts with a clear question and the right keywords, not a full sentence.

## Building a query
- Identify the **key concepts** in your question and brainstorm synonyms.
- Use **quotation marks** for exact phrases: "digital footprint".
- Use **Boolean logic**: AND narrows, OR broadens with synonyms, NOT / minus (-) excludes terms.
- Use **operators**: `site:` to search within a domain (site:.gov), `filetype:` for formats
  (filetype:pdf), and date filters for recent results.

## Choosing where to search
General web search engines are broad; library databases and scholarly search tools offer
peer-reviewed sources; specialised sites (government statistics, encyclopedias) offer
authoritative overviews.

## Evaluating results
Notice which results are **ads or sponsored**. Look beyond the first few results, skim
snippets and URLs, and open several sources to compare. Refine your query iteratively:
too many results means narrow it, too few means broaden it with synonyms.
""",
    "software_navigation": """# Module 5: Navigating Software Confidently

Most applications share common interface patterns, so skills transfer between tools.

## Common patterns
- **Menus and ribbons** group related commands (File, Edit, View, Insert, Help).
- **Toolbars and icons:** hover over an icon to see its tooltip and name.
- **Right-click context menus** show actions for the selected item.
- **Settings / Preferences** usually live under a gear icon or the File / app menu.
- **Search boxes** inside apps can locate commands when you do not know where they are.

## Productive habits
- Learn universal **keyboard shortcuts**: copy, paste, undo, redo, save, find.
- Save work often and understand where files are saved: local folders vs. cloud storage.
- Use a clear **file and folder naming** scheme and version names.
- When stuck, use built-in **Help**, official documentation or tutorials, and try one change
  at a time so you can undo it.
- Understand **sharing permissions** (view, comment, edit) before sharing cloud documents.
""",
}


# ---------------------------------------------------------------------------
# Embeddings: Chroma-native ONNX all-MiniLM-L6-v2 adapted to LlamaIndex
# ---------------------------------------------------------------------------


class ChromaOnnxEmbedding(BaseEmbedding):
    """LlamaIndex adapter for ChromaDB's bundled ONNX ``all-MiniLM-L6-v2`` (CPU, no torch)."""

    _ef: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", EMBEDDING_MODEL_NAME)
        super().__init__(**kwargs)
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vector] for vector in self._ef(list(texts))]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


def _get_embed_model() -> BaseEmbedding:
    return ChromaOnnxEmbedding()


def _get_llm(api_key: str) -> Any:
    """Build the Gemini LLM client via the Google Gen AI SDK (patched in offline unit tests)."""
    from google.genai import types
    from llama_index.llms.google_genai import GoogleGenAI

    timeout_ms = int(float(os.environ.get("GEMINI_TIMEOUT_S", "60")) * 1000)
    return GoogleGenAI(
        model=_gemini_model().removeprefix("models/"),
        api_key=api_key,
        temperature=0.3,
        max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "2")),
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


# ---------------------------------------------------------------------------
# Vector store lifecycle (lazy, thread-safe, auto-seeded)
# ---------------------------------------------------------------------------


class _TutorStore:
    def __init__(self, collection: Any, vector_store: ChromaVectorStore, index: VectorStoreIndex):
        self.collection = collection
        self.vector_store = vector_store
        self.index = index


_store: Optional[_TutorStore] = None
_store_lock = threading.Lock()


def reset_store() -> None:
    """Drop the cached Chroma/LlamaIndex handles (used by tests and re-configuration)."""
    global _store
    with _store_lock:
        _store = None


def _node_id(ref_doc_id: str, position: int) -> str:
    return hashlib.sha1(f"{ref_doc_id}::{position}".encode("utf-8")).hexdigest()


def _ingest(store: _TutorStore, documents: List[Document]) -> int:
    """Idempotently (re-)index documents: old chunks of the same doc id are replaced."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    positions: Dict[str, int] = {}
    for node in nodes:
        ref = node.ref_doc_id or node.node_id
        positions[ref] = positions.get(ref, -1) + 1
        node.id_ = _node_id(ref, positions[ref])
    for doc in documents:
        store.vector_store.delete(ref_doc_id=doc.doc_id)
    if nodes:
        store.index.insert_nodes(nodes)
    return len(nodes)


def _seed_documents() -> List[Document]:
    return [
        Document(
            text=text,
            id_=f"seed::{slug}",
            metadata={"source": f"builtin://digital_literacy/{slug}", "topic": slug},
        )
        for slug, text in SEED_MODULES.items()
    ]


def _get_store() -> _TutorStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        client = chromadb.PersistentClient(
            path=_chroma_path(),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=_collection_name(), configuration={"hnsw": {"space": "cosine"}}
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=_get_embed_model()
        )
        store = _TutorStore(collection, vector_store, index)
        if collection.count() == 0:
            _ingest(store, _seed_documents())
        _store = store
        return _store


def _load_directory(directory: str) -> tuple[Path, List[Document]]:
    """Read markdown/text files under ``directory`` (restricted to the project root)."""
    root = REPO_ROOT.resolve()
    target = Path(directory)
    target = (target if target.is_absolute() else Path.cwd() / target).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(
            status_code=400, detail=f"Directory must be inside the project root ({root})."
        )
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    documents: List[Document] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        rel = path.relative_to(root).as_posix()
        documents.append(
            Document(text=text, id_=f"file::{rel}", metadata={"source": rel, "topic": path.stem})
        )
    return target, documents


def index_directory(directory: str) -> Dict[str, Any]:
    target, documents = _load_directory(directory)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail=f"No {', '.join(SUPPORTED_DOC_EXTENSIONS)} files found in {directory}",
        )
    store = _get_store()
    chunks = _ingest(store, documents)
    return {
        "directory": str(target.relative_to(REPO_ROOT.resolve()) or "."),
        "documents_indexed": len(documents),
        "chunks_indexed": chunks,
        "collection": _collection_name(),
        "collection_size": store.collection.count(),
        "files": [doc.metadata["source"] for doc in documents],
    }


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TutorAskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4000,
        description="The student's question or the topic they need help with.",
    )
    student_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional telemetry from /predict (e.g. attempt_count, hint_count, bottom_hint, "
            "needs_intervention, skill_name) used to adapt instructional depth and tone."
        ),
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "How can I tell if this email from my 'bank' is a phishing attempt?",
                "student_context": {
                    "skill_name": "phishing_cyber_hygiene",
                    "attempt_count": 5,
                    "hint_count": 3,
                    "bottom_hint": 1,
                    "needs_intervention": True,
                },
                "top_k": 3,
            }
        }
    )


class SourceSnippet(BaseModel):
    source: str
    text: str
    score: Optional[float] = None


class TutorAskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    model_version: str


class TutorIndexRequest(BaseModel):
    directory: str = Field(
        default_factory=_default_docs_dir,
        description="Directory (inside the project) containing .md / .txt course documents.",
    )


class TutorIndexResponse(BaseModel):
    directory: str
    documents_indexed: int
    chunks_indexed: int
    collection: str
    collection_size: int
    files: List[str]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/tutor", tags=["Tutor (Digital Literacy RAG)"])


def _build_messages(body: TutorAskRequest, snippets: List[SourceSnippet]) -> List[ChatMessage]:
    ctx = body.student_context or {}
    if snippets:
        context_block = "\n\n".join(
            f"[{i}] (source: {s.source})\n{s.text}" for i, s in enumerate(snippets, start=1)
        )
    else:
        context_block = "(no relevant course material was retrieved)"
    user_prompt = (
        f"COURSE CONTEXT:\n{context_block}\n\n"
        f"STUDENT PROFILE (from the performance model):\n{_render_student_profile(ctx)}\n\n"
        f"INSTRUCTIONAL PLAN:\n{_instructional_plan(ctx)}\n\n"
        f"STUDENT QUESTION:\n{body.question.strip()}"
    )
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]


@router.post("/ask", response_model=TutorAskResponse, summary="Ask the digital literacy tutor")
def ask_tutor(body: TutorAskRequest) -> TutorAskResponse:
    """Retrieve grounding chunks from ChromaDB and generate Socratic guidance with Gemini."""
    api_key = _require_api_key()

    try:
        store = _get_store()
        retrieved = store.index.as_retriever(similarity_top_k=body.top_k).retrieve(body.question)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Vector store retrieval failed: {type(exc).__name__}: {exc}"
        ) from exc

    snippets = [
        SourceSnippet(
            source=str(item.node.metadata.get("source", "unknown")),
            text=item.node.get_content()[:MAX_SNIPPET_CHARS],
            score=None if item.score is None else round(float(item.score), 4),
        )
        for item in retrieved
    ]

    try:
        response = _get_llm(api_key).chat(_build_messages(body, snippets))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Gemini generation failed: {type(exc).__name__}: {exc}"
        ) from exc

    answer = (response.message.content or "").strip()
    if not answer:
        raise HTTPException(status_code=500, detail="Gemini returned an empty response.")

    return TutorAskResponse(
        answer=answer,
        sources=snippets,
        model_version=f"{TUTOR_VERSION}/{_gemini_model()}",
    )


@router.post("/index", response_model=TutorIndexResponse, summary="Ingest course documents")
def index_documents(body: Optional[TutorIndexRequest] = None) -> TutorIndexResponse:
    """Ingest .md / .txt files from a directory (default ``docs/``) into ChromaDB."""
    body = body or TutorIndexRequest()
    try:
        return TutorIndexResponse(**index_directory(body.directory))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Document indexing failed: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CLI: python -m app.tutor --index docs/
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest digital literacy course documents into the tutor's ChromaDB store."
    )
    parser.add_argument(
        "--index", metavar="DOCS_DIR", nargs="?", const=_default_docs_dir(),
        help="Directory of .md/.txt documents to index (default: docs/)",
    )
    args = parser.parse_args(argv)
    if args.index is None:
        parser.print_help()
        return 1
    try:
        result = index_directory(args.index)
    except HTTPException as exc:
        parser.error(str(exc.detail))
    print(
        f"[tutor] Indexed {result['documents_indexed']} document(s) / "
        f"{result['chunks_indexed']} chunk(s) into '{result['collection']}' "
        f"(collection size: {result['collection_size']})"
    )
    for name in result["files"]:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
