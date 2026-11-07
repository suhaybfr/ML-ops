"""
LLM-powered Conversational Tutor — FastAPI endpoints.

This module adds a /tutor endpoint family to the MLOps project, providing:
  - RAG-based Q&A over course documents (LlamaIndex + Qdrant vector store)
  - Explanation generation for model predictions
  - Practice question generation for a given skill/topic

Architecture:
    Student question
         │
         ▼
    [Qdrant vector store] ←─ course documents / skill descriptions
         │  (similarity search)
         ▼
    [LlamaIndex query engine]
         │  (retrieved context)
         ▼
    [Gemini API]  ←─ system prompt + context + question
         │
         ▼
    Tutor response

Setup:
    1. Set environment variables:
           GOOGLE_API_KEY=<your Gemini API key>
           QDRANT_HOST=localhost   (or qdrant container name)
           QDRANT_PORT=6333
    2. Index your course documents once:
           python app/tutor.py --index docs/
    3. Start the main FastAPI app — tutor endpoints are automatically included.

Dependencies (add to requirements.txt):
    google-generativeai
    llama-index
    llama-index-vector-stores-qdrant
    llama-index-llms-gemini
    llama-index-embeddings-gemini
    qdrant-client
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Optional imports (gracefully degrade if not installed) ──────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Settings,
        StorageContext,
    )
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False

# ── Configuration ────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "tutor_docs"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

SYSTEM_PROMPT = """You are an expert educational AI tutor helping students understand mathematics
and science concepts. You have access to course materials and can:
- Answer questions about specific topics clearly and concisely
- Explain why a student might have struggled (given their performance data)
- Generate practice questions tailored to the student's skill level
- Provide step-by-step explanations

Always be encouraging, clear, and pedagogically sound. If you don't know something,
say so rather than guessing."""

# ── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tutor", tags=["Tutor (LLM/RAG)"])

# ── Module-level singletons ──────────────────────────────────────────────────

_gemini_client: Optional[object] = None
_query_engine: Optional[object] = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not _GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai package not installed. Run: pip install google-generativeai")
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=GOOGLE_API_KEY)
        _gemini_client = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
    return _gemini_client


def _get_query_engine(docs_dir: str = "docs"):
    """Build or return cached LlamaIndex query engine backed by Qdrant."""
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    if not _LLAMA_AVAILABLE:
        raise RuntimeError("llama-index not installed. Run: pip install llama-index llama-index-vector-stores-qdrant llama-index-llms-gemini llama-index-embeddings-gemini")
    if not _QDRANT_AVAILABLE:
        raise RuntimeError("qdrant-client not installed. Run: pip install qdrant-client")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    from llama_index.llms.gemini import Gemini
    from llama_index.embeddings.gemini import GeminiEmbedding

    Settings.llm = Gemini(model_name=f"models/{GEMINI_MODEL}", api_key=GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=GOOGLE_API_KEY
    )

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index documents if directory exists
    if os.path.isdir(docs_dir):
        documents = SimpleDirectoryReader(docs_dir).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    _query_engine = index.as_query_engine(similarity_top_k=3)
    return _query_engine


# ── Pydantic schemas ─────────────────────────────────────────────────────────


class TutorQuestion(BaseModel):
    question: str = Field(..., description="The student's question or topic to explain.")
    skill_id: Optional[int] = Field(None, description="Optional skill ID for context.")
    use_rag: bool = Field(
        True, description="If True, retrieve context from the vector store before answering."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "Why do I keep getting hint penalties for skill 7?",
            "skill_id": 7,
            "use_rag": True,
        }
    }}


class PracticeRequest(BaseModel):
    topic: str = Field(..., description="Topic or skill name to generate questions for.")
    difficulty: str = Field("medium", description="easy | medium | hard")
    num_questions: int = Field(3, ge=1, le=10, description="Number of questions to generate.")


class ExplainPredictionRequest(BaseModel):
    skill_id: int
    attempt_count: int
    ms_first_response: float
    hint_count: int
    overlap_time: float
    prediction: int = Field(..., description="0=incorrect, 1=correct")
    confidence: float


class TutorResponse(BaseModel):
    answer: str
    sources_used: bool = False
    model: str = GEMINI_MODEL


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/status", summary="Check tutor availability")
def tutor_status():
    """Returns which tutor components are available."""
    return {
        "gemini_available": _GEMINI_AVAILABLE and bool(GOOGLE_API_KEY),
        "qdrant_available": _QDRANT_AVAILABLE,
        "llama_index_available": _LLAMA_AVAILABLE,
        "gemini_model": GEMINI_MODEL,
        "qdrant_host": f"{QDRANT_HOST}:{QDRANT_PORT}",
    }


@router.post("/ask", response_model=TutorResponse, summary="Ask the tutor a question")
def ask_tutor(body: TutorQuestion):
    """
    Ask the AI tutor a question.  If ``use_rag=true``, relevant course
    documents are retrieved from Qdrant and included in the context.
    """
    try:
        client = _get_gemini_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    context = ""
    sources_used = False

    if body.use_rag and _LLAMA_AVAILABLE and _QDRANT_AVAILABLE:
        try:
            qe = _get_query_engine()
            rag_response = qe.query(body.question)
            context = f"\n\n--- Relevant course material ---\n{rag_response.response}\n---\n"
            sources_used = True
        except Exception as exc:
            # RAG failure is non-fatal — fall back to pure LLM
            print(f"[Tutor] RAG retrieval failed: {exc}")

    prompt = f"{context}\nStudent question: {body.question}"
    if body.skill_id is not None:
        prompt += f"\n(Context: student is working on skill ID {body.skill_id})"

    try:
        response = client.generate_content(prompt)
        answer = response.text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {exc}")

    return TutorResponse(answer=answer, sources_used=sources_used)


@router.post("/explain-prediction", response_model=TutorResponse,
             summary="Explain a model prediction to the student")
def explain_prediction(body: ExplainPredictionRequest):
    """
    Given a model prediction for a student, generate a friendly natural-language
    explanation of why the model thinks they will (or won't) answer correctly.
    """
    try:
        client = _get_gemini_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    outcome = "answer correctly ✅" if body.prediction == 1 else "struggle ❌"
    confidence_pct = abs(body.confidence) * 100

    prompt = f"""
A machine learning model predicted that this student will {outcome} on their next question.

Student features:
- Skill ID:               {body.skill_id}
- Number of attempts:     {body.attempt_count}
- First response time:    {body.ms_first_response:.0f} ms
- Hints requested:        {body.hint_count}
- Overlap time fraction:  {body.overlap_time:.2f}

Model confidence score: {body.confidence:.3f} ({"high" if confidence_pct > 50 else "moderate"} confidence)

Please explain in simple, encouraging terms:
1. What these numbers suggest about the student's current understanding.
2. Why the model made this prediction.
3. One concrete tip the student can use to improve.

Keep the response concise (3-4 sentences) and encouraging.
"""

    try:
        response = client.generate_content(prompt)
        return TutorResponse(answer=response.text, sources_used=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {exc}")


@router.post("/practice-questions", response_model=TutorResponse,
             summary="Generate practice questions for a topic")
def generate_practice_questions(body: PracticeRequest):
    """
    Generate practice questions for a given topic/skill at a specified difficulty.
    """
    try:
        client = _get_gemini_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    prompt = f"""
Generate exactly {body.num_questions} {body.difficulty}-difficulty practice questions for the topic: "{body.topic}".

Format each question as:
Q1. <question text>
   A) <option>  B) <option>  C) <option>  D) <option>
   Answer: <correct letter>
   Explanation: <brief explanation>

Make the questions progressively more challenging and ensure they test conceptual understanding.
"""

    try:
        response = client.generate_content(prompt)
        return TutorResponse(answer=response.text, sources_used=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {exc}")


# ── CLI index builder ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index documents into Qdrant for RAG.")
    parser.add_argument("--index", metavar="DOCS_DIR", help="Directory of documents to index")
    args = parser.parse_args()

    if args.index:
        print(f"[Tutor] Indexing documents from: {args.index}")
        _get_query_engine(docs_dir=args.index)
        print("[Tutor] ✅ Indexing complete")
    else:
        parser.print_help()
