"""Supabase Postgres + pgvector RAG store and PDF library backend (Milestone 1).

Used automatically by ``app/tutor.py`` when ``DATABASE_URL`` is set. Embeds text
with the same CPU-only ONNX ``all-MiniLM-L6-v2`` (384-dim) model used locally,
stores chunks in ``rag_chunks (vector(384))``, tracks uploaded PDFs in
``rag_documents``, and optionally persists raw PDF bytes to Supabase Storage.

Failure policy: preserves the hard HTTP 500 contract on all DB/embedding/storage
failures (never returns canned fallback answers).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from fastapi import HTTPException
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.tutor_library import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LibraryBook,
    read_pages,
    sanitize_pdf_filename,
)
from src.storage import SCHEMA_SQL_PATH, SupabaseObjectStorage

logger = logging.getLogger(__name__)


def format_pgvector(vector: List[float]) -> str:
    """Serialize a float embedding vector into Postgres ``vector`` literal syntax."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


@dataclass
class RetrievedChunk:
    """Lightweight retrieved passage matching the fields ``app/tutor.py`` reads."""

    source: str
    text: str
    page: Optional[int] = None
    score: Optional[float] = None


class PgVectorTutorStore:
    """Postgres + pgvector(384) store for curriculum chunks and uploaded PDFs."""

    def __init__(self, dsn: str, auto_init_schema: bool = True) -> None:
        self.dsn = dsn
        self._schema_ready = False
        if auto_init_schema:
            self.ensure_schema()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, prepare_threshold=None)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        sql_text = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
            conn.commit()
        self._schema_ready = True

    def count(self, collection: str) -> int:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*)::bigint FROM rag_chunks WHERE collection = %s",
                    (collection,),
                )
                row = cur.fetchone()
        return int(row[0] if row else 0)

    def upsert_documents(
        self,
        collection: str,
        documents: List[Document],
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> int:
        """Idempotently replace chunks for each ``doc.doc_id`` in ``collection``."""
        self.ensure_schema()
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)
        texts = [node.get_content() for node in nodes]
        embeddings = embed_fn(texts) if texts else []

        doc_ids = [doc.doc_id for doc in documents]
        positions: dict[str, int] = {}
        rows = []
        for node, text, vec in zip(nodes, texts, embeddings):
            ref = node.ref_doc_id or node.node_id
            idx = positions.get(ref, 0)
            positions[ref] = idx + 1
            chunk_id = hashlib.sha1(f"{collection}::{ref}::{idx}".encode("utf-8")).hexdigest()
            meta = dict(node.metadata or {})
            source = str(meta.get("source", "unknown"))
            raw_page = meta.get("page")
            page = int(raw_page) if raw_page is not None else None
            rows.append(
                (
                    chunk_id,
                    collection,
                    ref,
                    source,
                    page,
                    idx,
                    text,
                    json.dumps(meta, default=str),
                    format_pgvector(vec),
                )
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                if doc_ids:
                    cur.execute(
                        "DELETE FROM rag_chunks WHERE collection = %s AND document_id = ANY(%s)",
                        (collection, doc_ids),
                    )
                if rows:
                    cur.executemany(
                        """
                        INSERT INTO rag_chunks (
                            chunk_id, collection, document_id, source, page,
                            chunk_index, content, metadata, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )
            conn.commit()
        return len(rows)

    def query(
        self,
        collection: str,
        query_text: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
        top_k: int = 3,
    ) -> List[RetrievedChunk]:
        self.ensure_schema()
        vec_lit = format_pgvector(embed_fn([query_text])[0])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source, page, content,
                           1.0 - (embedding <=> %s::vector) AS score
                    FROM rag_chunks
                    WHERE collection = %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                    """,
                    (vec_lit, collection, vec_lit, int(top_k)),
                )
                rows = cur.fetchall()
        return [
            RetrievedChunk(
                source=str(r[0]),
                page=int(r[1]) if r[1] is not None else None,
                text=str(r[2]),
                score=round(float(r[3]), 4) if r[3] is not None else None,
            )
            for r in rows
        ]

    # -- PDF Library operations ---------------------------------------------

    def pdf_collection_name(self, namespace: str, digest: str) -> str:
        scope = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        return f"course-pdf-{scope}-{digest}"

    def list_pdf_books(self, namespace: str) -> List[LibraryBook]:
        self.ensure_schema()
        scope = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        prefix = f"course-pdf-{scope}-%"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id, filename, pages, pages_indexed, chunks
                    FROM rag_documents
                    WHERE ready = TRUE AND collection LIKE %s
                    ORDER BY LOWER(filename) ASC, document_id ASC
                    """,
                    (prefix,),
                )
                rows = cur.fetchall()
        return [
            LibraryBook(
                id=str(r[0]),
                filename=str(r[1]),
                pages=int(r[2]),
                pages_indexed=int(r[3]),
                chunks=int(r[4]),
            )
            for r in rows
        ]

    def pdf_stats(self, namespace: str) -> tuple[int, int]:
        books = self.list_pdf_books(namespace)
        return len(books), sum(b.chunks for b in books)

    def has_ready_pdf(self, namespace: str, digest: str) -> bool:
        self.ensure_schema()
        col = self.pdf_collection_name(namespace, digest)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ready FROM rag_documents
                    WHERE document_id = %s AND collection = %s
                    """,
                    (digest, col),
                )
                row = cur.fetchone()
        return bool(row and row[0])

    def upload_pdf(
        self,
        namespace: str,
        filename: str,
        data: bytes,
        embed_fn: Callable[[List[str]], List[List[float]]],
        object_storage: Optional[SupabaseObjectStorage] = None,
    ) -> LibraryBook:
        name = sanitize_pdf_filename(filename)
        digest = hashlib.sha256(data).hexdigest()
        col = self.pdf_collection_name(namespace, digest)
        self.ensure_schema()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id, filename, pages, pages_indexed, chunks, ready
                    FROM rag_documents
                    WHERE document_id = %s AND collection = %s
                    """,
                    (digest, col),
                )
                existing = cur.fetchone()
        if existing and existing[5]:
            return LibraryBook(
                id=str(existing[0]),
                filename=str(existing[1]),
                pages=int(existing[2]),
                pages_indexed=int(existing[3]),
                chunks=int(existing[4]),
            )

        pages, documents = read_pages(data, name, digest)
        storage_path: Optional[str] = None
        try:
            if object_storage is not None:
                storage_path = object_storage.upload_pdf(digest, name, data)

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM rag_chunks WHERE collection = %s", (col,))
                    cur.execute(
                        """
                        INSERT INTO rag_documents (
                            document_id, collection, filename, pages,
                            pages_indexed, chunks, ready, storage_path
                        ) VALUES (%s, %s, %s, %s, %s, 0, FALSE, %s)
                        ON CONFLICT (document_id) DO UPDATE SET
                            collection = EXCLUDED.collection,
                            filename = EXCLUDED.filename,
                            pages = EXCLUDED.pages,
                            pages_indexed = EXCLUDED.pages_indexed,
                            chunks = 0,
                            ready = FALSE,
                            storage_path = EXCLUDED.storage_path
                        """,
                        (digest, col, name, pages, len(documents), storage_path),
                    )
                conn.commit()

            chunk_count = self.upsert_documents(col, documents, embed_fn)

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE rag_documents
                        SET chunks = %s, ready = TRUE
                        WHERE document_id = %s AND collection = %s
                        """,
                        (chunk_count, digest, col),
                    )
                conn.commit()

            return LibraryBook(
                id=digest,
                filename=name,
                pages=pages,
                pages_indexed=len(documents),
                chunks=chunk_count,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("pgvector PDF indexing failed for %s", digest)
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM rag_chunks WHERE collection = %s", (col,))
                        cur.execute(
                            "DELETE FROM rag_documents WHERE document_id = %s AND collection = %s",
                            (digest, col),
                        )
                    conn.commit()
            except Exception:
                pass
            raise HTTPException(
                500, f"PDF indexing failed: {type(exc).__name__}: {exc}"
            ) from exc

    def query_pdf(
        self,
        namespace: str,
        digest: str,
        query_text: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
        top_k: int = 3,
    ) -> List[RetrievedChunk]:
        if not self.has_ready_pdf(namespace, digest):
            raise HTTPException(404, "This PDF is unavailable. Select another library source.")
        col = self.pdf_collection_name(namespace, digest)
        return self.query(col, query_text, embed_fn, top_k=top_k)

    def delete_pdf(
        self,
        namespace: str,
        digest: str,
        object_storage: Optional[SupabaseObjectStorage] = None,
    ) -> bool:
        self.ensure_schema()
        col = self.pdf_collection_name(namespace, digest)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT storage_path FROM rag_documents "
                    "WHERE document_id = %s AND collection = %s",
                    (digest, col),
                )
                row = cur.fetchone()
                if not row:
                    return False
                storage_path = row[0]
                cur.execute("DELETE FROM rag_chunks WHERE collection = %s", (col,))
                cur.execute(
                    "DELETE FROM rag_documents WHERE document_id = %s AND collection = %s",
                    (digest, col),
                )
            conn.commit()
        if object_storage is not None and storage_path:
            try:
                object_storage.delete_pdf(str(storage_path))
            except Exception as exc:
                logger.warning("Failed to delete PDF object %s: %s", storage_path, exc)
        return True

    def clear_all_pdfs(
        self,
        namespace: str,
        object_storage: Optional[SupabaseObjectStorage] = None,
    ) -> int:
        self.ensure_schema()
        scope = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        prefix = f"course-pdf-{scope}-%"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, storage_path FROM rag_documents WHERE collection LIKE %s",
                    (prefix,),
                )
                rows = cur.fetchall()
                cur.execute("DELETE FROM rag_chunks WHERE collection LIKE %s", (prefix,))
                cur.execute("DELETE FROM rag_documents WHERE collection LIKE %s", (prefix,))
            conn.commit()
        if object_storage is not None:
            for _, spath in rows:
                if spath:
                    try:
                        object_storage.delete_pdf(str(spath))
                    except Exception as exc:
                        logger.warning("Failed to delete PDF object %s: %s", spath, exc)
        return len(rows)
