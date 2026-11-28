"""Persistent, content-addressed course PDFs for the shared tutor library."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from pathlib import PureWindowsPath
from typing import Any, Callable

from chromadb.errors import NotFoundError
from fastapi import HTTPException
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from pydantic import BaseModel
from pypdf import PdfReader

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT = 500_000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
logger = logging.getLogger(__name__)
# Embedded Chroma runs in a single server process. Only one upload writes at a time.
_writer = threading.Lock()


class LibraryBook(BaseModel):
    id: str
    filename: str
    pages: int
    pages_indexed: int
    chunks: int


def sanitize_pdf_filename(filename: str) -> str:
    """Strip any path components and enforce a safe, non-empty PDF filename."""
    cleaned = os.path.basename(PureWindowsPath(filename).name).strip()
    stem = cleaned[:-4].strip() if cleaned.lower().endswith(".pdf") else ""
    if (
        not cleaned
        or not cleaned.lower().endswith(".pdf")
        or not stem
        or stem in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or len(cleaned) > 200
        or not cleaned.isprintable()
    ):
        raise HTTPException(415, "Choose a file with a valid PDF filename.")
    return cleaned


def read_pages(data: bytes, filename: str, digest: str) -> tuple[int, list[Document]]:
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "Choose a valid PDF file.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise HTTPException(422, "Remove the PDF password and upload it again.")
        page_count = len(reader.pages)
        if page_count > MAX_PAGES:
            raise HTTPException(413, f"Choose a PDF with at most {MAX_PAGES} pages.")
        documents = []
        characters = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            characters += len(text)
            if characters > MAX_TEXT:
                raise HTTPException(
                    413, "This PDF contains too much text. Upload a chapter instead."
                )
            if text:
                documents.append(
                    Document(
                        text=text,
                        id_=f"{digest}/{number}",
                        metadata={"source": filename, "page": number},
                    )
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Unable to read this PDF. Try exporting it again.") from exc
    if not documents:
        raise HTTPException(422, "No readable text found. Apply OCR to scanned pages first.")
    return page_count, documents


class PdfLibrary:
    """Publish a collection after all its page chunks have been embedded."""

    def __init__(self, client: Any, namespace: str, index_for: Callable[[Any], Any]):
        self.client = client
        self.index_for = index_for
        scope = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        self.prefix = f"course-pdf-{scope}-"

    def _iter_ready_collections(self) -> list[Any]:
        ready = []
        for item in self.client.list_collections():
            if isinstance(item, str):
                if not item.startswith(self.prefix):
                    continue
                collection = self.client.get_collection(item)
            else:
                name = getattr(item, "name", "")
                if not name.startswith(self.prefix):
                    continue
                collection = item
            if (getattr(collection, "metadata", None) or {}).get("ready"):
                ready.append(collection)
        return ready

    def books(self) -> list[LibraryBook]:
        books = [
            LibraryBook.model_validate(collection.metadata)
            for collection in self._iter_ready_collections()
        ]
        return sorted(books, key=lambda book: (book.filename.casefold(), book.id))

    def stats(self) -> tuple[int, int]:
        """Return ``(ready_pdf_count, total_pdf_chunks)`` for monitoring gauges."""
        books = self.books()
        return len(books), sum(book.chunks for book in books)

    def lookup(self, digest: str):
        try:
            return self.client.get_collection(self.prefix + digest)
        except (NotFoundError, ValueError):
            return None

    def index(self, digest: str):
        collection = self.lookup(digest)
        if collection is None or not (collection.metadata or {}).get("ready"):
            raise HTTPException(404, "This PDF is unavailable. Select another library source.")
        return self.index_for(collection)

    def upload(self, filename: str, data: bytes) -> LibraryBook:
        name = sanitize_pdf_filename(filename)
        digest = hashlib.sha256(data).hexdigest()
        with _writer:
            collection = self.lookup(digest)
            if collection is not None and (collection.metadata or {}).get("ready"):
                return LibraryBook.model_validate(collection.metadata)
            pages, documents = read_pages(data, name, digest)
            # A process crash can leave an incomplete collection. Rebuild it on retry.
            if collection is not None:
                self.client.delete_collection(collection.name)
            collection = self.client.create_collection(
                self.prefix + digest,
                metadata={"ready": False},
                configuration={"hnsw": {"space": "cosine"}},
            )
            try:
                splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                nodes = splitter.get_nodes_from_documents(documents)
                self.index_for(collection).insert_nodes(nodes)
                book = LibraryBook(
                    id=digest,
                    filename=name,
                    pages=pages,
                    pages_indexed=len(documents),
                    chunks=len(nodes),
                )
                collection.modify(metadata={**book.model_dump(), "ready": True})
                return book
            except Exception as exc:
                logger.exception("PDF indexing failed for %s", digest)
                try:
                    self.client.delete_collection(collection.name)
                except Exception:
                    pass
                # Hard HTTP 500 contract on internal tutor/embedding/Chroma failures
                raise HTTPException(
                    500, f"PDF indexing failed: {type(exc).__name__}: {exc}"
                ) from exc

    def delete(self, digest: str) -> bool:
        with _writer:
            collection = self.lookup(digest)
            if collection is None:
                return False
            self.client.delete_collection(collection.name)
            return True

    def clear_all(self) -> int:
        removed = 0
        with _writer:
            for collection in list(self._iter_ready_collections()):
                self.client.delete_collection(collection.name)
                removed += 1
        return removed
