"""Persistent, content-addressed course PDFs for the shared tutor library."""

import hashlib
import io
import logging
import threading
from pathlib import PureWindowsPath

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
                raise HTTPException(413, "This PDF contains too much text. Upload a chapter instead.")
            if text:
                documents.append(Document(
                    text=text, id_=f"{digest}/{number}",
                    metadata={"source": filename, "page": number},
                ))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Unable to read this PDF. Try exporting it again.") from exc
    if not documents:
        raise HTTPException(422, "No readable text found. Apply OCR to scanned pages first.")
    return page_count, documents


class PdfLibrary:
    """Publish a collection after all its page chunks have been embedded."""

    def __init__(self, client, namespace: str, index_for):
        self.client = client
        self.index_for = index_for
        scope = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        self.prefix = f"course-pdf-{scope}-"

    def books(self) -> list[LibraryBook]:
        books = [
            LibraryBook.model_validate(collection.metadata)
            for collection in self.client.list_collections()
            if collection.name.startswith(self.prefix) and (collection.metadata or {}).get("ready")
        ]
        return sorted(books, key=lambda book: (book.filename.casefold(), book.id))

    def lookup(self, digest: str):
        try:
            return self.client.get_collection(self.prefix + digest)
        except NotFoundError:
            return None

    def index(self, digest: str):
        collection = self.lookup(digest)
        if collection is None or not (collection.metadata or {}).get("ready"):
            raise HTTPException(404, "This PDF is unavailable. Select another library source.")
        return self.index_for(collection)

    def upload(self, filename: str, data: bytes) -> LibraryBook:
        name = PureWindowsPath(filename).name.strip()
        if not name.lower().endswith(".pdf") or len(name) > 200 or not name.isprintable():
            raise HTTPException(415, "Choose a file with a valid PDF filename.")
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
                self.prefix + digest, metadata={"ready": False},
                configuration={"hnsw": {"space": "cosine"}},
            )
            try:
                splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                nodes = splitter.get_nodes_from_documents(documents)
                self.index_for(collection).insert_nodes(nodes)
                book = LibraryBook(id=digest, filename=name, pages=pages,
                                   pages_indexed=len(documents), chunks=len(nodes))
                collection.modify(metadata={**book.model_dump(), "ready": True})
                return book
            except Exception as exc:
                logger.exception("PDF indexing failed for %s", digest)
                self.client.delete_collection(collection.name)
                raise HTTPException(503, "Indexing failed. Please retry the upload.") from exc
