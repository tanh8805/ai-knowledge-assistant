import hashlib
import logging
from dataclasses import dataclass

from app.documents.loaders import get_loader
from app.embeddings.base import EmbeddingClient
from app.rag.chunker import chunk_text
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document_id: int
    filename: str
    chunks_added: int
    already_exists: bool


class DocumentIngestor:
    """File bytes -> loader -> text -> chunks -> embeddings -> vector store."""

    def __init__(
        self,
        embedding: EmbeddingClient,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        # Hashing the content lets the same file be uploaded twice without duplicate chunks.
        file_hash = hashlib.sha256(data).hexdigest()
        existing_id = self._vector_store.find_document(file_hash)
        if existing_id is not None:
            return IngestResult(existing_id, filename, chunks_added=0, already_exists=True)

        text = get_loader(filename).load(data)
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise ValueError(f"'{filename}' contains no extractable text")

        embeddings = self._embedding.embed(chunks)
        document_id = self._vector_store.add_document(filename, file_hash, chunks, embeddings)
        logger.info("Ingested %s as document %d with %d chunks", filename, document_id, len(chunks))
        return IngestResult(document_id, filename, chunks_added=len(chunks), already_exists=False)
