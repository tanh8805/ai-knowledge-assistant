from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float  # cosine similarity: 1.0 means same direction


class VectorStore(ABC):
    """Stores document chunks with their embeddings and finds the most similar ones."""

    @abstractmethod
    def find_document(self, file_hash: str) -> int | None:
        """Return the id of an already stored document with this content hash."""

    @abstractmethod
    def add_document(
        self,
        filename: str,
        file_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Store a document and its chunks. Returns the new document id."""

    @abstractmethod
    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        """Return up to `top_k` chunks, most similar first."""
