import math
from dataclasses import dataclass

from app.vectorstore.base import SearchResult, VectorStore


@dataclass(frozen=True)
class _StoredChunk:
    document_id: int
    filename: str
    chunk_index: int
    content: str
    embedding: list[float]


class InMemoryVectorStore(VectorStore):
    """Keeps everything in a Python list. Used in tests and for quick local experiments."""

    def __init__(self) -> None:
        self._documents: dict[str, int] = {}
        self._chunks: list[_StoredChunk] = []

    def find_document(self, file_hash: str) -> int | None:
        return self._documents.get(file_hash)

    def add_document(
        self,
        filename: str,
        file_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        document_id = len(self._documents) + 1
        self._documents[file_hash] = document_id
        for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            self._chunks.append(_StoredChunk(document_id, filename, index, content, embedding))
        return document_id

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        results = [
            SearchResult(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=cosine_similarity(embedding, chunk.embedding),
            )
            for chunk in self._chunks
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm
