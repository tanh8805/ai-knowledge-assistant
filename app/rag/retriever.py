from app.embeddings.base import EmbeddingClient
from app.vectorstore.base import SearchResult, VectorStore


class Retriever:
    """Finds the chunks most relevant to a question: embed the question, search the store."""

    def __init__(self, embedding: EmbeddingClient, vector_store: VectorStore, top_k: int) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        query_embedding = self._embedding.embed_query(query)
        return self._vector_store.search(query_embedding, top_k or self._top_k)
