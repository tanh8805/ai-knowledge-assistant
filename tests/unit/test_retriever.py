from app.rag.retriever import Retriever
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient

DIMENSION = 64
CHUNKS = [
    "Python is a programming language",
    "PostgreSQL is a relational database",
    "Bananas are a yellow fruit",
]


def make_retriever(top_k: int = 2) -> Retriever:
    embedding = FakeEmbeddingClient(DIMENSION)
    store = InMemoryVectorStore()
    store.add_document("facts.txt", "hash", CHUNKS, embedding.embed(CHUNKS))
    return Retriever(embedding, store, top_k=top_k)


def test_retrieve_returns_most_relevant_chunk_first() -> None:
    results = make_retriever().retrieve("which database is relational")

    assert results[0].content == "PostgreSQL is a relational database"


def test_retrieve_uses_default_top_k() -> None:
    assert len(make_retriever(top_k=2).retrieve("anything")) == 2


def test_retrieve_top_k_can_be_overridden() -> None:
    assert len(make_retriever(top_k=2).retrieve("anything", top_k=1)) == 1
