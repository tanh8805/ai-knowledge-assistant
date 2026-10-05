import pytest

from app.vectorstore.base import VectorStore
from app.vectorstore.memory import InMemoryVectorStore, cosine_similarity


def test_vector_store_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_search_returns_most_similar_chunks_first() -> None:
    store = InMemoryVectorStore()
    store.add_document(
        "notes.txt",
        "hash-1",
        chunks=["about cats", "about dogs", "about cars"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    results = store.search([0.1, 0.9, 0.0], top_k=2)

    assert [result.content for result in results] == ["about dogs", "about cats"]
    assert results[0].filename == "notes.txt"
    assert results[0].chunk_index == 1


def test_search_on_empty_store_returns_nothing() -> None:
    assert InMemoryVectorStore().search([1.0, 0.0], top_k=3) == []


def test_find_document_by_hash() -> None:
    store = InMemoryVectorStore()
    document_id = store.add_document("a.txt", "hash-a", ["text"], [[1.0]])

    assert store.find_document("hash-a") == document_id
    assert store.find_document("unknown") is None


def test_chunks_and_embeddings_must_have_same_length() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().add_document("a.txt", "hash-a", ["one", "two"], [[1.0]])


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
