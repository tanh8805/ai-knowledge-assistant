import pytest

from app.embeddings.base import EmbeddingClient
from tests.fakes import FakeEmbeddingClient

DIMENSION = 16


def test_embedding_client_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        EmbeddingClient(DIMENSION)  # type: ignore[abstract]


def test_fake_embedding_client_implements_embedding_client() -> None:
    assert isinstance(FakeEmbeddingClient(DIMENSION), EmbeddingClient)


def test_embed_returns_one_vector_per_text_with_configured_dimension() -> None:
    vectors = FakeEmbeddingClient(DIMENSION).embed(["first text", "second text", "third"])

    assert len(vectors) == 3
    assert all(len(vector) == DIMENSION for vector in vectors)


def test_embed_is_deterministic() -> None:
    client = FakeEmbeddingClient(DIMENSION)

    assert client.embed_query("same input") == client.embed_query("same input")
