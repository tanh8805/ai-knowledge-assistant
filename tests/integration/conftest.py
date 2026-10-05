from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.main import app
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient

DIMENSION = 32


@pytest.fixture
def client() -> Iterator[TestClient]:
    """API client wired to in-memory fakes: no database, no API keys, no network."""
    vector_store = InMemoryVectorStore()
    embedding = FakeEmbeddingClient(DIMENSION)
    app.dependency_overrides[dependencies.get_vector_store] = lambda: vector_store
    app.dependency_overrides[dependencies.get_embedding_client] = lambda: embedding

    yield TestClient(app)

    app.dependency_overrides.clear()
