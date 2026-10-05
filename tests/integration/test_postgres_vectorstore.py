"""Runs against a real PostgreSQL + pgvector database.

Skipped unless TEST_DATABASE_URL is set, e.g. with `docker compose up -d postgres`:
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_assistant pytest
"""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, text

from app.db.database import create_session_factory
from app.db.models import EMBEDDING_DIMENSION, Base, Document
from app.vectorstore.postgres import PostgresVectorStore
from tests.fakes import FakeEmbeddingClient

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")

TEST_HASH = "integration-test-hash"


@pytest.fixture
def store() -> Iterator[PostgresVectorStore]:
    session_factory = create_session_factory(DATABASE_URL or "")
    with session_factory.begin() as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(session.connection())

    yield PostgresVectorStore(session_factory)

    with session_factory.begin() as session:
        session.execute(delete(Document).where(Document.file_hash == TEST_HASH))


def test_add_and_search_document(store: PostgresVectorStore) -> None:
    embedding = FakeEmbeddingClient(EMBEDDING_DIMENSION)
    chunks = ["postgres stores vectors", "bananas are yellow"]

    document_id = store.add_document("db.txt", TEST_HASH, chunks, embedding.embed(chunks))
    results = store.search(embedding.embed_query("postgres vectors"), top_k=1)

    assert store.find_document(TEST_HASH) == document_id
    assert results[0].content == "postgres stores vectors"
    assert results[0].filename == "db.txt"
    assert results[0].score > 0
