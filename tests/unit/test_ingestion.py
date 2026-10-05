import pytest

from app.rag.ingestion import DocumentIngestor
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient

DIMENSION = 32


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def ingestor(store: InMemoryVectorStore) -> DocumentIngestor:
    return DocumentIngestor(FakeEmbeddingClient(DIMENSION), store, chunk_size=20, chunk_overlap=5)


def test_ingest_chunks_embeds_and_stores_document(
    ingestor: DocumentIngestor, store: InMemoryVectorStore
) -> None:
    result = ingestor.ingest("notes.txt", b"A short document that needs a few chunks.")

    assert result.chunks_added > 1
    assert not result.already_exists
    assert store.search([1.0] * DIMENSION, top_k=1)[0].filename == "notes.txt"


def test_ingesting_same_content_twice_does_not_duplicate(ingestor: DocumentIngestor) -> None:
    first = ingestor.ingest("notes.txt", b"Same content")
    second = ingestor.ingest("copy.txt", b"Same content")

    assert second.already_exists
    assert second.document_id == first.document_id
    assert second.chunks_added == 0


def test_document_without_text_is_rejected(ingestor: DocumentIngestor) -> None:
    with pytest.raises(ValueError, match="no extractable text"):
        ingestor.ingest("empty.md", b"   ")


def test_unsupported_file_type_is_rejected(ingestor: DocumentIngestor) -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestor.ingest("sheet.xlsx", b"data")
