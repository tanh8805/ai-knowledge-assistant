from app.core.config import get_settings
from app.db.models import EMBEDDING_DIMENSION, Chunk


def test_embedding_column_matches_configured_dimension() -> None:
    assert get_settings().embedding.dimension == EMBEDDING_DIMENSION


def test_chunk_embedding_column_uses_dimension() -> None:
    assert Chunk.__table__.c.embedding.type.dim == EMBEDDING_DIMENSION
