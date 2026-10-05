import pytest

from app.rag.chunker import chunk_text


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("Hello world", chunk_size=50, chunk_overlap=5) == ["Hello world"]


def test_chunks_respect_chunk_size() -> None:
    chunks = chunk_text("a" * 105, chunk_size=20, chunk_overlap=5)

    assert all(len(chunk) <= 20 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=2)

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_chunks_cover_the_whole_text() -> None:
    text = "0123456789" * 7
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=0)

    assert "".join(chunks) == text


def test_empty_or_blank_input_returns_no_chunks() -> None:
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []
    assert chunk_text("   \n\t", chunk_size=10, chunk_overlap=2) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=size, chunk_overlap=overlap)
