def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into fixed-size character windows that overlap by `chunk_overlap`.

    The overlap keeps a sentence that crosses a chunk boundary visible in both chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks
