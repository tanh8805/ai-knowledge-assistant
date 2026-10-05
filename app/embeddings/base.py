from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Provider-independent text embedding model."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector of length `dimension` per input text, in the same order."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
