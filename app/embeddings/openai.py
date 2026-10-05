from openai import OpenAI

from app.embeddings.base import EmbeddingClient

# OpenAI accepts up to 2048 inputs per embeddings request.
BATCH_SIZE = 2048


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        super().__init__(dimension)
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self._client.embeddings.create(
                model=self._model,
                input=texts[start : start + BATCH_SIZE],
                dimensions=self.dimension,
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors
