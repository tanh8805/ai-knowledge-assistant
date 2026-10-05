from google import genai
from google.genai import types

from app.embeddings.base import EmbeddingClient

# Gemini accepts up to 100 texts per embed request.
BATCH_SIZE = 100


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        super().__init__(dimension)
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        config = types.EmbedContentConfig(output_dimensionality=self.dimension)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts[start : start + BATCH_SIZE],
                config=config,
            )
            vectors.extend(list(embedding.values or []) for embedding in response.embeddings or [])
        return vectors
