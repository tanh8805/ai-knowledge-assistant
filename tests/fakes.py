"""In-memory test doubles for the provider abstractions. Never used by the application."""

import zlib

from app.embeddings.base import EmbeddingClient
from app.llm.base import LLMClient, LLMResponse, Message, ToolSpec

FAKE_MODEL = "fake-model"


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order and records every call it receives."""

    def __init__(self, responses: list[LLMResponse | str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def generate(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        self.calls.append((list(messages), list(tools or [])))
        if not self._responses:
            return LLMResponse(content="fake answer", model=FAKE_MODEL)

        response = self._responses.pop(0)
        if isinstance(response, str):
            return LLMResponse(content=response, model=FAKE_MODEL)
        return response


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic bag-of-words embedding: texts sharing words get similar vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % self.dimension] += 1.0
        return vector
