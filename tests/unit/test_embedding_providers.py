"""Provider tests with the SDK call replaced by a local stub. No network access."""

from typing import Any

import pytest
from google.genai import types
from openai.types import CreateEmbeddingResponse
from pydantic import SecretStr

from app.core.config import get_settings
from app.embeddings import gemini
from app.embeddings.factory import create_embedding_client
from app.embeddings.gemini import GeminiEmbeddingClient
from app.embeddings.openai import OpenAIEmbeddingClient

DIMENSION = 3


def test_openai_embeddings_are_returned_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIEmbeddingClient("test-key", "embed-model", DIMENSION)
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> CreateEmbeddingResponse:
        requests.append(kwargs)
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "embed-model",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(client._client.embeddings, "create", create)

    vectors = client.embed(["first", "second"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert requests[0]["dimensions"] == DIMENSION


def test_gemini_embeddings_are_requested_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini, "BATCH_SIZE", 2)
    client = GeminiEmbeddingClient("test-key", "embed-model", DIMENSION)
    batches: list[list[str]] = []

    def embed_content(**kwargs: Any) -> types.EmbedContentResponse:
        batches.append(kwargs["contents"])
        return types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(values=[1.0] * DIMENSION) for _ in kwargs["contents"]
            ]
        )

    monkeypatch.setattr(client._client.models, "embed_content", embed_content)

    vectors = client.embed(["a", "b", "c"])

    assert batches == [["a", "b"], ["c"]]
    assert len(vectors) == 3


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("openai", OpenAIEmbeddingClient), ("gemini", GeminiEmbeddingClient)],
)
def test_factory_builds_selected_provider(provider: str, expected: type) -> None:
    settings = get_settings().model_copy(deep=True)
    settings.embedding.provider = provider
    settings.openai_api_key = SecretStr("test-key")
    settings.gemini_api_key = SecretStr("test-key")

    client = create_embedding_client(settings)

    assert isinstance(client, expected)
    assert client.dimension == settings.embedding.dimension
