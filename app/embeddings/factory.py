from app.core.config import Settings
from app.embeddings.base import EmbeddingClient
from app.embeddings.gemini import GeminiEmbeddingClient
from app.embeddings.openai import OpenAIEmbeddingClient


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Build the embedding client for the provider selected in configuration."""
    config = settings.embedding
    api_key = settings.api_key_for(config.provider)
    if config.provider == "openai":
        return OpenAIEmbeddingClient(api_key, config.model, config.dimension)
    return GeminiEmbeddingClient(api_key, config.model, config.dimension)
