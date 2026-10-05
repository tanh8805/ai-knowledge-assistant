from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.llm.openai import OpenAIClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the LLM client for the provider selected in configuration."""
    config = settings.llm
    api_key = settings.api_key_for(config.provider)
    if config.provider == "openai":
        return OpenAIClient(api_key, config.model, config.temperature)
    return GeminiClient(api_key, config.model, config.temperature)
