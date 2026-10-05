from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

Provider = Literal["openai", "gemini"]


class MissingAPIKeyError(RuntimeError):
    """The selected provider has no API key configured."""


class AppConfig(BaseModel):
    name: str
    environment: str
    log_level: str


class LLMConfig(BaseModel):
    provider: Provider
    temperature: float
    models: dict[Provider, str]

    @property
    def model(self) -> str:
        return self.models[self.provider]


class EmbeddingConfig(BaseModel):
    provider: Provider
    dimension: int
    models: dict[Provider, str]

    @property
    def model(self) -> str:
        return self.models[self.provider]


class RAGConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int


class AgentConfig(BaseModel):
    max_steps: int


class EnvSettings(BaseSettings):
    """Secrets and environment-specific values, read from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    database_url: str
    llm_provider: Provider | None = None
    embedding_provider: Provider | None = None


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    rag: RAGConfig
    agent: AgentConfig
    openai_api_key: SecretStr | None
    gemini_api_key: SecretStr | None
    database_url: str

    def api_key_for(self, provider: Provider) -> str:
        key = self.openai_api_key if provider == "openai" else self.gemini_api_key
        if key is None or not key.get_secret_value():
            raise MissingAPIKeyError(f"{provider.upper()}_API_KEY is not set")
        return key.get_secret_value()


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
    """Merge config.yaml with environment values. Environment values win."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    env = EnvSettings()
    if env.llm_provider:
        config["llm"]["provider"] = env.llm_provider
    if env.embedding_provider:
        config["embedding"]["provider"] = env.embedding_provider

    return Settings(
        **config,
        openai_api_key=env.openai_api_key,
        gemini_api_key=env.gemini_api_key,
        database_url=env.database_url,
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()
