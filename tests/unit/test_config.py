from pathlib import Path

import pytest

from app.core.config import load_settings

CONFIG_YAML = """
app:
  name: test-app
  environment: test
  log_level: INFO
llm:
  provider: gemini
  temperature: 0.0
  models:
    openai: gpt-test
    gemini: gemini-test
embedding:
  provider: gemini
  dimension: 8
  models:
    openai: openai-embed-test
    gemini: gemini-embed-test
rag:
  chunk_size: 100
  chunk_overlap: 10
  top_k: 3
agent:
  max_steps: 4
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run from an empty directory so a developer's local .env is not picked up.
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db:5432/test")


def test_loads_values_from_yaml(config_path: Path) -> None:
    settings = load_settings(config_path)

    assert settings.app.name == "test-app"
    assert settings.rag.chunk_size == 100
    assert settings.agent.max_steps == 4
    assert settings.llm.model == "gemini-test"
    assert settings.embedding.model == "gemini-embed-test"


def test_environment_overrides_providers(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    settings = load_settings(config_path)

    assert settings.llm.model == "gpt-test"
    assert settings.embedding.model == "openai-embed-test"


def test_secrets_are_hidden_when_printed(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")

    settings = load_settings(config_path)

    assert "sk-very-secret" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-very-secret"
