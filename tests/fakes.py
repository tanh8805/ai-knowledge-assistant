"""In-memory test doubles for the provider abstractions. Never used by the application."""

from app.llm.base import LLMClient, LLMResponse, Message, ToolSpec

FAKE_MODEL = "fake-model"


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order and records every call it receives."""

    def __init__(self, responses: list[LLMResponse | str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def generate(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.calls.append((list(messages), list(tools or [])))
        if not self._responses:
            return LLMResponse(content="fake answer", model=FAKE_MODEL)

        response = self._responses.pop(0)
        if isinstance(response, str):
            return LLMResponse(content=response, model=FAKE_MODEL)
        return response
