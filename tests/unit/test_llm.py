import pytest

from app.llm.base import LLMClient, LLMResponse, Message, ToolCall
from tests.fakes import FakeLLMClient


def test_llm_client_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_fake_llm_client_implements_llm_client() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


def test_generate_returns_llm_response() -> None:
    llm = FakeLLMClient(["Hello!"])

    response = llm.generate([Message(role="user", content="Hi")])

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello!"


def test_fake_records_messages_it_received() -> None:
    llm = FakeLLMClient()
    messages = [Message(role="system", content="Be brief."), Message(role="user", content="Hi")]

    llm.generate(messages)

    assert llm.calls[0][0] == messages


def test_response_converts_to_assistant_message_with_tool_calls() -> None:
    call = ToolCall(id="1", name="calculator", arguments={"expression": "1 + 1"})
    response = LLMResponse(content="", model="m", tool_calls=(call,))

    message = response.to_message()

    assert message.role == "assistant"
    assert message.tool_calls == (call,)
