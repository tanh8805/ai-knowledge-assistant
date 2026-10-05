"""Provider conversion tests. They build SDK objects locally and never call an API."""

import pytest
from google.genai import types
from openai.types.chat import ChatCompletion
from pydantic import SecretStr

from app.core.config import MissingAPIKeyError, get_settings
from app.llm.base import Message, ToolCall
from app.llm.factory import create_llm_client
from app.llm.gemini import GeminiClient, parse_response, to_gemini_contents
from app.llm.openai import OpenAIClient, parse_completion, to_openai_message

CALL = ToolCall(id="call_1", name="calculator", arguments={"expression": "2 * 3"})


def test_openai_assistant_tool_call_is_serialized_as_json() -> None:
    message = Message(role="assistant", content="", tool_calls=(CALL,))

    converted = to_openai_message(message)

    assert converted["tool_calls"][0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression": "2 * 3"}',
    }


def test_openai_tool_result_references_call_id() -> None:
    message = Message(role="tool", content="6", tool_call_id="call_1", tool_name="calculator")

    assert to_openai_message(message) == {"role": "tool", "tool_call_id": "call_1", "content": "6"}


def test_openai_completion_is_parsed_into_llm_response() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression": "2 * 3"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )

    response = parse_completion(completion)

    assert response.content == ""
    assert response.model == "gpt-test"
    assert response.tool_calls == (CALL,)
    assert response.usage.input_tokens == 10


def test_gemini_contents_skip_system_and_map_roles() -> None:
    messages = [
        Message(role="system", content="Be brief."),
        Message(role="user", content="What is 2 * 3?"),
        Message(role="assistant", content="", tool_calls=(CALL,)),
        Message(role="tool", content="6", tool_call_id="call_1", tool_name="calculator"),
    ]

    contents = to_gemini_contents(messages)

    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "calculator"
    assert contents[2].parts[0].function_response.response == {"result": "6"}


def test_gemini_response_keeps_thought_signature() -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_1", name="calculator", args={"expression": "2 * 3"}
                            ),
                            thought_signature=b"sig",
                        )
                    ],
                )
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=7, candidates_token_count=3
        ),
    )

    parsed = parse_response(response, model="gemini-test")

    assert parsed.tool_calls == (
        ToolCall(id="call_1", name="calculator", arguments={"expression": "2 * 3"}, signature=b"sig"),
    )
    assert parsed.usage.output_tokens == 3


@pytest.mark.parametrize(("provider", "expected"), [("openai", OpenAIClient), ("gemini", GeminiClient)])
def test_factory_builds_selected_provider(provider: str, expected: type) -> None:
    settings = get_settings().model_copy(deep=True)
    settings.llm.provider = provider
    settings.openai_api_key = SecretStr("test-key")
    settings.gemini_api_key = SecretStr("test-key")

    assert isinstance(create_llm_client(settings), expected)


def test_factory_requires_api_key() -> None:
    settings = get_settings().model_copy(deep=True)
    settings.llm.provider = "openai"
    settings.openai_api_key = None

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        create_llm_client(settings)
