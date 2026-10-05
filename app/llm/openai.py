import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from app.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        optional: dict[str, Any] = {}
        if tools:
            optional["tools"] = [to_openai_tool(tool) for tool in tools]

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[to_openai_message(message) for message in messages],
            temperature=self._temperature,
            **optional,
        )
        return parse_completion(completion)


def to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def to_openai_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }

    converted: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        converted["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return converted


def parse_completion(completion: ChatCompletion) -> LLMResponse:
    message = completion.choices[0].message
    tool_calls = tuple(
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=json.loads(call.function.arguments or "{}"),
        )
        for call in message.tool_calls or []
        if call.type == "function"
    )
    usage = completion.usage
    return LLMResponse(
        content=message.content or "",
        model=completion.model,
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
        tool_calls=tool_calls,
    )
