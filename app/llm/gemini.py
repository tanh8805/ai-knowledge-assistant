from google import genai
from google.genai import types

from app.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        system_instruction = "\n\n".join(m.content for m in messages if m.role == "system")
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=self._temperature,
            tools=[to_gemini_tool(tools)] if tools else None,
            # The agent loop executes tools itself; stop the SDK from doing it.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=to_gemini_contents(messages),
            config=config,
        )
        return parse_response(response, self._model)


def to_gemini_tool(tools: list[ToolSpec]) -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.parameters,
            )
            for tool in tools
        ]
    )


def to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    """Convert the conversation, leaving out system messages (sent as system_instruction)."""
    contents = []
    for message in messages:
        if message.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=message.content)])
            )
        elif message.role == "assistant":
            contents.append(types.Content(role="model", parts=_assistant_parts(message)))
        elif message.role == "tool":
            response = types.FunctionResponse(
                id=message.tool_call_id,
                name=message.tool_name,
                response={"result": message.content},
            )
            contents.append(
                types.Content(role="user", parts=[types.Part(function_response=response)])
            )
    return contents


def _assistant_parts(message: Message) -> list[types.Part]:
    parts = [types.Part.from_text(text=message.content)] if message.content else []
    for call in message.tool_calls:
        function_call = types.FunctionCall(id=call.id, name=call.name, args=call.arguments)
        parts.append(types.Part(function_call=function_call, thought_signature=call.signature))
    return parts


def parse_response(response: types.GenerateContentResponse, model: str) -> LLMResponse:
    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else None

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, part in enumerate(parts or []):
        if part.function_call:
            call = part.function_call
            tool_calls.append(
                ToolCall(
                    id=call.id or f"call_{index}",
                    name=call.name or "",
                    arguments=call.args or {},
                    signature=part.thought_signature,
                )
            )
        elif part.text and not part.thought:
            texts.append(part.text)

    usage = response.usage_metadata
    return LLMResponse(
        content="".join(texts),
        model=response.model_version or model,
        usage=Usage(
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        ),
        tool_calls=tuple(tool_calls),
    )
