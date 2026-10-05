from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque token some providers (Gemini) return with a tool call and
    # require to be sent back unchanged in the next request.
    signature: bytes | None = None


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    # Set on assistant messages that request tool calls.
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on tool messages: which call this is the result of.
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """What the LLM is told about a tool: its name, purpose and JSON-schema arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls)


class LLMClient(ABC):
    """Provider-independent chat model. Business logic depends only on this class."""

    @abstractmethod
    def generate(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        """Send the conversation (and optional tools) to the model and return its reply."""
