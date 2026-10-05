from typing import Any

import pytest

from app.agents.base import Agent
from app.agents.tools import Tool
from app.llm.base import ToolSpec


class EchoTool(Tool):
    name = "echo"
    description = "Repeats the text it receives."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["text"])


def test_tool_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_agent_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_tool_spec_describes_tool_to_llm() -> None:
    assert EchoTool().spec() == ToolSpec(
        name="echo", description="Repeats the text it receives.", parameters=EchoTool.parameters
    )
