from typing import Any

from app.agents.tool_agent import MAX_STEPS_ANSWER, ToolCallingAgent
from app.agents.tools import Tool
from app.llm.base import LLMResponse, ToolCall
from tests.fakes import FakeLLMClient


class UppercaseTool(Tool):
    name = "uppercase"
    description = "Converts text to upper case."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments: dict[str, Any]) -> str:
        if "text" not in arguments:
            raise ValueError("'text' is required")
        return str(arguments["text"]).upper()


class UppercaseAgent(ToolCallingAgent):
    name = "uppercase"
    description = "Shouts."
    system_prompt = "Use the uppercase tool."


def tool_call(name: str = "uppercase", **arguments: Any) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake",
        tool_calls=(ToolCall(id="call_1", name=name, arguments=arguments),),
    )


def test_answers_directly_when_no_tool_is_needed() -> None:
    llm = FakeLLMClient(["Hello!"])

    answer = UppercaseAgent(llm, [UppercaseTool()], max_steps=5).run("Say hello")

    assert answer == "Hello!"
    assert len(llm.calls) == 1


def test_llm_is_offered_the_agent_tools() -> None:
    llm = FakeLLMClient(["Hello!"])

    UppercaseAgent(llm, [UppercaseTool()], max_steps=5).run("Say hello")

    _, tools = llm.calls[0]
    assert [tool.name for tool in tools] == ["uppercase"]


def test_selected_tool_is_executed_and_result_sent_back_to_llm() -> None:
    llm = FakeLLMClient([tool_call(text="hi"), "The result is HI"])

    answer = UppercaseAgent(llm, [UppercaseTool()], max_steps=5).run("Uppercase 'hi'")

    assert answer == "The result is HI"
    second_call_messages, _ = llm.calls[1]
    tool_message = second_call_messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.content == "HI"
    assert tool_message.tool_call_id == "call_1"


def test_tool_errors_are_reported_to_llm() -> None:
    llm = FakeLLMClient([tool_call(), "Sorry, I need some text."])

    UppercaseAgent(llm, [UppercaseTool()], max_steps=5).run("Uppercase nothing")

    assert llm.calls[1][0][-1].content == "Error: 'text' is required"


def test_unknown_tool_is_reported_to_llm() -> None:
    llm = FakeLLMClient([tool_call(name="delete_everything"), "Done."])

    UppercaseAgent(llm, [UppercaseTool()], max_steps=5).run("Do something")

    assert llm.calls[1][0][-1].content == "Error: unknown tool 'delete_everything'"


def test_stops_after_max_steps() -> None:
    llm = FakeLLMClient([tool_call(text="again")] * 10)

    answer = UppercaseAgent(llm, [UppercaseTool()], max_steps=3).run("Loop forever")

    assert answer == MAX_STEPS_ANSWER
    assert len(llm.calls) == 3
