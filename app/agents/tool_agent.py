import logging
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.base import Agent
from app.agents.tools import Tool
from app.llm.base import LLMClient, Message, ToolCall

logger = logging.getLogger(__name__)

MAX_STEPS_ANSWER = "I could not finish this task within the allowed number of steps."


@dataclass
class AgentState:
    messages: list[Message]
    steps: int = 0  # number of LLM calls made so far


class ToolCallingAgent(Agent):
    """An LLM that may call tools in a loop until it gives a final answer.

        START -> llm -> tool calls? -- yes --> tools -> llm
                                    -- no  --> END

    The loop also ends after `max_steps` LLM calls, so it can never run forever.
    Subclasses set `name`, `description`, `system_prompt` and pass their tools.
    """

    system_prompt: str

    def __init__(self, llm: LLMClient, tools: list[Tool], max_steps: int) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._max_steps = max_steps
        self._graph = self._build_graph()

    def run(self, task: str) -> str:
        initial_state = AgentState(
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task),
            ]
        )
        # Each step visits at most two nodes (llm and tools).
        final_state = self._graph.invoke(
            initial_state, config={"recursion_limit": 2 * self._max_steps}
        )

        last_message: Message = final_state["messages"][-1]
        if last_message.tool_calls:
            logger.warning("Agent %s stopped after %d steps", self.name, self._max_steps)
            return MAX_STEPS_ANSWER
        return last_message.content

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", self._call_tools)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", self._next_node, ["tools", END])
        graph.add_edge("tools", "llm")
        return graph.compile()

    def _call_llm(self, state: AgentState) -> dict[str, Any]:
        tool_specs = [tool.spec() for tool in self._tools.values()]
        response = self._llm.generate(state.messages, tool_specs)
        return {"messages": [*state.messages, response.to_message()], "steps": state.steps + 1}

    def _next_node(self, state: AgentState) -> str:
        wants_tools = bool(state.messages[-1].tool_calls)
        if wants_tools and state.steps < self._max_steps:
            return "tools"
        return END

    def _call_tools(self, state: AgentState) -> dict[str, Any]:
        results = [self._run_tool(call) for call in state.messages[-1].tool_calls]
        return {"messages": [*state.messages, *results]}

    def _run_tool(self, call: ToolCall) -> Message:
        tool = self._tools.get(call.name)
        if tool is None:
            output = f"Error: unknown tool '{call.name}'"
        else:
            try:
                output = tool.run(call.arguments)
            except ValueError as error:
                # Give the error back to the LLM so it can correct its arguments.
                output = f"Error: {error}"
        logger.info("Agent %s called %s", self.name, call.name)
        return Message(role="tool", content=output, tool_call_id=call.id, tool_name=call.name)
