from app.agents.tool_agent import ToolCallingAgent
from app.agents.tools import CalculatorTool
from app.llm.base import LLMClient


class GeneralAgent(ToolCallingAgent):
    name = "general"
    description = "Handles general questions, conversation and arithmetic calculations."
    system_prompt = (
        "You are a helpful assistant. "
        "Use the calculator tool for any arithmetic instead of computing it yourself."
    )

    def __init__(self, llm: LLMClient, max_steps: int) -> None:
        super().__init__(llm, [CalculatorTool()], max_steps)
