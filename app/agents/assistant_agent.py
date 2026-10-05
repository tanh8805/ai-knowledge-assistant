from app.agents.tool_agent import ToolCallingAgent
from app.agents.tools import CalculatorTool, RAGSearchTool
from app.llm.base import LLMClient
from app.rag.retriever import Retriever


class AssistantAgent(ToolCallingAgent):
    """The single-agent mode: one agent that decides by itself which tools to use."""

    name = "assistant"
    description = "General assistant that can search the uploaded documents and calculate."
    system_prompt = (
        "You are a helpful assistant. Search the user's documents when the question may be "
        "about them, and cite passages like [1]. Use the calculator for arithmetic. "
        "If you cannot find the answer, say so."
    )

    def __init__(self, llm: LLMClient, retriever: Retriever, max_steps: int) -> None:
        super().__init__(llm, [RAGSearchTool(retriever), CalculatorTool()], max_steps)
