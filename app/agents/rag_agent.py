from app.agents.tool_agent import ToolCallingAgent
from app.agents.tools import RAGSearchTool
from app.llm.base import LLMClient
from app.rag.retriever import Retriever


class RAGAgent(ToolCallingAgent):
    name = "rag"
    description = "Answers questions about the content of the uploaded documents."
    system_prompt = (
        "You answer questions about the user's documents. "
        "Always search the documents before answering, and you may search more than once "
        "with different queries. Answer only from the passages you found and cite them like [1]. "
        "If the documents do not contain the answer, say that you don't know."
    )

    def __init__(self, llm: LLMClient, retriever: Retriever, max_steps: int) -> None:
        super().__init__(llm, [RAGSearchTool(retriever)], max_steps)
