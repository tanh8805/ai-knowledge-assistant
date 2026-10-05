from app.agents.assistant_agent import AssistantAgent
from app.llm.base import LLMResponse, ToolCall
from app.rag.retriever import Retriever
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient, FakeLLMClient

DIMENSION = 64


def test_assistant_can_combine_search_and_calculator() -> None:
    embedding = FakeEmbeddingClient(DIMENSION)
    store = InMemoryVectorStore()
    chunks = ["A ticket costs 12 USD"]
    store.add_document("prices.txt", "hash", chunks, embedding.embed(chunks))
    llm = FakeLLMClient(
        [
            LLMResponse(
                content="",
                model="fake",
                tool_calls=(
                    ToolCall(id="1", name="search_documents", arguments={"query": "ticket cost"}),
                ),
            ),
            LLMResponse(
                content="",
                model="fake",
                tool_calls=(
                    ToolCall(id="2", name="calculator", arguments={"expression": "12 * 3"}),
                ),
            ),
            "Three tickets cost 36 USD [1].",
        ]
    )
    agent = AssistantAgent(llm, Retriever(embedding, store, top_k=1), max_steps=5)

    answer = agent.run("How much do three tickets cost?")

    assert answer == "Three tickets cost 36 USD [1]."
    assert "A ticket costs 12 USD" in llm.calls[1][0][-1].content
    assert llm.calls[2][0][-1].content == "36"
