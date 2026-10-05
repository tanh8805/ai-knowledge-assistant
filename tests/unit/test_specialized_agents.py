from app.agents.base import Agent
from app.agents.general_agent import GeneralAgent
from app.agents.rag_agent import RAGAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.llm.base import LLMResponse, ToolCall
from app.rag.retriever import Retriever
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient, FakeLLMClient

DIMENSION = 64
MAX_STEPS = 5


def call(name: str, **arguments: str) -> LLMResponse:
    return LLMResponse(
        content="", model="fake", tool_calls=(ToolCall(id="1", name=name, arguments=arguments),)
    )


def make_retriever() -> Retriever:
    embedding = FakeEmbeddingClient(DIMENSION)
    store = InMemoryVectorStore()
    chunks = ["The warranty lasts two years"]
    store.add_document("warranty.txt", "hash", chunks, embedding.embed(chunks))
    return Retriever(embedding, store, top_k=1)


def test_all_agents_implement_agent() -> None:
    llm = FakeLLMClient()
    agents = [
        RAGAgent(llm, make_retriever(), MAX_STEPS),
        GeneralAgent(llm, MAX_STEPS),
        SummarizerAgent(llm),
    ]

    assert all(isinstance(agent, Agent) for agent in agents)
    assert [agent.name for agent in agents] == ["rag", "general", "summarizer"]


def test_rag_agent_searches_documents() -> None:
    llm = FakeLLMClient([call("search_documents", query="warranty"), "Two years [1]."])

    answer = RAGAgent(llm, make_retriever(), MAX_STEPS).run("How long is the warranty?")

    assert answer == "Two years [1]."
    assert "The warranty lasts two years" in llm.calls[1][0][-1].content


def test_general_agent_uses_calculator() -> None:
    llm = FakeLLMClient([call("calculator", expression="15 * 4"), "15 * 4 = 60"])

    answer = GeneralAgent(llm, MAX_STEPS).run("What is 15 times 4?")

    assert answer == "15 * 4 = 60"
    assert llm.calls[1][0][-1].content == "60"


def test_summarizer_sends_text_with_summary_prompt() -> None:
    llm = FakeLLMClient(["- Short summary"])

    answer = SummarizerAgent(llm).run("Summarize: a very long text")

    messages, tools = llm.calls[0]
    assert answer == "- Short summary"
    assert messages[0].role == "system"
    assert messages[1].content == "Summarize: a very long text"
    assert tools == []
