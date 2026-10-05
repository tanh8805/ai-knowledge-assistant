import pytest

from app.agents.tools import CalculatorTool, RAGSearchTool
from app.rag.retriever import Retriever
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient

DIMENSION = 64


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", "14"), ("(10 - 4) / 4", "1.5"), ("-2 ** 3", "-8"), ("17 % 5", "2")],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert CalculatorTool().run({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "x + 1", "1 / 0", "2 ** 1000", "1 +", "'a' * 3"],
)
def test_calculator_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        CalculatorTool().run({"expression": expression})


def test_calculator_requires_expression() -> None:
    with pytest.raises(ValueError, match="expression"):
        CalculatorTool().run({})


def test_rag_search_returns_numbered_passages() -> None:
    embedding = FakeEmbeddingClient(DIMENSION)
    store = InMemoryVectorStore()
    chunks = ["Refunds are processed within 14 days", "Shipping is free over 50 USD"]
    store.add_document("policy.md", "hash", chunks, embedding.embed(chunks))
    tool = RAGSearchTool(Retriever(embedding, store, top_k=1))

    output = tool.run({"query": "when are refunds processed"})

    assert output == "[1] policy.md (chunk 0):\nRefunds are processed within 14 days"


def test_rag_search_with_no_documents() -> None:
    tool = RAGSearchTool(Retriever(FakeEmbeddingClient(DIMENSION), InMemoryVectorStore(), top_k=3))

    assert tool.run({"query": "anything"}) == "No relevant passages found."
