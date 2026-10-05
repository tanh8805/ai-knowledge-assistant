import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_client
from app.llm.base import LLMResponse, ToolCall
from app.main import app
from tests.fakes import FakeLLMClient

DOCUMENT = b"The library opens at 9 AM. Members can borrow five books at a time."


@pytest.fixture
def client_with_document(client: TestClient) -> TestClient:
    client.post("/documents", files={"file": ("library.txt", DOCUMENT)})
    return client


def test_retrieve_returns_matching_chunks(client_with_document: TestClient) -> None:
    response = client_with_document.post("/retrieve", json={"query": "library opens", "top_k": 1})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["filename"] == "library.txt"


def test_rag_chat_returns_answer_with_sources(client_with_document: TestClient) -> None:
    llm_answer = "The library opens at 9 AM [1]."
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([llm_answer])

    response = client_with_document.post("/chat", json={"message": "When does the library open?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == llm_answer
    assert body["sources"][0]["filename"] == "library.txt"


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_agent_mode_runs_tool_calling_agent(client: TestClient) -> None:
    calculator_call = LLMResponse(
        content="",
        model="fake",
        tool_calls=(ToolCall(id="1", name="calculator", arguments={"expression": "6 * 7"}),),
    )
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([calculator_call, "42"])

    response = client.post("/chat", json={"message": "What is 6 times 7?", "mode": "agent"})

    assert response.status_code == 200
    assert response.json()["answer"] == "42"
    assert response.json()["agent"] == "assistant"


def test_multi_agent_mode_reports_chosen_agent(client: TestClient) -> None:
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(["summarizer", "- summary"])

    response = client.post(
        "/chat", json={"message": "Summarize: long text here", "mode": "multi_agent"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "- summary",
        "mode": "multi_agent",
        "sources": [],
        "agent": "summarizer",
    }


def test_chat_rejects_unknown_mode(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hi", "mode": "swarm"})

    assert response.status_code == 422
