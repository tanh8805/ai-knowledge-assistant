import pytest

from app.agents.base import Agent
from app.agents.supervisor import Supervisor
from tests.fakes import FakeLLMClient


class RecordingAgent(Agent):
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"The {name} agent."
        self.tasks: list[str] = []

    def run(self, task: str) -> str:
        self.tasks.append(task)
        return f"{self.name} answer"


def make_agents() -> dict[str, RecordingAgent]:
    return {name: RecordingAgent(name) for name in ("rag", "general", "summarizer")}


def test_routes_task_to_agent_chosen_by_llm() -> None:
    agents = make_agents()
    supervisor = Supervisor(FakeLLMClient(["rag"]), agents, default_agent="general")

    result = supervisor.run("What does my contract say about holidays?")

    assert result.agent == "rag"
    assert result.answer == "rag answer"
    assert agents["rag"].tasks == ["What does my contract say about holidays?"]
    assert agents["general"].tasks == []


def test_routing_prompt_lists_all_agents() -> None:
    llm = FakeLLMClient(["general"])

    Supervisor(llm, make_agents(), default_agent="general").run("Hi")

    system_prompt = llm.calls[0][0][0].content
    assert "- rag: The rag agent." in system_prompt
    assert "- summarizer: The summarizer agent." in system_prompt


def test_tolerates_formatting_around_agent_name() -> None:
    supervisor = Supervisor(FakeLLMClient(["  Summarizer.\n"]), make_agents(), "general")

    assert supervisor.run("Summarize this").agent == "summarizer"


def test_unknown_choice_falls_back_to_default_agent() -> None:
    supervisor = Supervisor(FakeLLMClient(["weather"]), make_agents(), default_agent="general")

    result = supervisor.run("Will it rain?")

    assert result.agent == "general"
    assert result.answer == "general answer"


def test_default_agent_must_exist() -> None:
    with pytest.raises(ValueError):
        Supervisor(FakeLLMClient(), make_agents(), default_agent="missing")
