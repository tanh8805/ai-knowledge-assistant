import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.base import Agent
from app.llm.base import LLMClient, Message

logger = logging.getLogger(__name__)

ROUTING_PROMPT = (
    "You are a supervisor that sends each request to exactly one specialized agent.\n"
    "Available agents:\n{agents}\n\n"
    "Reply with only the name of the best agent, nothing else."
)


@dataclass(frozen=True)
class SupervisorResult:
    agent: str
    answer: str


@dataclass
class SupervisorState:
    task: str
    agent: str = ""
    answer: str = ""


class Supervisor:
    """Multi-agent entry point: an LLM picks the agent, the agent does the work.

                       +-> rag ---------+
        START -> route +-> general -----+-> END
                       +-> summarizer --+

    The supervisor only knows agents through the `Agent` interface, so adding an
    agent means adding an entry to the `agents` dict.
    """

    def __init__(self, llm: LLMClient, agents: dict[str, Agent], default_agent: str) -> None:
        if default_agent not in agents:
            raise ValueError(f"default agent '{default_agent}' is not in agents")
        self._llm = llm
        self._agents = agents
        self._default_agent = default_agent
        self._graph = self._build_graph()

    def run(self, task: str) -> SupervisorResult:
        final_state = self._graph.invoke(SupervisorState(task=task))
        return SupervisorResult(agent=final_state["agent"], answer=final_state["answer"])

    def route(self, task: str) -> str:
        agent_list = "\n".join(
            f"- {name}: {agent.description}" for name, agent in self._agents.items()
        )
        messages = [
            Message(role="system", content=ROUTING_PROMPT.format(agents=agent_list)),
            Message(role="user", content=task),
        ]
        choice = self._llm.generate(messages).content.strip().strip("`'\".").lower()
        if choice not in self._agents:
            logger.warning("Supervisor chose unknown agent %r, using %s", choice, self._default_agent)
            return self._default_agent
        return choice

    def _build_graph(self) -> Any:
        graph = StateGraph(SupervisorState)
        graph.add_node("route", lambda state: {"agent": self.route(state.task)})
        for name, agent in self._agents.items():
            graph.add_node(name, _agent_node(agent))
            graph.add_edge(name, END)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", lambda state: state.agent, list(self._agents))
        return graph.compile()


def _agent_node(agent: Agent) -> Callable[[SupervisorState], dict[str, str]]:
    def run_agent(state: SupervisorState) -> dict[str, str]:
        logger.info("Supervisor delegated task to %s", agent.name)
        return {"answer": agent.run(state.task)}

    return run_agent
