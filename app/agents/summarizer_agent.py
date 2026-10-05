from app.agents.base import Agent
from app.llm.base import LLMClient, Message


class SummarizerAgent(Agent):
    """A single LLM call with a summarization prompt. It needs no tools, so no loop."""

    name = "summarizer"
    description = "Summarizes or condenses a piece of text provided in the request."
    system_prompt = (
        "Summarize the text the user provides. Keep the key facts, "
        "use short bullet points, and do not add information that is not in the text."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, task: str) -> str:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        return self._llm.generate(messages).content
