from abc import ABC, abstractmethod
from typing import Any

from app.llm.base import ToolSpec


class Tool(ABC):
    """An action an agent can ask to perform.

    `parameters` is a JSON schema; the LLM uses it to produce the `arguments` passed to `run`.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> str:
        """Execute the tool. Raise ValueError for invalid arguments."""

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)
