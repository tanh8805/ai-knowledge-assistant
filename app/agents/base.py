from abc import ABC, abstractmethod


class Agent(ABC):
    """Something that can complete a task given as text.

    `name` and `description` let the supervisor decide which agent should handle a request.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, task: str) -> str:
        """Complete the task and return the final answer."""
