from abc import ABC, abstractmethod


class DocumentLoader(ABC):
    """Turns the raw bytes of an uploaded file into plain text."""

    @abstractmethod
    def load(self, data: bytes) -> str:
        """Return the document text. Raise ValueError if the file cannot be read."""
