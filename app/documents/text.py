from app.documents.base import DocumentLoader


class TextLoader(DocumentLoader):
    def load(self, data: bytes) -> str:
        # UnicodeDecodeError is a ValueError, matching the DocumentLoader contract.
        return data.decode("utf-8")
