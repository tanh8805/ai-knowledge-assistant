from pathlib import Path

from app.documents.base import DocumentLoader
from app.documents.markdown import MarkdownLoader
from app.documents.pdf import PDFLoader
from app.documents.text import TextLoader

LOADERS: dict[str, DocumentLoader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".pdf": PDFLoader(),
}


def get_loader(filename: str) -> DocumentLoader:
    """Pick the loader for a file by its extension."""
    extension = Path(filename).suffix.lower()
    if extension not in LOADERS:
        supported = ", ".join(LOADERS)
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")
    return LOADERS[extension]
