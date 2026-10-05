import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.documents.base import DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, data: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() for page in reader.pages]
        except PdfReadError as error:
            raise ValueError("Invalid PDF file") from error
        return "\n\n".join(page for page in pages if page)
