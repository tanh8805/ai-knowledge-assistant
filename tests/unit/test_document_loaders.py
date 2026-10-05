import pytest

from app.documents.base import DocumentLoader
from app.documents.loaders import get_loader
from app.documents.markdown import MarkdownLoader
from app.documents.pdf import PDFLoader
from app.documents.text import TextLoader


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_offset = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return pdf


def test_document_loader_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        DocumentLoader()  # type: ignore[abstract]


def test_text_loader_decodes_utf8() -> None:
    assert TextLoader().load("Xin chào".encode()) == "Xin chào"


def test_text_loader_rejects_binary_data() -> None:
    with pytest.raises(ValueError):
        TextLoader().load(b"\xff\xfe\x00")


def test_markdown_loader_strips_front_matter() -> None:
    data = b"---\ntitle: Notes\n---\n# Heading\nBody text"

    assert MarkdownLoader().load(data) == "# Heading\nBody text"


def test_markdown_loader_keeps_documents_without_front_matter() -> None:
    assert MarkdownLoader().load(b"# Heading\n---\nBody") == "# Heading\n---\nBody"


def test_pdf_loader_extracts_text() -> None:
    assert "Hello PDF" in PDFLoader().load(make_pdf("Hello PDF"))


def test_pdf_loader_rejects_invalid_pdf() -> None:
    with pytest.raises(ValueError):
        PDFLoader().load(b"not a pdf")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("notes.txt", TextLoader), ("README.MD", MarkdownLoader), ("paper.pdf", PDFLoader)],
)
def test_get_loader_picks_loader_by_extension(filename: str, expected: type) -> None:
    assert type(get_loader(filename)) is expected


def test_get_loader_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_loader("slides.pptx")
