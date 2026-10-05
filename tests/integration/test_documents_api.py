from fastapi.testclient import TestClient
from httpx import Response

from app.api.dependencies import get_embedding_client
from app.core.config import MissingAPIKeyError
from app.main import app


def upload(client: TestClient, filename: str, content: bytes) -> Response:
    return client.post("/documents", files={"file": (filename, content)})


def test_upload_text_document(client: TestClient) -> None:
    response = upload(client, "notes.txt", b"FastAPI is a Python web framework.")

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_added"] == 1
    assert body["already_exists"] is False


def test_uploading_same_file_twice_returns_existing_document(client: TestClient) -> None:
    first = upload(client, "notes.md", b"# Notes\nSame content")
    second = upload(client, "notes.md", b"# Notes\nSame content")

    assert second.status_code == 200
    assert second.json()["already_exists"] is True
    assert second.json()["document_id"] == first.json()["document_id"]


def test_unsupported_file_type_returns_400(client: TestClient) -> None:
    response = upload(client, "slides.pptx", b"binary")

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_missing_api_key_returns_503(client: TestClient) -> None:
    def missing_key() -> None:
        raise MissingAPIKeyError("GEMINI_API_KEY is not set")

    app.dependency_overrides[get_embedding_client] = missing_key

    response = upload(client, "notes.txt", b"text")

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
