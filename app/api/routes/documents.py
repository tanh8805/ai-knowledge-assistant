from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status

from app.api.dependencies import get_ingestor
from app.rag.ingestion import DocumentIngestor
from app.schemas.documents import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile,
    response: Response,
    ingestor: Annotated[DocumentIngestor, Depends(get_ingestor)],
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    try:
        result = ingestor.ingest(file.filename, file.file.read())
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    if result.already_exists:
        response.status_code = status.HTTP_200_OK
    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunks_added=result.chunks_added,
        already_exists=result.already_exists,
    )
