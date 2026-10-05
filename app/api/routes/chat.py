from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_pipeline, get_retriever
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
    Source,
)

router = APIRouter(tags=["chat"])


@router.post("/retrieve")
def retrieve(
    request: RetrieveRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RetrieveResponse:
    results = retriever.retrieve(request.query, request.top_k)
    return RetrieveResponse(results=[Source.model_validate(result) for result in results])


@router.post("/chat")
def chat(
    request: ChatRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> ChatResponse:
    result = pipeline.answer(request.message)
    return ChatResponse(
        answer=result.answer,
        sources=[Source.model_validate(source) for source in result.sources],
    )
