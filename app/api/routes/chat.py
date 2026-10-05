from typing import Annotated

from fastapi import APIRouter, Depends

from app.agents.assistant_agent import AssistantAgent
from app.agents.supervisor import Supervisor
from app.api.dependencies import (
    get_assistant_agent,
    get_rag_pipeline,
    get_retriever,
    get_supervisor,
)
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
    assistant: Annotated[AssistantAgent, Depends(get_assistant_agent)],
    supervisor: Annotated[Supervisor, Depends(get_supervisor)],
) -> ChatResponse:
    if request.mode == "agent":
        answer = assistant.run(request.message)
        return ChatResponse(answer=answer, mode=request.mode, agent=assistant.name)

    if request.mode == "multi_agent":
        result = supervisor.run(request.message)
        return ChatResponse(answer=result.answer, mode=request.mode, agent=result.agent)

    rag_answer = pipeline.answer(request.message)
    return ChatResponse(
        answer=rag_answer.answer,
        mode=request.mode,
        sources=[Source.model_validate(source) for source in rag_answer.sources],
    )
