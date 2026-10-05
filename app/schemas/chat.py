from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """A chunk used to answer a question."""

    model_config = ConfigDict(from_attributes=True)

    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrieveResponse(BaseModel):
    results: list[Source]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: Literal["rag"] = "rag"


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
