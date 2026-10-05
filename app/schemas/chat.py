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


ChatMode = Literal["rag", "agent", "multi_agent"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: ChatMode = Field(
        default="rag",
        description=(
            "rag: retrieve then answer. "
            "agent: one agent that chooses its own tools. "
            "multi_agent: a supervisor routes to a specialized agent."
        ),
    )


class ChatResponse(BaseModel):
    answer: str
    mode: ChatMode
    sources: list[Source] = Field(default=[], description="Chunks used (rag mode only).")
    agent: str | None = Field(default=None, description="Agent that produced the answer.")
