from dataclasses import dataclass

from app.llm.base import LLMClient, Message
from app.rag.retriever import Retriever
from app.vectorstore.base import SearchResult

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Cite the sources you use with their numbers, like [1]. "
    "If the context does not contain the answer, say that you don't know."
)

NO_CONTEXT_ANSWER = "I couldn't find any relevant information in the uploaded documents."


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    """Question -> retrieve relevant chunks -> build prompt -> LLM -> answer with sources."""

    def __init__(self, llm: LLMClient, retriever: Retriever) -> None:
        self._llm = llm
        self._retriever = retriever

    def answer(self, question: str) -> RAGAnswer:
        sources = self._retriever.retrieve(question)
        if not sources:
            return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[])

        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=build_prompt(question, sources)),
        ]
        response = self._llm.generate(messages)
        return RAGAnswer(answer=response.content, sources=sources)


def format_sources(sources: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{number}] {source.filename} (chunk {source.chunk_index}):\n{source.content}"
        for number, source in enumerate(sources, start=1)
    )


def build_prompt(question: str, sources: list[SearchResult]) -> str:
    return f"Context:\n{format_sources(sources)}\n\nQuestion: {question}"
