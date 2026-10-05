from app.rag.pipeline import NO_CONTEXT_ANSWER, RAGPipeline, build_prompt
from app.rag.retriever import Retriever
from app.vectorstore.base import SearchResult
from app.vectorstore.memory import InMemoryVectorStore
from tests.fakes import FakeEmbeddingClient, FakeLLMClient

DIMENSION = 64
CHUNKS = [
    "The office opens at 8 AM on weekdays",
    "Lunch is served in the cafeteria at noon",
]


def make_pipeline(llm: FakeLLMClient, chunks: list[str] = CHUNKS) -> RAGPipeline:
    embedding = FakeEmbeddingClient(DIMENSION)
    store = InMemoryVectorStore()
    if chunks:
        store.add_document("handbook.md", "hash", chunks, embedding.embed(chunks))
    return RAGPipeline(llm, Retriever(embedding, store, top_k=1))


def test_answer_comes_from_llm_with_sources() -> None:
    llm = FakeLLMClient(["The office opens at 8 AM [1]."])

    result = make_pipeline(llm).answer("When does the office open?")

    assert result.answer == "The office opens at 8 AM [1]."
    assert [source.content for source in result.sources] == [CHUNKS[0]]
    assert result.sources[0].filename == "handbook.md"


def test_retrieved_context_is_sent_to_llm() -> None:
    llm = FakeLLMClient()

    make_pipeline(llm).answer("When does the office open?")

    messages, _ = llm.calls[0]
    assert messages[0].role == "system"
    assert CHUNKS[0] in messages[1].content
    assert "When does the office open?" in messages[1].content


def test_no_documents_returns_fallback_without_calling_llm() -> None:
    llm = FakeLLMClient()

    result = make_pipeline(llm, chunks=[]).answer("Anything?")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.calls == []


def test_prompt_numbers_sources() -> None:
    sources = [
        SearchResult(document_id=1, filename="a.txt", chunk_index=0, content="first", score=0.9),
        SearchResult(document_id=2, filename="b.txt", chunk_index=3, content="second", score=0.8),
    ]

    prompt = build_prompt("question?", sources)

    assert "[1] a.txt (chunk 0):\nfirst" in prompt
    assert "[2] b.txt (chunk 3):\nsecond" in prompt
