"""Builds the application's objects from configuration.

Each builder is cached, so the whole app shares one instance. Tests replace them
with `app.dependency_overrides` to run without a database or API keys.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import get_settings
from app.db.database import create_session_factory
from app.embeddings.base import EmbeddingClient
from app.embeddings.factory import create_embedding_client
from app.llm.base import LLMClient
from app.llm.factory import create_llm_client
from app.rag.ingestion import DocumentIngestor
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.vectorstore.base import VectorStore
from app.vectorstore.postgres import PostgresVectorStore


@lru_cache
def get_llm_client() -> LLMClient:
    return create_llm_client(get_settings())


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return create_embedding_client(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    return PostgresVectorStore(create_session_factory(get_settings().database_url))


def get_ingestor(
    embedding: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentIngestor:
    rag = get_settings().rag
    return DocumentIngestor(embedding, vector_store, rag.chunk_size, rag.chunk_overlap)


def get_retriever(
    embedding: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Retriever:
    return Retriever(embedding, vector_store, get_settings().rag.top_k)


def get_rag_pipeline(
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RAGPipeline:
    return RAGPipeline(llm, retriever)
