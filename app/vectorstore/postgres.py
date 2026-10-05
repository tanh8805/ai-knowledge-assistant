from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, Document
from app.vectorstore.base import SearchResult, VectorStore


class PostgresVectorStore(VectorStore):
    """Stores chunks in PostgreSQL and searches them with pgvector's cosine distance."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_document(self, file_hash: str) -> int | None:
        with self._session_factory() as session:
            return session.scalar(select(Document.id).where(Document.file_hash == file_hash))

    def add_document(
        self,
        filename: str,
        file_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        document = Document(filename=filename, file_hash=file_hash)
        document.chunks = [
            Chunk(chunk_index=index, content=content, embedding=embedding)
            for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        with self._session_factory.begin() as session:
            session.add(document)
        return document.id

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        query = (
            select(Chunk, Document.filename, distance)
            .join(Document)
            .order_by(distance)
            .limit(top_k)
        )
        with self._session_factory() as session:
            rows = session.execute(query).all()

        return [
            SearchResult(
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=1.0 - distance,
            )
            for chunk, filename, distance in rows
        ]
