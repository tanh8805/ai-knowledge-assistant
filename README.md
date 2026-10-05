# AI Knowledge Assistant

## Overview

A question-answering service over your own documents, built in three layers:

1. **RAG** – upload `.txt`, `.md` or `.pdf` files, then ask questions answered from their content, with sources.
2. **Agent** – one LLM agent that decides by itself when to search the documents or use a calculator.
3. **Multi-Agent** – a supervisor that routes each request to a specialized agent (documents, general, summarizer).

It is a learning and portfolio project: small enough to read in an afternoon, but structured like a real
service (configuration, migrations, provider abstractions, tests, Docker).

## Features

- Document ingestion for `.txt`, `.md` (front matter stripped) and `.pdf`, with duplicate detection by content hash
- Character-based chunking with overlap
- Embeddings and chat from **OpenAI** or **Gemini**, selected by configuration
- Vector search in **PostgreSQL + pgvector** (cosine distance)
- RAG answers with numbered citations and the source chunks
- Tool-calling agent loop built with **LangGraph**, bounded by `agent.max_steps`
- Supervisor that routes to a RAG agent, a general agent (with calculator) or a summarizer agent
- Unit and API tests that run offline using fake providers and an in-memory vector store

## Architecture

```mermaid
flowchart TD
    Client([HTTP client]) --> API[FastAPI routes]

    API -->|POST /documents| Ingestor[DocumentIngestor]
    API -->|POST /retrieve| Retriever
    API -->|POST /chat mode=rag| RAG[RAGPipeline]
    API -->|POST /chat mode=agent| Assistant[AssistantAgent]
    API -->|POST /chat mode=multi_agent| Supervisor

    Ingestor --> Loader[DocumentLoader<br/>txt / md / pdf]
    Ingestor --> Chunker[chunk_text]
    Ingestor --> Embedding

    RAG --> Retriever
    RAG --> LLM
    Retriever --> Embedding[EmbeddingClient<br/>OpenAI / Gemini]
    Retriever --> Store[VectorStore<br/>Postgres / InMemory]
    Ingestor --> Store

    Supervisor --> LLM
    Supervisor --> RAGAgent & GeneralAgent & SummarizerAgent
    Assistant --> Tools[RAGSearchTool<br/>CalculatorTool]
    RAGAgent --> Tools
    GeneralAgent --> Tools
    Assistant & RAGAgent & GeneralAgent & SummarizerAgent --> LLM[LLMClient<br/>OpenAI / Gemini]
    Tools --> Retriever

    Store --> DB[(PostgreSQL + pgvector)]
```

Business logic depends only on the abstract classes (`LLMClient`, `EmbeddingClient`, `VectorStore`,
`DocumentLoader`, `Tool`, `Agent`). Concrete providers are chosen in one place, `app/api/dependencies.py`.
See [ARCHITECTURE.md](ARCHITECTURE.md) for details and [CODE_READING_ORDER.md](CODE_READING_ORDER.md)
for a guided tour of the code.

## Tech Stack

| Area | Choice |
| --- | --- |
| Language | Python 3.12 |
| API | FastAPI, Pydantic, Pydantic Settings |
| Database | PostgreSQL 16, pgvector, SQLAlchemy 2, Alembic, psycopg 3 |
| LLM / embeddings | `openai` SDK, `google-genai` SDK |
| Agents | LangGraph (basic `StateGraph` only) |
| Documents | pypdf |
| Config | `.env` (secrets) + `config.yaml` (PyYAML) |
| Tests | pytest, FastAPI `TestClient` (httpx) |
| Runtime | Docker, Docker Compose |

## Project Structure

```
app/
├── main.py              FastAPI app, routers, error handler
├── core/                config.py (settings), logging.py
├── api/                 dependencies.py (object wiring) and routes/ (health, documents, chat)
├── schemas/             request / response models
├── llm/                 LLMClient + OpenAI and Gemini clients + factory
├── embeddings/          EmbeddingClient + OpenAI and Gemini clients + factory
├── documents/           DocumentLoader + txt / md / pdf loaders, loader selection by extension
├── vectorstore/         VectorStore + PostgreSQL and in-memory implementations
├── rag/                 chunker, ingestion, retriever, RAG pipeline
├── agents/              Tool and Agent abstractions, LangGraph agent loop, agents, supervisor
└── db/                  SQLAlchemy models and session factory
alembic/                 database migrations
tests/
├── fakes.py             FakeLLMClient, FakeEmbeddingClient (tests only)
├── unit/                one file per component
└── integration/         API tests, optional real-PostgreSQL test
data/sample/             example documents to upload
```

## Setup

Requirements: Python 3.12+, Docker (for PostgreSQL), and an OpenAI or Gemini API key.

```bash
git clone https://github.com/tanh8805/ai-knowledge-assistant.git
cd ai-knowledge-assistant

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your API key(s)
```

To run the API locally (outside Docker), start only the database and point `DATABASE_URL` at it:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_assistant
alembic upgrade head
uvicorn app.main:app --reload
```

## Environment Variables

Secrets and environment-specific values go in `.env` (never committed). Everything else is in `config.yaml`.

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | yes | PostgreSQL URL, e.g. `postgresql://postgres:postgres@postgres:5432/ai_assistant` |
| `GEMINI_API_KEY` | if a provider is `gemini` | Google AI Studio key |
| `OPENAI_API_KEY` | if a provider is `openai` | OpenAI key |
| `LLM_PROVIDER` | no | `openai` or `gemini`; overrides `llm.provider` in `config.yaml` |
| `EMBEDDING_PROVIDER` | no | `openai` or `gemini`; overrides `embedding.provider` in `config.yaml` |

`config.yaml` holds model names per provider, chunk size and overlap, `top_k`, `agent.max_steps` and the
embedding dimension (768). The dimension must match the database column; changing it needs a new migration.
If you switch embedding provider, re-ingest your documents: vectors from different models are not comparable.

## Running with Docker

```bash
cp .env.example .env        # add GEMINI_API_KEY or OPENAI_API_KEY
docker compose up -d --build
curl http://localhost:8000/health
```

The `api` container runs `alembic upgrade head` and then starts Uvicorn on port 8000. Interactive API docs
are at http://localhost:8000/docs. Without an API key the server still starts; endpoints that need a model
return `503` with the name of the missing key.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | App name and docs link |
| `GET` | `/health` | Liveness check |
| `POST` | `/documents` | Upload a `.txt`, `.md` or `.pdf` file (multipart field `file`) |
| `POST` | `/retrieve` | Return the most similar chunks for `{"query": "...", "top_k": 3}` |
| `POST` | `/chat` | Ask a question: `{"message": "...", "mode": "rag" \| "agent" \| "multi_agent"}` |

```bash
curl -F file=@data/sample/remote_work_policy.md http://localhost:8000/documents

curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "How many days per week can I work remotely?", "mode": "rag"}'

curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "What is the home office budget for 4 employees?", "mode": "agent"}'

curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "Summarize: RAG retrieves chunks and puts them in the prompt.", "mode": "multi_agent"}'
```

`/chat` returns `answer`, `mode`, `sources` (RAG mode) and `agent` (agent modes).

## RAG Pipeline

Ingestion (`POST /documents`):

```
file bytes → DocumentLoader (by extension) → text → chunk_text → EmbeddingClient.embed → VectorStore.add_document
```

The SHA-256 hash of the file is stored, so uploading the same content twice does not create duplicate chunks.

Question answering (`/chat` with `mode=rag`):

```
question → EmbeddingClient.embed_query → VectorStore.search (top_k) → prompt with numbered chunks → LLMClient → answer + sources
```

If no chunks are found the pipeline answers that it found nothing, without calling the LLM.

## Agent Architecture

`ToolCallingAgent` (`app/agents/tool_agent.py`) is a LangGraph graph with two nodes:

```mermaid
flowchart LR
    START --> llm
    llm -->|tool calls and steps < max_steps| tools
    tools --> llm
    llm -->|no tool calls| END
```

The state is a dataclass with the message list and the number of LLM calls. The LLM sees each tool's name,
description and JSON schema; when it asks for a tool, the `tools` node runs it and appends the result as a
`tool` message. Tool errors are sent back to the LLM as text so it can correct itself. After `max_steps`
LLM calls the loop stops with a fixed message.

Tools: `RAGSearchTool` (searches the documents) and `CalculatorTool` (arithmetic by walking the Python AST,
never `eval`). In `agent` mode the `AssistantAgent` has both.

## Multi-Agent Architecture

```mermaid
flowchart LR
    START --> route
    route -->|"rag"| RAGAgent
    route -->|"general"| GeneralAgent
    route -->|"summarizer"| SummarizerAgent
    RAGAgent --> END
    GeneralAgent --> END
    SummarizerAgent --> END
```

The `Supervisor` asks the LLM which agent fits the request, given each agent's name and description, then
runs that one agent. An unknown answer falls back to the general agent. The supervisor depends on
`dict[str, Agent]`, so it does not know concrete agent classes.

| Agent | Tools | Purpose |
| --- | --- | --- |
| `RAGAgent` | `search_documents` | Questions about uploaded documents (may search several times) |
| `GeneralAgent` | `calculator` | General questions and arithmetic |
| `SummarizerAgent` | none (single LLM call) | Summarizing text given in the request |

This is single-hop routing: one agent answers each request. There is no agent-to-agent conversation.

## Testing

```bash
pytest
```

Tests never call a real LLM or embedding API. They use `FakeLLMClient` (scripted responses),
`FakeEmbeddingClient` (deterministic bag-of-words vectors) and `InMemoryVectorStore`. Provider tests build
SDK response objects locally to check the conversion code.

One integration test runs against a real PostgreSQL + pgvector database and is skipped unless
`TEST_DATABASE_URL` is set:

```bash
docker compose up -d postgres
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_assistant pytest
```

## Future Improvements

- Token-aware or sentence-aware chunking instead of fixed character windows
- An HNSW index on `chunks.embedding` once the data is large enough to need it
- Streaming responses from `/chat`
- Return sources from agent modes, not only from RAG mode
- Conversation history across requests
- Endpoints to list and delete documents
- Retries with backoff for provider rate limits
- An evaluation set to measure answer quality when changing prompts or chunking
