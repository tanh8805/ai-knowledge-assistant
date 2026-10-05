# Code Reading Order

A path through the code from the simplest files to the most connected ones. Each step only relies on
files you have already read. The whole `app/` package is about 1,500 lines.

Tip: keep [ARCHITECTURE.md](ARCHITECTURE.md) open next to this guide, and run `pytest` after each part —
the test file for a component is the fastest way to see how it is used.

---

## Part 1 – Configuration and data models

### 1. `config.yaml` and `.env.example`
- **Why read it:** every tunable value in the project comes from one of these two files.
- **What to understand:** secrets and the database URL go in `.env`; models, chunk size, `top_k`,
  `max_steps` and the embedding dimension go in `config.yaml`. `LLM_PROVIDER` / `EMBEDDING_PROVIDER`
  in `.env` override the providers in `config.yaml`.
- **Depends on:** nothing.
- **Used by:** `app/core/config.py`, `docker-compose.yml`.

### 2. `app/core/config.py`
- **Why read it:** shows how the two files become one typed `Settings` object.
- **What to understand:** `EnvSettings` (pydantic-settings) reads the environment, `load_settings()` merges
  it into the YAML data, and `get_settings()` caches the result. API keys are `SecretStr`, so they are
  hidden when printed. `api_key_for()` raises `MissingAPIKeyError`, which the API turns into a 503.
- **Depends on:** `config.yaml`, `.env`.
- **Used by:** `app/main.py`, both factories, `app/api/dependencies.py`, `alembic/env.py`.

### 3. `app/core/logging.py`
- **Why read it:** it is a few lines; read it to know there is no logging framework.
- **What to understand:** standard `logging.basicConfig` with the level from `config.yaml`.
- **Depends on:** nothing.
- **Used by:** `app/main.py`.

### 4. `app/db/models.py`
- **Why read it:** the only persistent data in the project.
- **What to understand:** two tables, `documents` and `chunks`; `file_hash` is unique for duplicate
  detection; `embedding` is a pgvector `Vector(768)` column whose size must match `config.yaml`.
- **Depends on:** SQLAlchemy, pgvector.
- **Used by:** `app/vectorstore/postgres.py`, `alembic/env.py`.

---

## Part 2 – Abstractions

These files are short and contain no SDK code. Together they describe what the system can do.

### 5. `app/llm/base.py`
- **Why read it:** the vocabulary used by RAG and every agent.
- **What to understand:** `Message`, `ToolCall`, `ToolSpec`, `LLMResponse` are frozen dataclasses owned by
  this project, not by OpenAI or Google. `LLMClient.generate(messages, tools)` is the only method business
  logic calls. `LLMResponse.to_message()` turns a reply back into conversation history.
- **Depends on:** nothing.
- **Used by:** LLM providers, `rag/pipeline.py`, all agents, `tests/fakes.py`.

### 6. `app/embeddings/base.py`
- **Why read it:** the second provider boundary.
- **What to understand:** `embed()` for many texts, `embed_query()` for one; `dimension` is fixed per client.
- **Depends on:** nothing.
- **Used by:** embedding providers, `rag/retriever.py`, `rag/ingestion.py`.

### 7. `app/documents/base.py`
- **Why read it:** shows the contract every file format must follow.
- **What to understand:** loaders take `bytes` (uploads, not paths) and raise `ValueError` on bad input.
- **Depends on:** nothing.
- **Used by:** the three loaders.

### 8. `app/vectorstore/base.py`
- **Why read it:** the database boundary.
- **What to understand:** three operations (find by hash, add, search) and the `SearchResult` dataclass
  that the rest of the code uses instead of database rows.
- **Depends on:** nothing.
- **Used by:** both vector stores, `rag/retriever.py`, `rag/ingestion.py`.

### 9. `app/agents/base.py`
- **Why read it:** the interface the supervisor routes between.
- **What to understand:** an agent has a `name`, a `description` (read by the routing LLM) and `run(task)`.
- **Depends on:** nothing.
- **Used by:** every agent, `agents/supervisor.py`.

### 10. `app/agents/tools.py` — the `Tool` class only (top of the file)
- **Why read it:** the interface the agent loop uses to run tools.
- **What to understand:** `parameters` is a JSON schema; `spec()` converts a tool into the `ToolSpec` the
  LLM sees. Leave the concrete tools below it for Part 6.
- **Depends on:** `llm/base.py`.
- **Used by:** `agents/tool_agent.py`.

---

## Part 3 – Concrete implementations

### 11. `tests/fakes.py`
- **Why read it:** the simplest possible implementations of `LLMClient` and `EmbeddingClient`. Reading them
  first makes the real providers easier to follow.
- **What to understand:** `FakeLLMClient` returns scripted responses and records every call;
  `FakeEmbeddingClient` hashes words into buckets so texts with shared words are similar.
- **Depends on:** `llm/base.py`, `embeddings/base.py`.
- **Used by:** almost every test.

### 12. `app/llm/openai.py`
- **Why read it:** a real provider translating to and from the internal models.
- **What to understand:** `to_openai_message` / `to_openai_tool` build the request; `parse_completion` turns
  the SDK object into `LLMResponse`. Tool arguments are JSON strings in OpenAI's format.
- **Depends on:** `llm/base.py`, `openai` SDK.
- **Used by:** `llm/factory.py`.

### 13. `app/llm/gemini.py`
- **Why read it:** the same job for a provider with a different shape — compare it with step 12.
- **What to understand:** system messages become `system_instruction`; assistant role is `model`; tool
  results are `function_response` parts; the `thought_signature` is kept in `ToolCall.signature` and sent
  back; automatic function calling is disabled because our agent runs tools itself.
- **Depends on:** `llm/base.py`, `google-genai` SDK.
- **Used by:** `llm/factory.py`.

### 14. `app/llm/factory.py`
- **Why read it:** the single place where a provider name becomes a class.
- **What to understand:** reads `settings.llm.provider`, fetches the key, returns an `LLMClient`.
- **Depends on:** `core/config.py`, both LLM providers.
- **Used by:** `api/dependencies.py`.

### 15. `app/embeddings/openai.py`, `app/embeddings/gemini.py`, `app/embeddings/factory.py`
- **Why read it:** same pattern as the LLM providers, on a smaller scale.
- **What to understand:** both request the configured `dimension` from the API and split inputs into
  batches within the provider's limit.
- **Depends on:** `embeddings/base.py`, the SDKs, `core/config.py`.
- **Used by:** `api/dependencies.py`.

### 16. `app/documents/text.py`, `markdown.py`, `pdf.py`
- **Why read it:** three small loaders showing why the abstraction exists.
- **What to understand:** `MarkdownLoader` reuses `TextLoader` and strips front matter; `PDFLoader` wraps
  pypdf errors in `ValueError` to keep the contract.
- **Depends on:** `documents/base.py`, pypdf.
- **Used by:** `documents/loaders.py`.

### 17. `app/documents/loaders.py`
- **Why read it:** how a filename selects a loader.
- **What to understand:** a plain dict from extension to loader — no registry class needed.
- **Depends on:** the three loaders.
- **Used by:** `rag/ingestion.py`.

---

## Part 4 – Infrastructure

### 18. `app/db/database.py`
- **Why read it:** where database connections come from.
- **What to understand:** one function that returns a `sessionmaker`; no global engine.
- **Depends on:** SQLAlchemy.
- **Used by:** `api/dependencies.py`, the PostgreSQL integration test.

### 19. `alembic/env.py` and `alembic/versions/0001_create_documents_and_chunks.py`
- **Why read it:** how the schema in step 4 is created in a real database.
- **What to understand:** the migration enables the `vector` extension, then creates both tables. The
  Docker container runs `alembic upgrade head` before starting the API.
- **Depends on:** `core/config.py` (`DATABASE_URL`), `db/models.py`.
- **Used by:** `Dockerfile` start command, developers running `alembic`.

### 20. `app/vectorstore/memory.py`
- **Why read it:** vector search without a database, so the idea is visible in plain Python.
- **What to understand:** cosine similarity = dot product / (norm × norm); results sorted by score.
- **Depends on:** `vectorstore/base.py`.
- **Used by:** unit and API tests.

### 21. `app/vectorstore/postgres.py`
- **Why read it:** the production vector store.
- **What to understand:** chunks are saved with their document in one transaction; search orders by
  pgvector's `cosine_distance` and converts distance to similarity.
- **Depends on:** `vectorstore/base.py`, `db/models.py`.
- **Used by:** `api/dependencies.py`.

---

## Part 5 – RAG

### 22. `app/rag/chunker.py`
- **Why read it:** the first step that changes data shape (text → chunks).
- **What to understand:** fixed-size character windows; each window starts `chunk_size - chunk_overlap`
  after the previous one.
- **Depends on:** nothing.
- **Used by:** `rag/ingestion.py`.

### 23. `app/rag/ingestion.py`
- **Why read it:** the whole write path in one method.
- **What to understand:** hash → skip duplicates → load → chunk → embed → store.
- **Depends on:** `documents/loaders.py`, `rag/chunker.py`, `EmbeddingClient`, `VectorStore`.
- **Used by:** `POST /documents`.

### 24. `app/rag/retriever.py`
- **Why read it:** the read path's first half.
- **What to understand:** embed the query, search the store. It is a concrete class on purpose: there is
  one strategy, and it is testable through its two abstract dependencies.
- **Depends on:** `EmbeddingClient`, `VectorStore`.
- **Used by:** `rag/pipeline.py`, `RAGSearchTool`, `POST /retrieve`.

### 25. `app/rag/pipeline.py`
- **Why read it:** the core RAG idea in about 50 lines.
- **What to understand:** retrieved chunks are numbered in the prompt so the LLM can cite `[1]`; with no
  chunks the LLM is not called. `format_sources` is reused by the RAG search tool.
- **Depends on:** `LLMClient`, `rag/retriever.py`.
- **Used by:** `/chat` in `rag` mode, `agents/tools.py`.

---

## Part 6 – Agent

### 26. `app/agents/tools.py` — `RAGSearchTool` and `CalculatorTool`
- **Why read it:** what the agent can actually do.
- **What to understand:** RAG search is the retriever exposed as a tool. The calculator walks the Python
  AST and only allows numbers and arithmetic operators, never `eval`.
- **Depends on:** `Tool`, `rag/retriever.py`, `rag/pipeline.py`.
- **Used by:** the tool-calling agents.

### 27. `app/agents/tool_agent.py`
- **Why read it:** the most important file of the agent layer.
- **What to understand:** the LangGraph graph `START → llm → tools → llm … → END`, the `AgentState`
  dataclass, how `_next_node` decides to stop, how `max_steps` prevents infinite loops, and how tool
  errors are returned to the LLM.
- **Depends on:** `Agent`, `Tool`, `LLMClient`, LangGraph.
- **Used by:** `AssistantAgent`, `RAGAgent`, `GeneralAgent`.

### 28. `app/agents/assistant_agent.py`
- **Why read it:** the single-agent mode (`/chat` with `mode=agent`).
- **What to understand:** a subclass only sets a name, description, prompt and tools; the loop is inherited.
- **Depends on:** `tool_agent.py`, `tools.py`.
- **Used by:** `api/dependencies.py`.

### 29. `app/agents/rag_agent.py`, `general_agent.py`, `summarizer_agent.py`
- **Why read it:** the specialized agents the supervisor chooses between.
- **What to understand:** two reuse the tool loop with one tool each; `SummarizerAgent` implements `Agent`
  directly with one LLM call — proof that the `Agent` interface is not tied to LangGraph.
- **Depends on:** `tool_agent.py`, `tools.py`, `Agent`, `LLMClient`.
- **Used by:** `api/dependencies.py` (`get_supervisor`).

---

## Part 7 – Multi-Agent

### 30. `app/agents/supervisor.py`
- **Why read it:** the top layer of the project.
- **What to understand:** the routing prompt lists agent names and descriptions; the answer is normalized
  and falls back to a default agent; a second LangGraph graph has one node per agent and a conditional
  edge from `route`. The supervisor only knows `dict[str, Agent]`.
- **Depends on:** `Agent`, `LLMClient`, LangGraph.
- **Used by:** `/chat` in `multi_agent` mode.

---

## Part 8 – API

### 31. `app/schemas/documents.py` and `app/schemas/chat.py`
- **Why read it:** the HTTP contract.
- **What to understand:** request validation (non-empty message, `top_k` bounds, allowed `mode` values) and
  `Source.model_validate(search_result)` mapping dataclasses to responses.
- **Depends on:** Pydantic.
- **Used by:** the routes.

### 32. `app/api/dependencies.py`
- **Why read it:** where every object in the app is created and connected.
- **What to understand:** cached builders for expensive clients, `Depends()` chains for the rest, and why
  that makes `app.dependency_overrides` possible in tests.
- **Depends on:** almost everything above.
- **Used by:** the routes.

### 33. `app/api/routes/health.py`, `documents.py`, `chat.py`
- **Why read it:** thin HTTP handlers.
- **What to understand:** each route validates, calls one object, and maps errors (`ValueError` → 400).
  `/chat` dispatches on `mode` to the RAG pipeline, the assistant agent or the supervisor.
- **Depends on:** `api/dependencies.py`, schemas.
- **Used by:** `app/main.py`.

### 34. `app/main.py`
- **Why read it:** the application entry point; it ties the routers together.
- **What to understand:** settings and logging are set up once; `MissingAPIKeyError` becomes a 503 with the
  missing variable name and no stack trace.
- **Depends on:** routes, `core/config.py`, `core/logging.py`.
- **Used by:** Uvicorn (`uvicorn app.main:app`), API tests.

---

## Part 9 – Tests

### 35. `tests/conftest.py` and `pytest.ini`
- **Why read it:** how tests run without a real environment.
- **What to understand:** a placeholder `DATABASE_URL` is set; `pythonpath = .` makes `app` importable.
- **Depends on:** nothing.
- **Used by:** every test.

### 36. `tests/unit/test_rag_pipeline.py`, `test_tool_agent.py`, `test_supervisor.py`
- **Why read it:** the three tests that best explain the three layers.
- **What to understand:** how a scripted `FakeLLMClient` drives tool selection, tool execution, the final
  answer and the `max_steps` limit without any network call.
- **Depends on:** `tests/fakes.py`, the components under test.
- **Used by:** `pytest`.

### 37. `tests/integration/conftest.py` and `test_chat_api.py`
- **Why read it:** API testing with dependency overrides.
- **What to understand:** the `client` fixture replaces the LLM, embeddings and vector store with fakes;
  individual tests override the LLM again to script a specific conversation.
- **Depends on:** `app/main.py`, `api/dependencies.py`, `tests/fakes.py`.
- **Used by:** `pytest`.

### 38. `tests/integration/test_postgres_vectorstore.py`
- **Why read it:** the one test that uses a real database.
- **What to understand:** skipped unless `TEST_DATABASE_URL` is set; checks that pgvector search returns
  the expected chunk.
- **Depends on:** `vectorstore/postgres.py`, a running PostgreSQL with pgvector.
- **Used by:** `pytest` with `TEST_DATABASE_URL`.

---

## Part 10 – Docker

### 39. `Dockerfile`
- **Why read it:** how the API is packaged.
- **What to understand:** single stage on `python:3.12-slim`, runs as a non-root user, migrates the
  database then starts Uvicorn.
- **Depends on:** `requirements.txt`, `app/`, `alembic/`, `config.yaml`.
- **Used by:** `docker-compose.yml`.

### 40. `docker-compose.yml`
- **Why read it:** how the API and database run together.
- **What to understand:** `postgres` uses the `pgvector/pgvector` image with a health check; `api` waits for
  it, reads `.env` if it exists, and gets a `DATABASE_URL` pointing at the `postgres` service.
- **Depends on:** `Dockerfile`, `.env`.
- **Used by:** `docker compose up`.
