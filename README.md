# Medical RAG System

Moving computers: follow [fresh Windows setup](docs/NEW_MACHINE_SETUP.md),
[the private GitHub handoff review](docs/GITHUB_HANDOFF_REVIEW.md), and
[the unresolved Ollama issue](docs/OLLAMA_ZERO_VECTOR_ISSUE.md). The reviewed
[recovery snapshot](reproducibility/README.md) preserves the exact stopped state.
Publication is pending explicit approval; no GitHub URL is confirmed yet.

## Overview

This repository is the structural foundation for a multilingual, evidence-grounded
medical retrieval-augmented generation system. The approved source documents are
preserved under `data/knowledge_base/source/` and are organized by KB ID.

## Project status

Stage 1 provides a working FastAPI foundation with validated MTI API schemas and
automated tests. `GET /health` reports liveness and configured model identities.
`POST /chat` validates input and returns HTTP 500 with
`{"error": "Chat pipeline is not implemented yet"}`. It does not generate answers.

Stage 2.1 provides reusable single-document ingestion, structural issue reporting,
checksum-bound context corrections, and deterministic build/chunk identities.
The explicit nine-document bootstrap produces 153 tables and 113 chunks. Parsing
a candidate document does not add it to the answer corpus or approve it.

Stage 3 adds local Ollama BGE-M3 embeddings and exact NumPy cosine retrieval.
The approved 113-chunk index preserves full source metadata and checks source,
build, and embedding-model identities before use. A 24-query benchmark records
English, Arabic, mixed-language, Franco-Arabic, and out-of-KB retrieval behavior.

Stage 3.1 adds conservative multilingual normalization, reviewed same-name aliases,
local BM25, and reciprocal rank fusion. Dense remains the unchanged default;
`search --mode lexical` and `search --mode hybrid` select the new paths. No new
dependencies, models, ingestion rebuild, or vector rebuild are needed.

Generation, session persistence, frontend, and Docker implementation remain
future work. The API contract and unfinished `/chat` behavior are unchanged.

## Structure

- `src/medical_rag/`: application packages for configuration, API, ingestion,
  retrieval, generation, safety, sessions, analytics, handoff, and observability.
- `frontend/`: user chat and administrative interface boundaries.
- `data/knowledge_base/source/`: immutable approved source documents.
- `data/knowledge_base/processed/`: extracted source blocks, headings, and table arrays.
- `data/knowledge_base/chunks/`: section-aware chunks and citation provenance.
- `data/knowledge_base/metadata/`: ingestion manifest and document checksums.
- `storage/`: runtime vector stores, sessions, analytics, and logs.
- `tests/`: unit, integration, API, safety, retrieval, and evaluation tests.
- `scripts/`: ingestion, evaluation, benchmarking, and maintenance entry points.
- `docs/`: architecture, API, safety, evaluation, and deployment documentation.

## Setup and usage

From the repository root in PowerShell, using the existing Python environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn medical_rag.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

`LLM_MODEL` and `EMBEDDING_MODEL` configure the model identities at application
startup. The defaults, `qwen2.5:7b` and `bge-m3`, are assessment examples, not a
completed model-selection decision. Set environment variables before starting, or
copy `.env.example` to `.env` and add `--env-file .env` to the command. A `.env`
file is not loaded automatically. Health does not verify model connectivity or
readiness. Blank model names fail configuration validation at startup.

Rebuild the nine approved sources and their chunks:

```powershell
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
```

See the [ingestion walkthrough](docs/ingestion.md) for output formats, chunking
choices, source examples, and limitations. Rebuilds replace fixed JSON outputs
and never save changes to the original DOCX files. Generated data stays ignored
by Git under the existing rules.

Future upload code can call
`medical_rag.ingestion.processing.pipeline.ingest_document()` to obtain parsed
data, chunks, checksums, build identity, and issues without writing any KB files.
Medical approval, persistence, and indexing remain outside that function.

Build, search, and evaluate the approved bootstrap index:

```powershell
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py build
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --question "What is the active ingredient in Cataflam?"
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode hybrid --question "Cataflam byetkawen men eh?"
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py compare
```

Omit `--question` for interactive search. These commands require a running local
Ollama server with the configured embedding model already installed for dense
and hybrid modes; lexical requires the published index but makes no model calls. They never
download models. Configure `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`,
`OLLAMA_TIMEOUT_SECONDS`, `EMBEDDING_BATCH_SIZE`, and `RETRIEVAL_INDEX_DIR` through
environment variables. Retrieval commands do not load `.env` automatically.
See the [retrieval guide](docs/retrieval.md) and
[measured evaluation](docs/retrieval_evaluation.md) for details and limitations.
The [multilingual walkthrough](docs/retrieval_multilingual.md) explains the new
modes; the [Stage 3.1 comparison](docs/retrieval_comparison.md) records frozen and
held-out results, failures, provenance checks, and timings. `evaluate` defaults to
`reports/evaluation/stage31_<mode>.json`, preserving the original Stage 3 report.

Run all tests without starting a model server:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

`pytest.ini` makes the existing `src/` package importable during tests.
`requirements.txt` uses UTF-8 for tool compatibility; the original Stage 1 pins
are preserved. Stage 2 adds only `python-docx==1.2.0` and its required XML parser,
`lxml==6.1.3`. Stage 3 adds only `numpy==2.5.4`; HTTP calls reuse HTTPX.
The ingestion smoke test reads the approved sources and uses pytest
temporary directories for outputs. A restricted environment may need permission
to use those temporary directories.

## Documentation

- [Architecture](docs/architecture/README.md)
- [API](docs/api/README.md)
- [Document ingestion](docs/ingestion.md)
- [Dense retrieval](docs/retrieval.md)
- [Retrieval evaluation](docs/retrieval_evaluation.md)
- [Multilingual retrieval](docs/retrieval_multilingual.md)
- [Stage 3.1 comparison](docs/retrieval_comparison.md)
- [Medical safety](docs/safety/README.md)
- [Evaluation](docs/evaluation/README.md)
- [Deployment](docs/deployment/README.md)
- [Fresh-machine setup](docs/NEW_MACHINE_SETUP.md)
- [GitHub handoff review](docs/GITHUB_HANDOFF_REVIEW.md)
- [Ollama zero-vector troubleshooting](docs/OLLAMA_ZERO_VECTOR_ISSUE.md)
- [Model decision record](MODEL_DECISION.md)
- [Token and performance report](TOKEN_USAGE.md)
