# Medical RAG System

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

Model integrations, embeddings, retrieval, generation, session persistence,
frontend, and Docker implementation remain future work. The API contract and
unfinished `/chat` behavior are unchanged.

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

Run all tests without starting a model server:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

`pytest.ini` makes the existing `src/` package importable during tests.
`requirements.txt` uses UTF-8 for tool compatibility; the original Stage 1 pins
are preserved. Stage 2 adds only `python-docx==1.2.0` and its required XML parser,
`lxml==6.1.3`. The ingestion smoke test reads the approved sources and uses pytest
temporary directories for outputs. A restricted environment may need permission
to use those temporary directories.

## Documentation

- [Architecture](docs/architecture/README.md)
- [API](docs/api/README.md)
- [Document ingestion](docs/ingestion.md)
- [Medical safety](docs/safety/README.md)
- [Evaluation](docs/evaluation/README.md)
- [Deployment](docs/deployment/README.md)
- [Model decision record](MODEL_DECISION.md)
- [Token and performance report](TOKEN_USAGE.md)
