# Medical RAG System

Moving computers: follow [fresh Windows setup](docs/NEW_MACHINE_SETUP.md),
[the private GitHub handoff review](docs/GITHUB_HANDOFF_REVIEW.md), and
[the unresolved Ollama issue](docs/OLLAMA_ZERO_VECTOR_ISSUE.md). The reviewed
[recovery snapshot](reproducibility/README.md) preserves the exact stopped state.
The project is backed up in the private
[GitHub repository](https://github.com/omar225/Medical-Rag-System).
The nine-document Stage 2.1 chunks, processed JSON, and ingestion manifest are
directly tracked; cloning does not require an ingestion rebuild.

## Overview

This repository is the structural foundation for a multilingual, evidence-grounded
medical retrieval-augmented generation system. The approved source documents are
preserved under `data/knowledge_base/source/` and are organized by KB ID.

## Project status

Stage 4A implements `POST /chat`: conservative safety checks, the existing hybrid
retriever, explicit subject/field evidence checks, one structured local Qwen call,
and Python citation/grounding validation. Unsupported questions abstain; unsafe
requests escalate; operational failures return HTTP 500. `GET /health` remains a
liveness/configuration endpoint. See the [Stage 4A walkthrough](docs/stage4a.md)
for the exact architecture, limitations, measurements, and function explanations.

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

Stage 5 adds a bilingual React/Vite/TypeScript chat interface in `frontend/`,
connected to this FastAPI app through a local development proxy. It includes
expandable real-source citations, Arabic/English layout, distinct safety/scope
states, and technical error handling. See [the frontend guide](frontend/README.md)
and [Stage 5 verification](docs/stage5.md) for setup, screenshots, and results.

Stage 5.1 adds friendly conversation and clarification, conservative spelling
handling, generic-versus-brand ingredient recognition, a bounded lexical retry,
and readable verified-source answers. See the [diagnosis and evaluation report](docs/stage51.md)
for before/after examples, remaining Arabic limitations, and the two held-out false abstentions.

The API response shape is unchanged. Session persistence, admin, and Docker
implementation remain future milestones. Retrieval code, aliases, source artifacts,
safety rules, original evaluation fixtures, and vector files remain preserved.

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
startup. Stage 4A defaults to `qwen3.5:4b` and `bge-m3`; both must already be
installed in local Ollama. Nothing downloads models automatically. Set environment variables before starting, or
copy `.env.example` to `.env` and add `--env-file .env` to the command. A `.env`
file is not loaded automatically. Health does not verify model connectivity or
readiness. Blank model names fail configuration validation at startup.

Rebuild the nine approved sources and their chunks:

```powershell
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
```

See the [ingestion walkthrough](docs/ingestion.md) for output formats, chunking
choices, source examples, and limitations. Rebuilds replace fixed JSON outputs
and never save changes to the original DOCX files. The nine approved processed
and chunk JSON files and ingestion manifest are tracked for fresh-clone use;
other generated outputs remain ignored. Rebuilding is optional maintenance.

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
For a restricted Windows account, create a writable temporary parent and pass
a new, disposable `--basetemp` directory beneath it; pytest clears that directory.
Live chat tests require `RUN_CHAT_TESTS=1`. The separate end-to-end evaluation is:

```powershell
.\.venv\Scripts\python.exe scripts/evaluation/chat.py --cold
```

This reads the ten user-supplied MTI questions plus additional development cases
and writes `reports/evaluation/stage4a_chat.json`. `--cold` unloads only the two
configured Ollama runners before an extra cold request; it does not clear OS or
driver caches. The application never reads evaluation fixtures.
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
