# MTI Medical RAG: drug-information chatbot

A chatbot for pharmacists and doctors that answers **only** from the nine approved knowledge-base
documents (KB-01 … KB-09). It cites the exact source lines it used, and it escalates emergencies,
dose requests and individual treatment decisions. It accepts Arabic (including Egyptian dialect and
Franco-Arabic), English and mixed questions.

| | |
| --- | --- |
| LLM | `qwen3.5:4b`, open weights, served by Ollama. Why: [MODEL_DECISION.md](MODEL_DECISION.md) |
| Embeddings | `bge-m3`, open weights, served by Ollama |
| Tokens and latency | [TOKEN_USAGE.md](TOKEN_USAGE.md) |
| Test results | [docs/evaluation.md](docs/evaluation.md): 199 labelled questions, 98% correct status, 100% genuine citations |
| Design | [docs/architecture.md](docs/architecture.md) |

## Run it with Docker (one command)

Requirements: Docker with Compose v2, about 12 GB of free disk space (Ollama image ≈ 6 GB, models
≈ 4.5 GB, app image ≈ 0.6 GB), and internet access on the first start.

```bash
docker compose up --build
```

Then open <http://localhost:8000> for the chat UI. The API is on the same port (`POST /chat`,
`GET /health`).

With an NVIDIA GPU (much faster; needs the NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### What starts

| Container | Image | Job |
| --- | --- | --- |
| `ollama` | `ollama/ollama` (pinned tag) | Serves both open-weights models on the internal network. No port is published. |
| `models` | same image | Runs once: `ollama pull qwen3.5:4b` and `ollama pull bge-m3`, then exits. `app` waits for it to succeed. |
| `app` | built from the [Dockerfile](Dockerfile) | FastAPI service plus the built React UI, on port 8000. |

### Where models and data live

| What | Where | Lifetime |
| --- | --- | --- |
| Model weights (≈ 4.5 GB) | Docker volume `ollama-models` | Downloaded on the first start, reused afterwards |
| Approved KB (DOCX sources, processed JSON, chunks, reviewed aliases) | Copied into the `app` image from `data/knowledge_base/` | Fixed per image; checksums are verified at startup |
| Vector index (113 × 1024 floats) | Docker volume `vector-index` | Built from the KB on the first start (seconds), rebuilt automatically if the KB or embedding model changes |
| Chat sessions | `app` memory (last 3 turns per `session_id`, 1-hour expiry) | Lost on restart; no patient data is stored |

On the first start, `models` downloads the weights, then `app` builds the index and starts serving.
Later starts skip both steps. To start completely fresh: `docker compose down -v`.

### Configuration and secrets

No configuration is required, and the project needs **no secrets or API keys**. Optional settings
(model names, timeouts, port) are listed in [.env.example](.env.example). Copy it to `.env` and
Compose picks it up. `.env` files are excluded from the image by [.dockerignore](.dockerignore) and
from Git by [.gitignore](.gitignore).

On CPU only, a 4B model can exceed the 30-second budget on long answers. A GPU is recommended.

## API (exact MTI contract)

`POST /chat`

```json
{"session_id": "abc123", "message": "ايه المادة الفعالة في Controloc؟"}
```

```json
{"answer": "المادة الفعالة في Controloc هي Pantoprazole، ...",
 "citations": [{"doc_id": "KB-02", "title": "...", "section": "... > 1. Proton Pump Inhibitors (PPIs)",
                "snippet": "Controloc | Pantoprazole"}],
 "status": "answered", "language": "ar"}
```

- `status`:
  - `answered`: the KB supports the answer; at least one citation.
  - `out_of_scope`: unrelated, or not in the KB.
  - `escalated`: emergency, dose request, harmful request, or an individual clinical decision.
- `session_id` is optional. With it, the last 3 turns are kept, so follow-ups like *"and its side
  effects?"* work. Without it, each message is independent.
- `GET /health` → `{"status": "ok", "llm": "qwen3.5:4b", "embedding": "bge-m3"}`.
- An operational failure, or no answer within 30 s, returns HTTP 500 `{"error": "..."}`. Invalid
  input (for example an empty message) returns HTTP 422.
- The same routes are also served under `/api/...` for the web UI.

## How it works

```text
startup   approved DOCX → checked chunks (one per drug class) → NumPy vectors + BM25 + medicine-name index

request   message (+ last turns of the session)
  1. safety screen      deterministic, high recall: emergency / dose request / harmful intent → escalated
  2. medicine names     brands, ingredients and classes from the KB tables + reviewed Arabic aliases
                        (typo-tolerant; Arabic spellings of Latin brands matched by sound, as hints)
  3. retrieval          sections defining the named medicines + hybrid BGE-M3/BM25 (RRF), ≤ 6 whole chunks
  4. one Qwen call      JSON: category (answer / out_of_scope / needs_clinician / emergency), answer, quotes
  5. verification       every quote must exist in its chunk (published snippets are real source lines);
                        every medicine name and number in the answer must appear in the cited chunks;
                        one retry if a check fails, the answer was cut off, or the model refused a
                        question naming a KB medicine; otherwise a safe refusal
```

Deterministic code handles only the red flags and the grounding checks; the language model does the
language understanding. [docs/architecture.md](docs/architecture.md) explains each choice.

### Safety policy (for professional users)

| Situation | Behaviour |
| --- | --- |
| Possible emergency now (breathing, blue lips, unconsciousness, seizure, severe allergy, chest pain, bleeding, very low sugar, overdose or ingestion, self-harm) | `escalated` with emergency instructions, before any retrieval |
| Request for a dose, amount or frequency (the KB contains no doses) | `escalated`, refer to the leaflet or prescriber |
| Diagnose, choose or change a treatment for a specific patient | `escalated` (model category `needs_clinician`) |
| "My patient has X, what does the KB say about Y?" | answered from the KB, without deciding for the patient |
| Not covered by the KB, or unrelated | `out_of_scope` |
| Answer cannot be grounded in the cited text | `out_of_scope` (never a guess) |

## Local development (without Docker)

Requires Python 3.12 and Ollama running locally with `ollama pull qwen3.5:4b` and `ollama pull bge-m3`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # Windows; use .venv/bin on Linux/macOS
.venv/Scripts/python -m uvicorn medical_rag.api.main:app --app-dir src --port 8000
```

The API serves the built UI from `frontend/dist` when it exists. For UI development (Node ≥ 22.12):
`cd frontend && npm ci && npm run dev` opens <http://127.0.0.1:5173> and forwards `/api` to port 8000.

| Command | What it does |
| --- | --- |
| `.venv/Scripts/python -m pytest -q` | Offline tests (no models needed) |
| `RUN_CHAT_TESTS=1 RUN_OLLAMA_TESTS=1 .venv/Scripts/python -m pytest tests/test_live.py` | Live checks against Ollama |
| `.venv/Scripts/python scripts/evaluate.py [--url http://localhost:8000]` | Labelled question sets → `reports/evaluation/results.{json,md}` |
| `.venv/Scripts/python scripts/ingest.py` | Re-ingest the nine approved DOCX files |
| `.venv/Scripts/python scripts/retrieval.py build` | Rebuild the vector index (also `search`, `evaluate`, `compare`) |
| `cd frontend && npm test` | Browser tests with the API mocked (`npm run test:live` uses the real API) |

## Repository layout

```text
src/medical_rag/
  api/            FastAPI app and the exact public schemas
  ingestion/      DOCX → structured blocks → section chunks (approved sources only)
  retrieval/      corpus checks, Arabic normalization and reviewed aliases, BGE-M3 client,
                  NumPy index, dense / BM25 / hybrid (RRF) search
  knowledge/      knowledge base loaded once per process; medicine-name index
  safety/         deterministic safety screen, language helpers, fixed user messages
  generation/     Ollama client, prompt, verification checks
  services/       request flow, greeting replies, session history
data/knowledge_base/   approved DOCX sources, processed JSON, chunks, reviewed aliases and corrections
frontend/              React + Vite chat UI
tests/                 one test module per component; datasets/ holds the labelled question sets
scripts/               evaluate.py, ingest.py, retrieval.py
reports/evaluation/    recorded evaluation results
```

## Limitations

- The knowledge base is the only source. Anything outside it is refused, even when well known.
- Verification checks quotes, medicine names, brand/ingredient pairs and numbers. It does not prove
  that every paraphrase is semantically exact; the evaluation sets and the citations make that
  reviewable.
- Session history is in memory and is lost on restart.
- Changing the approved documents needs re-ingestion (`scripts/ingest.py`) and a new image; the index
  rebuilds automatically.
