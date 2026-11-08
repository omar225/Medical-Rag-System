# MTI API contract - Stage 1

## POST /chat

Request JSON:

```json
{"session_id": "abc123", "message": "What is the active ingredient in Cataflam?"}
```

`message` is a required string. Leading/trailing whitespace is trimmed; empty or
whitespace-only input is rejected. Numbers, arrays, objects, booleans, and null
are rejected rather than converted to text. `session_id` is an optional string
and may be omitted or null. Omission represents a new logical session in the
future contract; Stage 1 does not create, store, or resume sessions. Unexpected
request fields are rejected.

Stage 1 accepts valid requests and always returns HTTP 500:

```json
{"error": "Chat pipeline is not implemented yet"}
```

Invalid requests return FastAPI's HTTP 422 `{"detail": [...]}` validation errors.
No medical answer or scope/safety classification is generated in Stage 1.

The future successful response uses HTTP 200 and exactly these fields:

| Field | Type / rule |
| --- | --- |
| `answer` | Nonblank plain-text string |
| `citations` | List of citation objects |
| `status` | `answered`, `out_of_scope`, or `escalated` |
| `language` | `ar`, `en`, or `mixed`; describes the **answer**, not the request |

Each citation has exactly `doc_id`, `title`, `section`, and `snippet`, all required
nonblank strings. `answered` requires at least one valid citation. `out_of_scope`
and `escalated` require an empty citation list. Extra response/citation fields are
rejected. Pydantic enforces these structural rules; future retrieval and evidence
checks must verify that citations contain actual approved KB excerpts and that
the answer is supported. Language accuracy and classification semantics also
require future pipeline logic.

Processing failures use HTTP 500 with exactly `{"error": "..."}`. The complete
RAG pipeline must meet a 30-second response deadline; end-to-end timeout handling
and performance verification are pending the pipeline implementation.

## GET /health

HTTP 200, with exactly the keys below (values shown are configurable defaults):

```json
{"status": "ok", "llm": "qwen2.5:7b", "embedding": "bge-m3"}
```

`LLM_MODEL` and `EMBEDDING_MODEL` are read from the environment when the app is
created. These defaults are examples; the project's model decision is pending.
Stage 1 health reports API liveness and configured model identities only. Model
connectivity/readiness checks will be implemented in a later stage.

## Running and testing

See the repository [README](../../README.md) for PowerShell commands. Interactive
OpenAPI documentation is available at `/docs`, and its JSON schema at
`/openapi.json`. Tests use an in-process TestClient and synthetic citation data;
they do not read the approved DOCX files or require Ollama, a GPU, or embeddings.
