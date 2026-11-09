# MTI API contract - Stage 4A

## POST /chat

Request JSON:

```json
{"session_id": "abc123", "message": "What is the active ingredient in Cataflam?"}
```

`message` is a required string. Leading/trailing whitespace is trimmed; empty or
whitespace-only input is rejected. Numbers, arrays, objects, booleans, and null
are rejected rather than converted to text. `session_id` is an optional string
and may be omitted or null. Omission represents a new logical session in the
contract; Stage 4A does not create, store, or resume sessions. Unexpected
request fields are rejected.

Stage 4A runs safety checks, hybrid retrieval, evidence assessment, one structured
local Ollama generation call, and Python source/citation/grounding verification.
Operational failures, including dependency construction and timeouts, return HTTP 500:

```json
{"error": "Chat service failed or timed out; please retry later."}
```

Invalid requests return FastAPI's HTTP 422 `{"detail": [...]}` validation errors.
Normal classification responses use HTTP 200. Source/LLM unavailability is never
reported as a medical scope decision.

The successful response uses HTTP 200 and exactly these fields:

| Field | Type / rule |
| --- | --- |
| `answer` | Nonblank plain-text string |
| `citations` | List of citation objects |
| `status` | `answered`, `out_of_scope`, or `escalated` |
| `language` | `ar`, `en`, or `mixed`; describes the **answer**, not the request |

Each citation has exactly `doc_id`, `title`, `section`, and `snippet`, all required
nonblank strings. `answered` requires at least one valid citation. `out_of_scope`
and `escalated` require an empty citation list. Extra response/citation fields are
rejected. Python verifies the active corpus, complete chunk/document identities,
retrieval membership, source subsection/subject association, and exact supporting
excerpts. Unverified generated paraphrases are discarded; a source-only extractive
fallback is available only after references and evidence scope pass validation.
The language field describes the actual answer's script, including any fallback.

Processing failures use HTTP 500 with exactly `{"error": "..."}`. The complete
RAG pipeline has a configurable 30-second HTTP deadline. Recorded serial local
measurements are in the [Stage 4A results](../evaluation/stage4a_results.md), with
limitations; they do not guarantee universal cold-start or concurrent compliance.

## GET /health

HTTP 200, with exactly the keys below (values shown are configurable defaults):

```json
{"status": "ok", "llm": "qwen3.5:4b", "embedding": "bge-m3"}
```

`LLM_MODEL` and `EMBEDDING_MODEL` are read from the environment when the app is
created. Health reports API liveness and configured model identities only; it
does not verify readiness. No models are downloaded automatically.

## Running and testing

See the repository [README](../../README.md) for PowerShell commands. Interactive
OpenAPI documentation is available at `/docs`, and its JSON schema at
`/openapi.json`. Tests use an in-process TestClient and synthetic citation data;
they do not require Ollama or a GPU. Opt-in live tests separately exercise the
installed Qwen model, actual BGE-M3 embeddings, and approved source corpus.
