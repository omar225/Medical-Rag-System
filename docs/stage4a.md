# Stage 4A grounded medical chat

This milestone implements the first `/chat` service over the existing nine-source,
113-chunk corpus. It is a development prototype, not a clinically validated system.
It changes no approved DOCX/chunk files, retrieval aliases, embedding code, BM25,
RRF constants, frozen retrieval fixtures, or vector files. There is no frontend,
admin service, database, session persistence, or deployment change.

## Request flow and important functions

`api.main.chat()` retains the public Pydantic contract and dispatches blocking
work off the FastAPI event loop. Operational exceptions/timeouts become exactly
HTTP 500 `{"error":"Chat service failed or timed out; please retry later."}`.
No exception details, model reasoning, source contents, or user input are logged.
Successful telemetry logs contain only category, timings, and token counts.

`services.chat.ChatService.run()` performs, in order:

1. `safety_category()` checks conservative multilingual emergency, dosing,
   personalized decision, and dangerous-request rules. Escalation happens before
   any corpus read, embedding, or generation. Static escalation messages are
   separate from KB medical information.
2. `load_bootstrap_corpus()` validates the active approved sources/builds, then
   the existing `HybridRetriever.search(message, top_k=5)` runs unchanged.
3. `EvidenceChecks.validate_results()` compares complete chunk/document metadata
   with the active corpus and its fingerprint. Wrong/stale metadata is an
   operational error, never an `out_of_scope` classification.
4. `admit()` requires explicit question subject coverage and the requested
   information field in a passage. RRF/cosine values are not used as support
   thresholds. Unknown intents, absent subjects, ambiguous mixed requests, no
   evidence, and detected instruction-like passages abstain. Combination-only
   tables cannot establish standalone-agent effects.
5. `build_messages()` sends one complete matching approved passage, labelled
   with its original retrieval position E1..E5, to `OllamaChat.generate()`.
   It admits whole passages with a conservative UTF-8 byte budget including
   schema/output/template reserves; it never silently truncates source text.
6. One non-streaming `/api/chat` call returns a strict internal `GroundedDraft`:
   proposed answer, `evidence_sufficient`, and labelled exact supporting quotes.
   For ingredient lookups, the decoder's quotation/label choices are constrained
   to exact matching table rows from this request, preventing reconstructed quotes.
   Instructions require source-only facts and treating source/question text as
   data. No agents, tools, second LLM, or retry/repair generation is involved.
7. The source loader runs again before publication. `verify()` rejects unknown
   labels, fabricated quotations, wrong information subsections, or unsupported
   subject associations. `field_text()` prevents quoting effects as
   contraindications. Subject context can come from the source's class heading
   or uniform standalone-ingredient table, not another drug's nearby mention.
8. `verify_paraphrase()` checks every answer segment against its quotations,
   using conservative vocabulary/number/qualifier checks. Ingredient answers
   must retain the complete table value; same-ingredient brand comparisons must
   retain complete formulation qualifiers. This is intentionally restrictive.
9. If references and subject/field checks pass but the paraphrase fails, Python
   discards the proposed wording and publishes only the **exact verified source
   excerpts**, with a localized introduction (`verified_excerpts`). Invalid
   references never receive this fallback. Model-declared insufficient evidence
   and unsupported references produce `out_of_scope` with empty citations.
10. Citation fields come exclusively from stored document/chunk metadata, with
    the verified source quote as `snippet`. `answer_language()` classifies the
    actual final text, including an extractive fallback; it does not trust a
    model-provided language or copy the input's language label.

`open_chat_service()` owns request-scoped HTTPX clients and composes the existing
retrieval classes. Health starts without connecting to Ollama. Nothing builds an
index, downloads a model, or stores `session_id`.

## Model and configuration

`ollama list` confirmed Qwen was already installed. No model was downloaded.
The selected `qwen3.5:4b` is Q4_K_M, 4.2B parameters, digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.
`qwen2.5:7b` remains configurable as a later comparison, not a tested alternative.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL` | `qwen3.5:4b` | Installed generation model |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `768` | JSON output cap; token-limit termination fails |
| `LLM_CONTEXT_SIZE` | `8192` | Context passed to Ollama |
| `LLM_TIMEOUT_SECONDS` | `25` | Show + generation budget |
| `CHAT_TIMEOUT_SECONDS` | `30` | HTTP response deadline |

Retrieval keeps the existing `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, index, and
alias configuration. `.env` still requires Uvicorn `--env-file .env`.
`/api/show` advertises Qwen's `thinking` capability; the request explicitly uses
`think:false`. Alternative models without that capability receive no thinking
parameter; runtimes advertising named controls use their advertised low/default
value. This follows [Ollama's thinking controls](https://github.com/ollama/ollama/blob/main/docs/capabilities/thinking.mdx)
and [structured chat API](https://github.com/ollama/ollama/blob/main/docs/api.md).
The model's own other sampling defaults remain in effect; seed is explicitly 0.

## Examples

Supported general information returns `answered` with real source metadata, for
example Cataflam's source row `Cataflam | Diclofenac potassium` from KB-03.
Unsupported malaria treatment and job-letter questions return `out_of_scope` with
empty citations. Adult daily medication-dose requests and severe breathing
distress with blue lips return `escalated` with empty citations. Complete actual
responses and measurements are in the separate evaluation report.

Franco-Arabic intent cues request Arabic-script generation. Arabic responses may
preserve source medical names/clinical lines in English and therefore truthfully
receive `mixed`. The evidence comparison uses the existing reviewed aliases and
a small, explicit translation vocabulary in `generation/evidence.py`; this view
does not modify retrieval inputs, aliases, ranking, or vector representations.

## Tests and evaluation

The default offline suite mocks every chat model/embedding boundary. New tests
cover supported/unsupported questions, multilingual full responses, paraphrased
emergencies/dosing/personal decisions, invalid references and quotes, provenance
changes, incorrect field/combination scope, rejection of hallucinated proposed
wording, extractive fallback, injection, unavailable/timed-out models, malformed
JSON, token-limit termination, HTTP 500s, deadline handling, and schema invariants.
Existing API tests now inject an operational failure instead of asserting the
obsolete unimplemented-chat placeholder; the required response shapes remain.

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
$env:RUN_OLLAMA_TESTS='1'
$env:RUN_CHAT_TESTS='1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_ollama_embeddings.py tests/integration/test_live_chat.py -p no:cacheprovider
.\.venv\Scripts\python.exe scripts/evaluation/chat.py --cold
```

`tests/evaluation/datasets/chat_stage4a.json` contains the exact ten questions
supplied in this session and 18 additional English, Arabic, mixed, and Franco
cases. The application never imports it. The known Arabic salbutamol top-five
miss expects conservative abstention, explicitly distinguished from absent corpus
coverage. Results are reported separately for MTI and additional cases.

`scripts/evaluation/chat.py` evaluates the actual FastAPI route via TestClient.
It records expected/actual status, factual phrase spot-checks, independent
retrieved/approved citation checks, language, HTTP latency, prompt/generated
tokens, model-load/prompt/generation times, source/code/dataset/model identities,
and sampled whole-device VRAM. Phrase checks are **not** full correctness or
entailment evaluation: all medical outputs still need human review.

The canonical raw run is `reports/evaluation/stage4a_chat.json`; reports remain
ignored runtime artifacts. See `docs/evaluation/stage4a_results.md` for the measured
results and limitations. Original Stage 3.1 reports are preserved.

## Limits and next recommendations

The rule-based safety gate cannot recognize every spelling, context, negation,
or indirect unsafe request. It can over-escalate hypothetical/negated questions.
Severe respiratory distress and blue lips receive immediate escalation, consistent
with [NHS emergency guidance](https://www.nhs.uk/symptoms/shortness-of-breath/),
without assuming a country-specific emergency telephone number.

Exact citations establish provenance, not medical truth. Lexical grounding cannot
prove relationships, causal entailment, completeness, correct translation, or
clinical appropriateness. The extractive fallback preserves source qualifiers
without asserting that an unverified generated translation is correct. Source
medical accuracy, terminology, contraindication scope, and safety policy require
clinician review; engineering approval of sources is not clinical approval.

Only explicit supported information intents are admitted. Missing terminology,
fixed top-five retrieval misses, unrecognized paraphrases, cross-document requests,
or instruction-like source text can yield conservative false abstentions. There
is no calibrated answerability score and no claim that all KB questions work.
Prompt-injection screening is incomplete defense in depth, not a security proof.

The HTTP deadline returns a generic 500, but cancellation cannot instantly stop a
blocking worker or remote model task. Clients close at dependency cleanup and
stage deadlines stop subsequent work. No response is published after the route
times out. Serial local benchmarks do not establish concurrent throughput or
universal cold-start compliance. No concurrency/load deployment was implemented.

Next: clinician review of the recorded outputs, broader unseen/adversarial safety
and evidence tests, and independent model/translation evaluation. Preserve the
fixed retrieval baseline; do not tune it to the ten development questions.
Stop at Stage 4A; frontend, deployment, sessions, and further milestones are not
part of this change.
