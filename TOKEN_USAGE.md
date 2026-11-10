# Token usage and performance

**Hardware:** NVIDIA RTX 4060 Ti 8 GB, AMD Ryzen 5 7500F, 32 GB RAM, Windows 11, Ollama 0.40.15. Both
models ran fully on the GPU.

**How it was measured:**

- Token counts are Ollama's own counters (`prompt_eval_count`, `eval_count`). The service logs them
  for every request as `Chat measurements`, without question or answer text.
- Latency comes from the 199-question evaluation over HTTP (`scripts/evaluate.py --url`), which is
  how the hidden set will be run.
- The split of the prompt (system prompt, passages, question) was measured by sending each part to
  Ollama separately, for 40 answerable dev questions.

## Tokens per model call

| Call | When | Tokens in | Tokens out | Time (median) |
| --- | --- | --- | --- | --- |
| Safety screen, greeting replies | every request | 0 (no model) | 0 | < 10 ms |
| BGE-M3 query embedding | every request that reaches retrieval | ≈ 14 (9–25) | one 1024-dim vector | ≈ 20 ms for all of retrieval (embedding + BM25 + RRF + name lookup) |
| **Qwen generation, first call** | every request that reaches retrieval | **≈ 2,950** (1,930–3,840) | ≈ 210 (40–950) | 3.0 s (0.6 s reading the prompt, 2.4 s writing) |
| Qwen retry (13% of model requests) | a failed check, a cut-off answer, or a refusal of a named KB medicine, and ≥ 10 s left | ≈ 3,200 more (first prompt + draft + note) | ≈ 100 more | adds 1–10 s |
| BGE-M3 index build | first start only, or after a KB / model change | 35,369 (113 chunks) | 113 vectors | ≈ 9 s |

Out of 199 evaluation questions, 35 (18%) never reach a model: emergencies, dose requests, harmful
requests and greetings answer in under 10 ms. Requests that needed a retry used about 6,180 prompt
and 320 generated tokens in total.

### Inside the generation prompt (mean ≈ 2,960 tokens)

| Part | Tokens | Share |
| --- | --- | --- |
| Retrieved passages (5–6 whole drug-class chunks, ≈ 400 each) | ≈ 2,035 | 69% |
| System prompt and language instruction | ≈ 916 | 31% |
| Question, hints and notes | ≈ 13 | < 1% |
| JSON schema (constrained decoding, not prompt text) | 0 | 0% |

## Speed

| Path | p50 | p95 | Max |
| --- | --- | --- | --- |
| All 199 questions, end to end over HTTP | 2.6 s | 8.6 s | 27.6 s |
| Requests that reach the model (164) | 3.1 s | 9.4 s | 27.6 s (a cut-off answer, then the shorter retry) |
| Escalation, greeting (no model) | < 0.01 s | < 0.01 s | < 0.01 s |

| Stage of one model request | Measured |
| --- | --- |
| Retrieval (query embedding, BM25, fusion, name lookup) | 0.02 s p50, 0.04 s p95 |
| Prompt processing | ≈ 5,000 tokens/s → 0.6 s p50 |
| Answer generation | ≈ 73 tokens/s → 2.4 s p50, 7.9 s p95 |
| Model load after Ollama unloaded it (first request after idle) | up to 6.8 s, once |
| Startup: corpus checks, index load, BM25, name index | ≈ 1.5 s (plus ≈ 9 s if the index must be built) |

Every request finished inside the 30 s contract. On a CPU-only machine, generation speed drops
sharply; the README recommends the GPU override.

## Which call costs the most, and what I would do about it

**The Qwen generation call.** It accounts for 99% of tokens and over 95% of latency. Passages make up
most of its tokens, but output tokens make up most of its time: 73 tokens/s out against 5,000 tokens/s
in. A 300-token answer takes 4 s to write, while its 3,000-token prompt takes 0.6 s to read.

1. **Shorter outputs:** keep `think` off. With thinking on, the same 8 prompts produced 3,481 instead
   of 271 output tokens and took 48 s instead of 4.5 s ([MODEL_DECISION.md](MODEL_DECISION.md)). The
   prompt already asks for "only what was asked". Next step: cap list answers per question type.
2. **Fewer passages when a named medicine's defining chunk is found.** That chunk usually answers on
   its own; sending 2–3 chunks instead of 5–6 would cut 800–1,200 prompt tokens (≈ 30–40%).
3. **Reuse the prompt cache.** The system prompt is byte-identical and first in every request, so
   Ollama can reuse its key/value cache across requests; keep it that way.
4. **Cheaper retries.** A retry resends the whole prompt (≈ 3,200 tokens). Sending only the cited
   passages back would halve that.

## Which numbers could feed an admin panel

| Number | Admin panel in production? | Why |
| --- | --- | --- |
| Requests by status and reason (`answered`, `out_of_scope`, `escalated`, `unverified:*`, `model_*`) | Yes | Logged per request at no cost. Shows coverage gaps, safety load and refusal trends. |
| Prompt / generated tokens per request | Yes | Ollama reports them per call; they drive capacity planning and catch prompt growth. |
| End-to-end and retrieval latency (p50, p95) | Yes | Logged per request; the p95 against the 30 s contract is the main service-level target. |
| Retry rate (`verified_after_retry`, `unverified:*`) | Yes | An early signal of model or prompt regressions. |
| Model load time | Yes, as an alert | A high value means the model was unloaded; tells you to tune Ollama's keep-alive. |
| Prompt split (system / passages / question) | No, development only | Needs extra calls to measure; it guides prompt design, not live operations. |
| Thinking on/off comparison | No, development only | A one-off design experiment. |
| Tokens/s and per-stage timings on this workstation | No, development only | Hardware-specific; useful for tuning, not for monitoring another machine. |
| Evaluation accuracy and fact coverage | No, development only | Needs labelled questions; it is a release gate, not a live metric. |
