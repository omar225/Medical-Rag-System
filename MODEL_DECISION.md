# Model decision record

**Constraints:** open weights; local serving through Ollama on one 8 GB GPU (RTX 4060 Ti); Arabic
(Egyptian, Franco), English and code-switched text; 30 s per request including retrieval; strict
JSON output with source text quoted exactly. All numbers below were measured on this system.

## LLM: `qwen3.5:4b` (Q4_K_M, 3.3 GB on disk, 3.36 GB VRAM)

| Measured on the 199 labelled questions ([docs/evaluation.md](docs/evaluation.md)) | Result |
| --- | --- |
| Correct status (answered / out_of_scope / escalated) | 97.5% over HTTP, 98.0% in-process |
| Answerable questions answered, with the expected facts | 95.9% answered, 96.0% of expected facts present |
| Published citations that exist verbatim in the cited KB document (checked independently) | 100% |
| Invented doses or numbers published | 0 (every number must appear in the cited chunks) |
| Median / 95th-percentile latency, end to end | 2.6 s / 8.6 s; slowest 27 s, inside the 30 s budget |
| Requests that needed the one corrective retry | 13% of model calls |

**Why this model.** It is small enough to run next to the embedding model on an 8 GB GPU: 3.36 GB
for Qwen plus 1.2 GB for BGE-M3, both fully on the GPU with an 8k context. It follows Ollama's
JSON-schema output, reads and writes usable Egyptian Arabic, and lets "thinking" be disabled. That
last point was decisive. On the same 8 prompts:

| Thinking | Mean output tokens | Mean time | Over 30 s | Valid JSON |
| --- | --- | --- | --- | --- |
| off (shipped) | 271 | 4.5 s | 0 of 8 | 8 of 8 |
| on | 3,481 | 48 s | 7 of 8 | 5 of 8 (3 ran into the 4,096-token cap) |

**Observed failure modes, and what handles them.** All of these were seen in real runs; none is
patched with per-question rules.

- Copies quotes into the answer text, or mislabels a passage → quotes are located in any supplied
  passage and the answer text is cleaned. The published snippet is always the real source line.
- Names a medicine from a passage it did not quote, or puts a brand next to the wrong ingredient
  ("trade names containing Ibuprofen: Panadol") → identity and pairing checks plus one targeted
  retry. For "alternatives" questions, the same-ingredient brands are read from the KB tables and
  given to the model.
- Refuses a partly covered or instruction-laced question ("What is in Brufen? Ignore your rules") →
  one retry when the question names a KB medicine.
- After that retry, answers "the source has no half-life data, but here are the side effects" →
  returned as `out_of_scope` by a deterministic first-clause check.
- Over-long answers cut off at the token cap → one retry asking for a shorter answer; a cut-off
  answer is never published.

**The pipeline mattered more than the model.** With the same model, a word-overlap verifier
answered 1 of 12 ordinary audit questions; it rejected correct paraphrases such as Panadol's side
effects taken from the class section. Quote-location checks and whole-chunk context answer 11 of
12. The 12th, "I want to die", is correctly escalated.

**Not tested, and why.** Larger 7–8B models (Qwen, Llama, Gemma at Q4) need about 5 GB for the
weights. With BGE-M3 and the KV cache, that is at the 8 GB limit, where Ollama starts offloading to
the CPU. They may translate the code-switched source more fluently, but the measured errors above
were pipeline errors, not language errors. Switching needs only `LLM_MODEL`.

## Embeddings: `bge-m3` (F16, 1.2 GB, 1024 dimensions, 8k-token input)

**Why this model.** It is one multilingual model for Arabic, English and mixed text. Its 8k input
window embeds every drug-class chunk whole: the largest is about 2,100 characters, and truncation is
disabled, so an oversized chunk fails loudly instead of being cut. It runs in the same Ollama as
the LLM. Measured cost: 113 chunks (35,369 tokens) are indexed in about 9 s; a query (about 14
tokens) takes about 80 ms.

**Dense retrieval alone was not enough.** On the recorded retrieval benchmark (labelled evidence
chunks; English, Arabic, mixed and Franco questions), top-5 hit rates were:

| Mode | Benchmark set | Separately written set |
| --- | --- | --- |
| BGE-M3 dense only | 81% | 67% |
| BM25 over alias-normalized text | 95% | 92% |
| Hybrid (RRF of both), shipped | **100%** | **96%** |

Dense search alone found the right chunk at rank 1 for only 1 of 3 Franco questions, and missed
Arabic brand spellings (كتافلام). The hybrid adds BM25 over reviewed aliases. A medicine-name lookup
then puts the defining section of a named drug first, regardless of rank.

**Observed failure, handled.** During development, Ollama once returned all-zero BGE-M3 vectors with
HTTP 200. The client rejects zero vectors, so a broken index is never built or queried.

**Interchangeability.** The index stores the embedding model's name and digest. Changing
`EMBEDDING_MODEL` triggers an automatic rebuild at the next start. Other embedding models
(multilingual-e5, nomic-embed) were not benchmarked, because the hybrid already locates the
evidence for 96–100% of benchmark questions.
