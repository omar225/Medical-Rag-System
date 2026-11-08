# Model Decision Record

## Requirements

Stage 3 requires local multilingual dense embeddings for English, Arabic,
Egyptian phrasing, mixed terminology, and representative Franco-Arabic queries.
Embedding retrieval must preserve source provenance and reject context overflow
instead of silently truncating text. No LLM generation is implemented.

## Candidate models

The requested initial embedding model is Ollama `bge-m3:latest`. Other models
have not been benchmarked in this stage. LLM selection remains pending.

## Evaluation and measurements

The installed model was already available; no model was downloaded. The local
Ollama runtime reported version `0.40.15`, F16 BGE-M3, and model digest
`7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
The 113 source chunks produced 1,024-dimensional vectors in 8.79 seconds after
an initial model-load probe. The probe itself took 36.61 seconds, so these are
warm-model build/query measurements rather than a cold-start guarantee.

The frozen benchmark has 21 answerable queries and three out-of-KB controls.
Macro Recall@1/3/5 was 52.99% / 64.22% / 67.21%; Hit@5 was 80.95%.
See [the evaluation report](docs/retrieval_evaluation.md) for all queries and
the distinction between evidence recall and any-hit rate.

## Decision

Use BGE-M3 as the initial Stage 3 dense baseline, not as a medical-answerability
or safety mechanism. Arabic brand spelling and Franco-Arabic failures remain.
Do not treat this small benchmark as production clinical validation. A changed
model name or digest requires rebuilding the index; the implementation enforces
that requirement. No other model or generation integration was added.

## Stage 3.1 observations (2026-11-08)

The installed BGE-M3 model and source vectors remain unchanged. Conservative
query normalization, reviewable same-name/term aliases, local BM25, and fixed
RRF (k=60, 20 candidates per branch) improve evidence discovery without a new
model or dependency. Dense remains the unchanged selectable default. Hybrid
performs one rewritten-query embedding; lexical performs no model inference.

On the unchanged frozen benchmark, dense/lexical/hybrid Hit@5 is
80.95% / 95.24% / 100%. On separately written held-out questions (24 answerable,
five negative controls), it is 66.67% / 91.67% / 95.83%. No parameters or aliases
were adjusted after the held-out run. This is not externally blinded validation.

Frozen median latency was 82.31 / 47.59 / 138.63 ms; held-out median was
142.09 / 48.84 / 197.19 ms. The first lexical query includes a 994.53 ms cache
build. A separate model-unloaded probe took 2.288 seconds, followed by an
88.10 ms query; OS caches were not cleared. These measurements do not establish
a cold-start optimization over the earlier Stage 3 probe.

Hybrid still misses an unlisted Arabic spelling of Fastum Gel, can regress
top-one ranking, and returns neighbors for every negative control. It is a
reasonable candidate for controlled Stage 4 development, with broader terminology
review and evaluation required before medical use. A subsequent opt-in live check
failed because Ollama returned all-zero vectors, reproduced after one manual
runner reload. The existing validator rejected them. Resolve this runtime failure
and pass the live check before proceeding to Stage 4. Stage 4 is not implemented.
See [the full comparison](docs/retrieval_comparison.md) and
[engineering explanation](docs/retrieval_multilingual.md).
