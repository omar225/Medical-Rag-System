# Token Usage and Performance

## Measurement methodology

Retrieval timings are measured using wall-clock timers around index construction
and individual searches, with cold/model-unloaded probes reported separately.
See [Stage 3 evaluation](docs/retrieval_evaluation.md) and
[Stage 3.1 comparison](docs/retrieval_comparison.md) for conditions and raw-report
locations. Exact per-chunk tokenizer counts and assistant token usage were not
collected; no such measurements are claimed.

## Results

The frozen Stage 3.1 comparison measured median dense / lexical / hybrid query
latencies of 82.31 / 47.59 / 138.63 ms. The first lexical search includes a
994.53 ms cache construction. The separate unloaded-model probe took 2.288 seconds
followed by an 88.10 ms query. These are recorded original-machine observations,
not predictions for a new GPU. Later Ollama returned invalid all-zero embeddings;
that failure remains unresolved and is documented separately.

## Optimization findings

Normalization, BM25 and RRF improved the frozen hybrid Hit@5 to 100% and held-out
Hit@5 to 95.83%. There was no normalization-only ablation and no benchmark-driven
token-count optimization. Reproduction on another machine must record its GPU,
driver, runtime and model digest before comparing performance.
