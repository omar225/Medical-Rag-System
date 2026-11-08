# Stage 3: local embeddings and exact dense retrieval

This page describes the unchanged dense baseline. Stage 3.1 selectable lexical
and hybrid paths are explained in the [multilingual guide](retrieval_multilingual.md)
and [comparison report](retrieval_comparison.md). The CLI now defaults evaluation
output to `stage31_<mode>.json`; the original Stage 3 report is preserved.

Stage 3 retrieves original evidence. It does not generate medical answers,
classify scope/safety, or change `POST /chat`. The original DOCX files and
Stage 2.1 ingestion outputs are not rewritten by retrieval.

## Commands

From the repository root in PowerShell:

```powershell
# Build the approved-nine index.
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py build

# Search by question; omit --question for an interactive prompt.
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --question "What is the active ingredient in Cataflam?" --top-k 5

# Write complete benchmark results, source metadata, and per-query latencies.
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate

# Run all offline tests. The live Ollama test is skipped unless opted in.
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --tb=short

# Run the separate live embedding integration check.
$env:RUN_OLLAMA_TESTS='1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_ollama_embeddings.py -p no:cacheprovider --tb=short
```

The live check is optional for normal development. If Ollama is unavailable,
start its local server. If the model is missing, the commands report the manual
setup command `ollama pull bge-m3:latest`. No code calls the pull endpoint or
downloads models automatically.

Configuration uses environment variables (not automatic `.env` loading):

| Variable | Default |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `EMBEDDING_MODEL` | `bge-m3` (resolved to `bge-m3:latest`) |
| `OLLAMA_TIMEOUT_SECONDS` | `180` |
| `EMBEDDING_BATCH_SIZE` | `8` |
| `RETRIEVAL_INDEX_DIR` | Project `storage/vector_store/dense/` |

`search --json` returns all result metadata rather than only human-readable
source text. `evaluate --dataset <path> --output <path>` selects another reviewed
fixture and output destination. No second public API endpoint was added.

## Embeddings and BGE-M3

An embedding is a numerical representation of text. A learned text encoder maps
questions and document passages into the same vector space, where meaningfully
related texts can have similar directions. It does not turn those texts into
verified medical facts.

Ollama tokenizes the question or source text and runs BGE-M3. This local model
returns 1,024 values per input. Both questions and documents use the same model
and model digest: different encoders can assign different meanings to the same
vector coordinates, even when their dimensions happen to match.

The client sends raw chunk text and raw questions, with no benchmark-specific
rewrites, brand alias dictionary, or Franco-Arabic transliteration. BGE-M3 does
not require query instructions, according to its
[official model card](https://huggingface.co/BAAI/bge-m3). Only its dense vectors
are used here. Sparse retrieval and multi-vector retrieval are not implemented.

Every embedding request is `POST /api/embed` with `truncate: false`. Ollama's
[API contract](https://docs.ollama.com/api/embed) specifies that over-context
input then produces an error instead of truncation. All 113 current chunks were
accepted. The model's advertised context is 8,192 tokens; no exact per-chunk
tokenizer count is claimed. A mocked over-context test verifies that the entire
input is submitted and the error is surfaced without a truncating retry.

The client also checks response model identity, vector count, consistent
dimensions, finite numeric values, float32 representability, and nonzero vector
length. Timeouts, connection errors, missing models, and malformed responses
produce operational errors. The implementation reuses HTTPX and adds only NumPy.

## Corpus selection is separate from ingestion

`prepare_approved_corpus(chunks, approved_documents)` accepts an arbitrary set
explicitly selected by a trusted caller. It validates each chunk's document ID,
source checksum, configuration/build identity, section path, source spans, and
membership. It preserves additional chunk and document provenance.

`load_bootstrap_corpus()` is the current policy adapter. It selects exactly the
nine IDs/filenames already in `APPROVED_SOURCES`; it never scans arbitrary chunk
files into the corpus. It checks original source hashes, processed metadata,
manifest counts, and reviewed corrections. Synthetic/staging files cannot join
by appearing in a directory, and adding another document to the bootstrap
manifest is rejected. Parsing success or `requires_review: false` does not
authorize indexing.

The existing approved bootstrap includes KB-04, whose original blank table value
still requires review. That flag and the original blank are preserved. No new
medical approval decision is made here. Unresolved section scopes are rejected
before indexing.

## Index construction and storage

1. Validate and canonically sort approved chunks by document ID and chunk ID.
2. Resolve the installed model's digest using `/api/tags`; do not pull it.
3. Embed source text in configured batches, checking count and dimensions at
   each batch boundary.
4. Confirm that the model digest did not change during the build.
5. Normalize vectors and write float32 NPY data with pickle disabled.
6. Write the complete JSON manifest and atomically replace `index.json` last.

The default directory contains:

- `vectors-<sha256>.npy`: immutable, checksum-named numerical rows.
- `index.json`: full original chunk text/metadata, document provenance, ordered
  chunk IDs, source/build identities, corpus fingerprint, model name/digest,
  dimensions, vector checksum, and measured build time.

The measured vector file is 462,976 bytes, of which 462,848 bytes are numerical
payload (`113 * 1024 * 4`). No vector database is needed for such a small matrix.
Older checksum-named vector files may remain after later builds; no automatic
retention/deletion policy is implemented.

Publishing JSON last ensures a failed embedding build does not replace the
active manifest. Readers verify the referenced numerical file before use. They
reject missing/invalid indexes, wrong model names or digests, changed corpus
fingerprints, mismatched row IDs/counts, damaged checksums, invalid dimensions,
and nonfinite or unnormalized vectors. A changed source/build requires a new
index build rather than silently serving stale evidence.

## Search flow and cosine similarity

`DenseRetriever.search(question, top_k=5)` validates the query, reloads the current
approved corpus snapshot, validates the index, and checks the live model digest.
It embeds the question, checks model identity again, and normalizes the result.

Cosine similarity is:

```text
score = dot(question, chunk) / (length(question) * length(chunk))
```

With normalized vectors, all scores are simply `matrix @ question`. NumPy
examines all 113 rows, sorts by score, and returns the requested top-k, capped
by corpus size. Ties follow deterministic canonical row order.

Each result has `score`, the complete `chunk`, its document provenance, and
embedding identity. Scores measure vector similarity, not a calibrated
probability or medical confidence. There is no arbitrary similarity threshold
and no scope/safety classification. Unrelated questions still return neighbors.

For later citations, `doc_id`, `title`, `section`, and actual source text can
supply the public citation fields. Source spans, source SHA-256, build ID, and
reviewed corrections allow the excerpt to be traced. Retrieval alone does not
prove that a complete answer exists, that every returned passage is relevant,
or that a treatment is appropriate for a particular patient.

## Evaluation and limitations

The fixture is `tests/evaluation/datasets/retrieval_queries.json`, outside
retrieval ranking logic. It was labeled from actual source passages before
running the benchmark. It is bound to the approved corpus fingerprint and
validates its evidence snippets before evaluation. Corpus changes require
reviewing the labels, not silently adapting expected rankings.

There are 21 answerable queries across six evidence topics and three out-of-KB
controls. Recall@k is the fraction of all labeled relevant chunks retrieved.
Hit@k records whether at least one appears. Brufen has seven labeled evidence
chunks, so its Recall@5 cannot reach 100% even with an ideal top five. Negative
controls are recorded with their actual neighbors and excluded from recall;
their manual benchmark labels are not runtime classifier outputs.

Full JSON results and per-query timing are written to
`reports/evaluation/stage3_retrieval.json`. See the dated
[evaluation snapshot](retrieval_evaluation.md) for measurements and inspected
good/bad results. All three sampled source/vector re-embedding checks (KB-03
NSAIDs, KB-07 Biguanides, KB-08 ARBs) reproduced their stored vectors at cosine
approximately 1.0, confirming actual row alignment beyond metadata checks.

Current limitations include Arabic brand spelling and ARB paraphrase misses,
weak Franco-Arabic retrieval, broad mixed-content chunks, no abstention
mechanism, and a very small benchmark. Model cold-start latency differs from
warm queries. The initial model-load probe took 36.61 seconds; do not interpret
the warm-query timings as an end-to-end deadline guarantee.

Document/build identities and row metadata allow a future caller to select new
approved builds and identify rows for replacement/deletion. The current CLI
builds the whole selected bootstrap; it does not implement incremental update
operations, uploads, approval workflows, or active-version administration.
Future updates can retain unaffected vectors when the embedding identity/input
format remains identical. Changing the model requires re-embedding the corpus.

No LLM generation, public API changes, ingestion rebuild, ANN framework, or
Stage 4 functionality was added.
