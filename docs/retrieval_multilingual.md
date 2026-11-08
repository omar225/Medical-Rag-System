# Stage 3.1: multilingual retrieval walkthrough

Stage 3.1 adds local lexical and hybrid retrieval over the existing 113 approved
chunks. The Stage 3 dense retriever, vector index, original 24 questions, source
DOCX files, and ingestion outputs remain unchanged. No dependencies or models
were added. `POST /chat` still has its existing unfinished behavior.

## Architecture and commands

```text
question -> dense ------------------------------------> cosine ranking
question -> normalization + reviewed aliases -> BM25 --> lexical ranking
question -> normalization + reviewed aliases -> BM25 --+
                                               BGE-M3 --+-> RRF -> hybrid ranking
```

These are three selectable paths. Dense receives the original input. Hybrid
embeds the rewritten question once, using the existing dense retriever; it does
not also embed the original question. Source vectors remain exactly as built
in Stage 3. The lexical path makes no model requests.

```powershell
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode dense --question "Cataflam byetkawen men eh?"
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode lexical --question "Cataflam byetkawen men eh?"
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode hybrid --question "Cataflam byetkawen men eh?" --json
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate --mode hybrid
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py compare
```

Dense remains the default. `evaluate` now writes `stage31_<mode>.json`, preserving
the original `stage3_retrieval.json`. `compare` warms the installed model with one
neutral question and writes all six runs to
`reports/evaluation/stage31_comparison.json`. Do not rebuild ingestion or vectors
to try these modes. Lexical still requires the valid existing published index,
even though it does not use vectors to rank. An HTTP client is constructed by
the CLI in lexical mode, but no request is sent.

`RETRIEVAL_ALIASES_FILE` optionally selects the reviewable JSON registry; its
default is `data/knowledge_base/metadata/retrieval_aliases.json`. Existing Ollama
and index settings still apply. Environment variables are not loaded from `.env`
automatically. Omit `--question` for an interactive session that reuses BM25.

## Dense versus lexical

Dense retrieval compares learned representations of meaning. It helps when a
question and a passage use different wording. A brand name can still be weakly
represented, spelled differently, or overwhelmed by other words in a long,
mixed-topic passage. The actual evaluation demonstrates such failures even for
English Losec, so multilingual model support alone does not guarantee name lookup.

Lexical retrieval matches terms directly. BM25 rewards matching rare words more
than common words, limits the benefit of repeating a term, and adjusts for chunk
length. It can locate a drug name without understanding the surrounding dialect.
It can also rank an irrelevant long summary containing that name too highly.
Neither method verifies that a passage supports every part of a question.

The implementation uses unique query terms, `k1=1.2`, `b=0.75`, and
`idf=log(1+(N-df+0.5)/(df+0.5))`. Here each chunk is one BM25 document. These
standard choices match the documented
[BM25 formula/defaults](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html);
Lucene is a reference, not a dependency. The derived search view contains source
text plus missing source section/context headings. Filename-derived document
titles are excluded because they can be overbroad. Original chunks are never
rewritten. Zero-overlap chunks are excluded from lexical results.

## Conservative normalization and aliases

Normalization applies NFKC, case folding, whitespace cleanup, Arabic Alef
normalization, Alef Maqsura to Yaa, Arabic diacritic/tatweel removal, decimal digit
normalization, dash normalization, and Arabic/Latin letter boundary separation.
Formatting controls are removed. It preserves negation words, numbers, drug
qualifiers, and the distinction between Ta Marbuta and Haa. Retrieval tokenization
uses alphanumeric words; punctuation is not a BM25 term. This is for searching,
not a clinically equivalent rewrite of the original question.

The registry has 25 confirmed and two ambiguous entries. Each has an independent
ID, canonical source term, variants, status, and review note. Same-name aliases
map spellings such as Arabic Cataflam to `Cataflam`, never to its ingredient.
Canonical targets must occur as bounded terms in the current approved source
text. Missing targets are flagged and left unchanged. Presence verifies vocabulary,
not clinical equivalence; the review note records the intended linguistic identity.
The supplied review is an engineering review, not independent clinical approval.

Longest matching aliases take priority, so `Panadol Extra` retains its qualifier.
An optional Arabic definite article is recognized. Attached prepositions such as
the prefixes in `بالميتفورمين` and `للسالبوتامول` are deliberately not stripped.
Broad pressure-medicine phrases and `mamno3` remain ambiguous and are not expanded.
Franco digits are not globally transliterated: `3` and `7` may be letters in a
word or actual quantities. Only explicit reviewed variants are handled.

Every lexical/hybrid result records the original, normalized, and rewritten
question, applied/ambiguous/unavailable alias IDs, and registry fingerprint.
No question is written to an application log by this module. CLI JSON and
evaluation artifacts contain questions explicitly; production retention policy
remains a separate responsibility.

## Fusion, identity, and efficiency

RRF adds `1/(60+rank)` for each candidate's position in each list. Each branch
supplies up to 20 candidates, or the requested top-k if larger. Missing candidates
contribute zero; equal fused scores sort by chunk ID. Combining positions avoids
pretending that BM25 and cosine scores share a scale. This follows the
[original RRF method](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
The result includes both component ranks/scores. RRF scores are not confidence.

All modes load the explicit approved corpus and validate the published snapshot,
source/build identity, model name, vector checksum, and row alignment. Dense and
hybrid additionally verify live model identity around inference. Lexical makes
no claim about the currently running Ollama model: it uses the stored identity
only as provenance. Hybrid rejects changed chunk/document metadata between branches.
Every returned result retains full source text, spans, source checksum, build ID,
review/correction metadata, and embedding identity for later citation.

BM25 is cached in memory per corpus/registry fingerprint; changing either rebuilds
it. Files and the published index are still validated on each search. At 113
chunks this deliberately favors explainability over a new service or persistent
lexical index. Hybrid does one model inference per question, plus the existing
identity checks. No scheduling or reranker was introduced.

## Future approved uploads and remaining limits

Normalization, aliases, BM25, and RRF do not depend on the nine document IDs.
A future approval adapter can supply a new `ApprovedCorpus` and matching published
index using the existing generic validation interface. The current CLI retains
the explicit approved-nine adapter. Parsing or alias presence cannot approve an
uploaded document. New approved builds invalidate the cache; changed embedding
inputs require the corresponding vector update. Incremental persistence and
upload approval remain future work.

The [measured comparison](retrieval_comparison.md) includes successes, regressions,
negative controls, cold/warm timings, and frozen input hashes. In particular,
normalization plus fusion is evaluated as one hybrid configuration; there is no
ablation proving how much improvement each component causes. No settings were
tuned after the held-out run. Retrieval success establishes evidence discovery,
not medical answerability, patient suitability, completeness, or safety.

The measured hybrid quality supports a controlled Stage 4 development experiment,
but a subsequent live check found zero-valued embeddings from the local Ollama
runtime, also reproduced after one manual runner reload. Resolve that operational
failure and pass the live check before proceeding. The existing client rejects
these vectors rather than serving invalid results. No automatic retry, scheduling,
runtime configuration, or baseline change was added. Retrieval remains insufficient
for unrestricted medical answers. Stage 4 was not implemented.
