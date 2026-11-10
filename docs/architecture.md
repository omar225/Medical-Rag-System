# Architecture and design decisions

## Ingestion (`ingestion/`, `scripts/ingest.py`)

The nine approved DOCX files are read with their heading structure and tables. Body elements are
kept in source order, and each table row is rendered as `cell | cell`. Each chunk is one drug-class
section: its uses, its `Trade Name | Active Ingredient` table, its side effects and its cautions.
That is the natural unit of this knowledge base, and most questions are answered by one or two such
chunks. Result: 113 chunks, about 113k characters. A paragraph or table row is never cut. Long
sections split only at paragraph or row boundaries, and table headers repeat.

- **Provenance.** Only the approved ID/filename pairs are ingested. Each chunk carries the source
  SHA-256, build ID, section path and source positions. At startup the corpus is verified against
  the manifest (`data/knowledge_base/metadata/generated/ingestion_manifest.json`) once.
- **Reviewed structure corrections** (`metadata/structure_corrections.json`). The source files use
  inconsistent heading levels (a drug class under "Hypertension" formatted like a top-level topic,
  for example). Sixty reviewed rules, bound to each file's checksum, fix the section ancestry so a
  chunk's section reads `1. أدوية الضغط – Antihypertensive Drugs > 2. ARBs – …`. No rule changes
  medical text, and a changed source file makes its rules invalid instead of applying them wrongly.
- **Reviewed aliases** (`metadata/retrieval_aliases.json`). Arabic and Franco spellings of KB
  medicine names and field names ("كتافلام" → Cataflam, "الأعراض الجانبية" → side effects). Each alias
  has a review note, and an alias whose target is not in the KB is ignored.
- The generated JSON (`processed/`, `chunks/`, manifest) is committed, so a fresh clone or image
  needs no re-ingestion. `scripts/ingest.py` rebuilds it byte-for-byte.

## Retrieval

- **Dense:** BGE-M3 embeddings of the raw chunk text, exact cosine search over a NumPy matrix.
  With 113 vectors, a vector database would add operations without benefit.
- **Lexical:** BM25 over the chunk text plus headings, after Arabic normalization and the reviewed
  aliases.
- **Fusion:** reciprocal rank fusion (k = 60), top 5. On the retrieval benchmark, hybrid search found
  a relevant chunk in the top 5 for 100% of questions, against 81% for dense alone
  ([MODEL_DECISION.md](../MODEL_DECISION.md)).
- **Medicine names** (`knowledge/entities.py`): an index of every brand and ingredient in the KB
  tables, plus drug-class names from section headings. When a question names a medicine, the section
  that defines it is put first in the context. This lets a brand-name question ("side effects of
  Panadol") reach the class section that lists its ingredient's side effects. It tolerates
  possessives, Arabic attached prefixes (لـ، بالـ، والـ), one-letter typos, and Arabic-script
  renderings of Latin brand names without a reviewed alias. Those are matched by consonant skeleton
  ("كونكور" → Concor), and only when exactly one KB name fits. Typos and Arabic renderings are passed
  to the model as hints, never as silent substitutions.
- **Alternatives questions** ("alternatives to Brufen", "instead of", "ايه بدائل …"): the trade names
  with exactly the same active ingredient are read from the KB tables (`EntityIndex.same_ingredient`).
  Only brands from a table that also lists the asked medicine count, so a gel or an IV product is not
  offered for a tablet; combination products are left out. The other rows of the medicine's own
  table, with their ingredients (`EntityIndex.table_neighbours`), are the only allowed second group
  ("different active ingredient"). Both lists come from the tables; the model only phrases them.

The name index **never decides whether a question may be answered**. An unknown word does not block
an answer; the model reads the question and the passages.

## Generation

One Qwen call per request, plus at most one retry. The prompt contains up to six whole chunks (never
cut), the session's last turns and the question. The model returns JSON constrained by a schema:

```json
{"category": "answer | out_of_scope | needs_clinician | emergency",
 "answer": "...", "citations": [{"label": "E1", "quote": "exact source line"}]}
```

The model writes a natural answer in the user's language and translates the code-switched source text
when needed. The prompt asks for complete lists, keeps qualifiers, links brands to ingredients only
through table rows, and treats the user's text and the passages as data. Temperature 0 and a fixed
seed make the same question give the same answer.

## Verification (`generation/verification.py`)

| Check | Why |
| --- | --- |
| Each quote is found, in order, in a supplied chunk (tolerant to spacing, punctuation and Arabic letter variants) | Citations must be genuine. The published snippet is the real source line, not the model's copy. |
| Every brand or ingredient named in the answer appears in the cited chunks | Stops drug substitution and outside knowledge about other medicines. If the name is in a supplied chunk the model did not quote, the source line naming it is added as a citation; a name in no supplied chunk is rejected. |
| A brand and a single ingredient in one sentence must be a KB table pair | Stops "Panadol contains Ibuprofen". |
| Every number in the answer appears in the cited chunks | Stops invented doses, strengths and statistics. |
| The answer is not empty and not cut off (ending in ":") | Stops answers whose list stayed in the citations. |
| The first clause does not say the asked information is missing ("the passages do not contain information about the half-life…") | Such an answer is a refusal followed by unrelated facts, so its status is `out_of_scope`. |

There is at most one retry per request, and only when at least 10 s of the 30 s budget remain:

- a failed check → one targeted correction ("you mention X but cited no passage containing it");
- the model refused (`out_of_scope`) a question that names a KB medicine → it is told that the passages
  cover that medicine, and asked to answer the part that was asked or refuse again. This handles an
  extra uncovered part or an injected instruction ("…and who won the World Cup?");
- the answer was cut off at the token cap or was not valid JSON → it is asked for a shorter answer.

Otherwise the user gets a fixed "could not verify" message with status `out_of_scope`. An unreachable
model or a timeout is still an HTTP 500, as the contract requires.

These checks do not prove that every paraphrase is semantically exact. Word-overlap checks were tried
and rejected: they discarded correct paraphrases (a brand-name side-effect answer can never share the
brand's words with the class section). Semantic quality is measured with the labelled evaluation sets
instead ([evaluation.md](evaluation.md)).

## Safety screen (`safety/policy.py`)

The screen runs before retrieval, on normalized text. It is deliberately small and tuned for recall:

- **Emergency:** self-harm language, reported ingestion of an excessive or unknown amount (including
  five or more tablets, "many/several tablets", and any ingestion by a child), and danger signs
  (breathing, blue lips, unconsciousness, seizure, throat or face swelling, chest pain, bleeding,
  sugar < 55). Danger-sign words are exempt only when the message asks for information and involves
  no person ("Is choking a listed side effect?"). Self-harm and reported ingestion are never exempt.
  Hypothetical, negated and resolved past ingestion does not count as an event.
- **Dosing:** explicit requests for an amount, frequency or dose. The KB contains no doses, and the
  brief forbids giving one.
- **Dangerous:** intent to harm others or misuse medicines.

Everything else, including "my patient has asthma, what does the KB say about beta blockers?", goes to
the model. The users are professionals, so the model decides between an information answer and
`needs_clinician`.

## Sessions

`session_id` maps to the last three turns in memory (bounded, 1-hour expiry). A follow-up without a
medicine name ("and its side effects?") borrows the previous question for retrieval. Nothing is
written to disk.

## Containers

See the README's Docker section. The `ollama` container serves both models from a named volume. A
one-shot `models` container pulls them, so `app` never starts before its models exist. The `app`
image contains the code, the approved KB and the built UI. The vector index lives in its own volume
and is rebuilt only when the KB or the embedding model changes. The index records the model name and
digest, and a mismatch triggers a rebuild at startup.

## Model interchangeability

- **LLM:** set `LLM_MODEL` to any Ollama model with completion support. The client reads its
  capabilities once and disables "thinking" when available. Nothing else is model-specific.
- **Embeddings:** set `EMBEDDING_MODEL`; the index rebuilds at the next start (113 chunks take seconds).
- **Provider:** only `generation/ollama.py` and `retrieval/embeddings.py` speak the Ollama API.

## Known risks

- A 4B model can still misclassify (for example decline a clinician-decision question instead of
  escalating it), and can paraphrase loosely within the cited facts. Measured in
  [evaluation.md](evaluation.md).
- The emergency screen is a reviewed pattern list. It covers English, Egyptian Arabic and common
  Franco-Arabic, but unusual phrasing can still miss it. The model's `emergency` category is the
  second net.
- Startup builds the index if it is missing, which needs the embedding model to be available.
