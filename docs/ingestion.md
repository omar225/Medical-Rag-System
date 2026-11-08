# Stage 2.1: reusable ingestion and context corrections

Ingestion extracts source information and divides it into traceable chunks. It
does not translate, summarize, validate medical accuracy, grant approval, or
make documents authoritative. This stage uses the existing Python and
`python-docx` implementation and adds no dependencies.

## Two entry points

The reusable operation processes one file and returns ordinary Python records:

```python
from pathlib import Path
from medical_rag.ingestion.processing.pipeline import ingest_document

result = ingest_document(
    Path("candidate.docx"),
    doc_id="candidate-10",
    source_metadata={
        "original_filename": "Original document name.docx",
        "source_reference": "administrator-supplied source reference",
    },
)
```

It accepts optional `title`, `max_chars`, and reviewed `corrections`. It returns
`document`, `chunks`, `source_sha256`, `config_id`, `build_id`, `issues`, and
`requires_review`. It never writes KB files, modifies the source, registers the
document, approves it, or creates vectors. Assigned IDs use safe ASCII letters,
digits, hyphens, and underscores, with a maximum length of 128. IDs are not
derived from filenames. `original_filename` preserves the source name when a
future upload is stored under a generated filename. Supplied titles retain
`title_source: source_metadata` provenance.

The approved-nine bootstrap remains:

```powershell
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
```

`discover_approved_sources()` selects the existing nine ID/filename pairs from
`APPROVED_SOURCES`. `rebuild_knowledge_base()` calls `ingest_document()` once per
selected source, then writes the same output paths. Extra files and KB folders
are ignored. A missing source, stale correction checksum, or extraction error
stops the bootstrap before outputs are replaced. Review issues remain visible
in the artifacts; the bootstrap does not resolve approval policy.

Processing one document at a time lets future uploads and replacements reuse
exactly the same reader and chunker without rebuilding unrelated documents.
The catalog decides which files are in this bootstrap, not which files the
parser is capable of reading.

## Functions and execution flow

1. `ingest_document()` validates its assigned ID, metadata, and settings, reads
   one immutable byte snapshot, and calculates its source/build identities.
2. `read_docx()` uses `Document(BytesIO(...))`. It reads body elements in source
   order and preserves original 1-based body positions, including gaps for
   ignored empty paragraphs. It never saves the DOCX. `heading_level()` reads
   explicit outline levels, heading styles, and inherited style outlines.
3. `read_table()` retains row/column cell arrays. Leading Word repeated-header
   flags are honored; otherwise the first row is retained as a tentative header.
   `render_table()` joins cells with ` | ` and rows with newlines. Exact cell
   strings remain in processed JSON.
4. `resolve_structure()` builds section paths from observed heading levels and
   optional reviewed rules. It reports uncertainty instead of globally promoting
   numbered level-3 headings. Short colon labels are tentative structural labels;
   lettered paragraphs, short fully bold paragraphs, and a few common structural
   labels can trigger review without automatic promotion.
5. `chunk_document()` packs major sections using the resolved paths. It carries
   heading-only groups into the next content group. Long sections split at
   paragraph or table-row boundaries. `_render()` repeats necessary source
   headings, context topics, and table headers for continuation chunks.
6. The bootstrap writer replaces nine `processed/KB-XX.json` files, nine
   `chunks/KB-XX.json` files, and
   `metadata/generated/ingestion_manifest.json`. The complete source set is
   processed before writing. Each file replacement is atomic; the entire set
   is not a database transaction.

The character target remains **2,400 Unicode characters**. A paragraph, cell, or
row is never cut to meet it. Indivisible oversized content is marked for review.
There is no blanket sliding overlap. Character counts are not token counts;
the exact embedding tokenizer must later check complete inputs and special
tokens before indexing. No tokenizer, embedding, or retrieval integration is
implemented here.

## Reviewed corrections

The bootstrap reads `data/knowledge_base/metadata/structure_corrections.json`.
The generic ingestion function does not automatically load that catalog. Any
caller may supply the same correction format for an unrelated document:

```python
corrections = {
    "source_sha256": "exact SHA-256 of the DOCX bytes",
    "rules": [
        {"position": 1, "heading_level": 1,
         "reason": "Reviewed source topic above its drug classes"}
    ],
}
```

Supported rule fields:

- `heading_level`: effective level 1-9 for an existing source paragraph; its
  original text and `source_heading_level` remain available.
- `parent_position`: an earlier source heading, or `None` to clear inherited
  parent scope. It cannot refer to arbitrary text or a future block.
- `new_section`: start a separate chunk group, including for an unheaded table.
- `topic_positions`: earlier source headings used as parallel context topics,
  rather than pretending one topic is the parent of another.

Rules require existing, unique positions and a nonblank reason. Source checksums
prevent applying old positions to changed documents. Corrections are recorded in
`applied_corrections`, notes, and informational issues. Blocks and chunks retain
`correction_positions`, including inherited corrections. Provenance is tied to
the deterministic build rather than a nondeterministic processing timestamp.

The reviewed bootstrap data contains 60 local corrections across nine source
checksums. The parser and chunker contain no branches for specific KB IDs.

## Corrected source examples

Before, KB-08 block 15 had this chunk section:

```text
2. ARBs – Angiotensin II Receptor Blockers
```

After retaining the original topic at block 1 as a parent:

```text
1. أدوية الضغط – Antihypertensive Drugs > 2. ARBs – Angiotensin II Receptor Blockers
```

The actual drug table at block 19, side effects at block 21, and contraindications
at blocks 22-28 remain together and unchanged. Each chunk also has source,
configuration, and build identifiers, plus the relevant correction positions.

Other corrected examples:

- KB-01: Asthma and COPD SABA/LABA sections retain different topic ancestry.
- KB-02: table 121 becomes its own GERD-scoped chunk, separate from H. pylori
  warnings. The summary at 193-194 retains both GERD and vomiting as parallel
  topics. Shared warnings at 106/114 and 130/134 use their class scopes.
- KB-03: NSAID blocks 16-36 retain Pain & Fever; blocks 132-149 retain Headache &
  Migraine. Aspirin and Topical Salicylates labels retain separate subscopes.
- KB-04: shared antibiotic warnings at 67/75 no longer inherit Lincosamides.
- KB-06: route-specific effects remain beneath their source effect labels.
- KB-07: Diabetes remains the parent of all classes, including Biguanides;
  lettered insulin subtype labels are preserved structurally.
- KB-08: Antihypertensive and Dyslipidemia ancestry is retained; Fibrates remains
  a sibling class, and its original numbering is not rewritten.
- KB-09: the summary at 223-224 has only its actual summary heading as its
  section, with IBS and Arthritis as parallel topics. No Arthritis-only prefix
  is used for this mixed table.

No original medical paragraph, cell value, drug name, warning, or qualifier is
changed. These are structural corrections, not a medical accuracy review.

## Identity, issues, and future uploads

`source_sha256` identifies the exact source bytes. `config_id` hashes the explicit
pipeline version, installed python-docx version, and chunk-size setting.
`build_id` hashes the document ID, source checksum, configuration, relevant source
metadata, original filename, and corrections. Changed source, settings, or
corrections produce different build IDs. Identical inputs/settings produce the
same records and IDs. The pipeline version must be bumped when parsing or
chunking semantics change.

Canonical public-ingestion chunk IDs are
`<doc_id>-<full-build-sha256>-<ordinal>`. They are stable within a build, not
persistent identities across arbitrary edits. Low-level reader/chunker calls
without build metadata retain their old ordinal format for compatibility.
Original source spans, inclusive table row ranges, checksums, topic paths, and
correction provenance support later citation verification. Public MTI citations
still have exactly `doc_id`, `title`, `section`, and `snippet`.

Issue levels are:

- `info`: first-row header inference and recorded reviewed corrections.
- `review`: missing/ambiguous headings, uncertain labels, blank headers or data
  values, merged/irregular grids, extra text outside supported body parts, and
  indivisible oversized chunks.
- `error`: nested tables, unsupported body structures, or invalid DOCX packages.
  The public function returns no partial document/chunks for extraction errors.

`requires_review` means there is at least one review/error issue. A false value
does not mean medically approved, authoritative, or indexed. Missing section
headings remain unresolved (`section: ""`, `scope_uncertain: true`) rather than
using a filename as a fabricated source section. Some ambiguity cannot be
detected automatically; unfamiliar documents still need human inspection.

Future upload code must store the source, assign an ID, call this function, and
present the artifacts/issues for review. Medical approval, immutable artifact
storage, replacement/deletion operations, active-version selection, vector
index updates, and retrieval eligibility checks are still future work. A new
source version can reuse the document ID while producing a new build. No full
corpus rebuild is required by the ingestion function.

## Default bootstrap and verification

| KB ID | Source tables | Chunks |
| --- | ---: | ---: |
| KB-01 | 19 | 20 |
| KB-02 | 19 | 13 |
| KB-03 | 15 | 12 |
| KB-04 | 9 | 6 |
| KB-05 | 8 | 9 |
| KB-06 | 16 | 16 |
| KB-07 | 22 | 9 |
| KB-08 | 30 | 14 |
| KB-09 | 15 | 14 |
| Total | 153 | 113 |

The extra chunk isolates KB-02's broad classification table. Chunk count is not
a fixed invariant. There are 213 informational issues (153 inferred headers and
60 corrections) and one review issue: KB-04 table 83, row 7, column 2 has a blank
classification value for `Strepsils Plus / Orofar` in the original source. The
blank is preserved; no value is invented. All original 1,566 nonempty body
paragraphs and 1,262 data rows remain covered.

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
```

Tests cover an independent synthetic tenth document with an arbitrary ID, no KB
writes or corpus registration, multilingual text, source/row/header coverage,
known corrected contexts, deterministic builds, changed-source/settings/review
identities, stale corrections, invalid IDs/metadata, and unfamiliar/unsupported
structures. Synthetic fixtures stay in pytest temporary directories and never
join the approved source catalog. The existing FastAPI tests remain in the suite.

Verified on 2026-11-08: **103 tests passed in 8.18 seconds**, with one existing
Starlette TestClient/HTTPX deprecation warning. An initial run found a namespace
error in handling an unfamiliar XML container; that error was fixed before the
passing run. The normal rebuild produced the counts above. Independent raw-DOCX
XML comparison verified every retained paragraph, table cell, data row, and
header association. All nine original DOCX files match Git HEAD bytes. Repeated
ingestion/build determinism is covered by the tests. `pip check` and
`git diff --check` also passed.
