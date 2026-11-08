# Knowledge-Base Metadata

`generated/ingestion_manifest.json` is rebuilt by:

```powershell
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
```

The schema-version-2 manifest records source checksums, title provenance, raw and
resolved heading counts, table/chunk counts, configuration and build IDs, applied
corrections, structural issues, and `requires_review`. It has no timestamp, so
unchanged inputs and settings produce identical output bytes. `source_path` is
relative to `data/knowledge_base/source/`.

`structure_corrections.json` is tracked review data for the bootstrap. Each entry
contains an exact source SHA-256 and rules referring to original body positions.
Rules may adjust a heading level, select an existing parent, start a new section,
or reference existing headings as parallel context topics. Every rule has a
reason. No rule contains replacement medical text or a fabricated heading.
The bootstrap selects the entry by its catalog ID, while the reusable resolver
only sees the checksum and rules. A changed source checksum is rejected rather
than applying stale positions.

Issue levels are `info`, `review`, and `error`. Header inference alone is
informational. A blank source classification value at KB-04 block 83 remains
flagged for review. `requires_review: false` is not medical approval; no approval
state is created by ingestion.

Generated metadata is ignored by Git under the existing rules. Source identity
is defined by `APPROVED_SOURCES` in the DOCX reader; extra documents are ignored,
and a missing or renamed approved file is an error. See the
[ingestion documentation](../../../docs/ingestion.md) for formats and limitations.
