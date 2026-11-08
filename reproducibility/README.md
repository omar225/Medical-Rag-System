# Reviewed recovery snapshot

`stage31_recovery.zip` is a deliberately selected archive of the original
Stage 2.1 generated JSON, the active Stage 3 dense index (manifest plus its one
referenced numerical file), and all four Stage 3/3.1 evaluation/probe reports.
`snapshot_manifest.json` records every entry's byte length and SHA-256 and the
archive checksum. These files are unchanged byte copies, not a new ingestion,
embedding, or evaluation run. No model weights, DOCX originals, logs, secrets,
virtual environment, inactive vector files, or machine paths are included.

Keeping this compact snapshot is justified because the unresolved Ollama
zero-vector issue may prevent rebuilding vectors immediately on the new machine.
Lexical retrieval also requires the published index as its provenance gate.
The snapshot preserves the last successfully built index and exact experimental
reports, rather than relying on a later model/runtime to reproduce numerical rows.
It contains medical source excerpts and is part of the assets awaiting explicit
private-upload permission. It is not a public distribution artifact.

From a fresh checkout, before running ingestion/index builds:

```powershell
Get-FileHash .\reproducibility\stage31_recovery.zip -Algorithm SHA256
Get-Content .\reproducibility\snapshot_manifest.json -TotalCount 8
# Compare the archive hash above with archive_sha256 in the manifest.
Expand-Archive -LiteralPath .\reproducibility\stage31_recovery.zip -DestinationPath .
```

Do not use `-Force` to overwrite later experiments. Restore into a fresh checkout
or review destination files first. The nine source DOCX files, corrections,
aliases, dependencies, and code must also match. Normal retriever validation
still verifies corpus/build, model identity, numerical checksum, and row alignment.
Restoring a snapshot does not approve new documents or fix zero-valued query
embeddings. Dense/hybrid require a healthy compatible Ollama runtime; lexical
requires no model calls. See [fresh-machine setup](../docs/NEW_MACHINE_SETUP.md).
