# Private GitHub handoff: approval review

Historical preparation record: the review below predates publication. The project
is now in the private `omar225/Medical-Rag-System` repository on `master`.
The 19 Stage 2.1 ingestion JSON files are now directly tracked; the original
recovery archive remains unchanged. Use [NEW_MACHINE_SETUP.md](NEW_MACHINE_SETUP.md)
for current clone/setup instructions. The original preparation findings below are
retained as a record, not current publication status.

## Publication state

**Prepared locally; not staged, committed, created on GitHub, or pushed.**
Current branch: `master`; current HEAD: `9fb62d7` (parent `51ccddc`). No remote
exists. GitHub CLI is not installed. Existing Stage 3/3.1 changes remain preserved
in the working tree. No history reset, discard, branch rename or force push occurred.

Proposed private repository: **`mti-medical-rag`**, under the account selected by
interactive GitHub authentication. Proposed commit message:
`Preserve Stage 3.1 retrieval and add portable development handoff`.
Repository URL, final commit SHA, verified remote branch, and visibility remain
pending explicit approval and publication; they must not be inferred from this name.

## Required asset permission

All nine original DOCX files are **already present in the initial Git commit**.
Any push that preserves this history uploads them, even if they are untracked in
a later commit. Do not try to bypass approval with `.gitignore` or a deletion.
No confidentiality/restriction label was found in the checked document XML text
and metadata. This does not grant redistribution rights. The clinical word
"assessment" occurs in KB-02 and is not itself a confidentiality label.

No assessment PDF was found in the workspace or reachable Git history. It was
therefore not inspected and is not in the proposed file list. Its confidentiality
status is unknown; keep it out of GitHub and transfer it separately only through
an authorized secure channel. Its exact filename/location is unavailable here.

**Explicit approval is required for the nine DOCX originals and derived medical
excerpts** in tests, documentation, corrections/aliases, and the recovery archive.
If permission is denied or uncertain, do not publish this existing history. Keep
the local project intact and securely transfer the source assets listed below;
a different history strategy would require separate authorization.

## Proposed contents and exclusions

The exact proposed tracked-file inventory is [HANDOFF_FILE_LIST.txt](HANDOFF_FILE_LIST.txt).
It includes existing tracked files plus all selected untracked implementation work,
not just files modified for this handoff. **No new commit has been made yet.**

| Asset | Proposed treatment | Reason / recovery |
| --- | --- | --- |
| `src/medical_rag/`, all tests and scripts | Commit all | Preserves application, API contract, ingestion, retrieval and tests |
| Both frozen and held-out fixtures, aliases, structural corrections | Commit | Preserves labels, review decisions and context/build provenance |
| Nine original DOCX files | Already tracked; upload only after explicit permission | Byte-exact originals are required for checksum-bound rebuilds |
| Requirements, pytest config, safe environment example, docs/model decision/token report | Commit | Recreates configuration and preserves decisions/limitations |
| `reproducibility/stage31_recovery.zip` plus entry manifest | Commit after medical-content permission | Deliberately selected 1.67 MB snapshot: 19 ingestion JSON files, active index JSON and one vector file, four exact evaluation/probe reports |
| Live `processed/`, `chunks/`, generated metadata, `storage/vector_store/`, raw evaluation directories | Leave ignored | Restore exact reviewed snapshot or regenerate with approved sources/pinned tools; no progress is lost |
| `.venv`, caches, `.handoff-verification/` | Exclude | Machine-specific; recreate environment. Verification copy is local only |
| `.env`, tokens, credentials, private keys, session databases, analytics, logs | Exclude | Not required to reproduce code; provision locally if later needed |
| Model weights, model caches, inactive runtime vector files | Exclude | Explicitly pull BGE-M3 outside Git; only active numerical snapshot is preserved |
| Assessment PDF | Not found; not selected | Needs separate location/confidentiality review and permission or secure transfer |
| Legacy file named `gitignore`, Docker placeholders, empty boundaries | Retain existing history | Actual Git rules come from `.gitignore`; Docker/frontend are not implemented |

Original source paths requiring private-upload approval or secure transfer:

- `data/knowledge_base/source/KB-01/asthma w copd.docx`
- `data/knowledge_base/source/KB-02/gerd& vomiting.docx`
- `data/knowledge_base/source/KB-03/pain headache ibs arthritis.docx`
- `data/knowledge_base/source/KB-04/pharyngitis.docx`
- `data/knowledge_base/source/KB-05/أدوية الالتهاب الرئوي.docx`
- `data/knowledge_base/source/KB-06/ادويه الحساسيه و الجيوب الانفيه.docx`
- `data/knowledge_base/source/KB-07/ادويه السكر.docx`
- `data/knowledge_base/source/KB-08/ادويه الضغط و الكوليسترول.docx`
- `data/knowledge_base/source/KB-09/ادويه القولون و التهابات المفاصل.docx`

The archive is a selected backup, not permission to index arbitrary documents.
Restored/rebuilt artifacts must still pass the existing explicit approval,
source checksum, build, corpus fingerprint, vector checksum and model identity gates.

## Security and portability review

- Reviewed proposed files plus archive entries, DOCX XML, and reachable historical
  blobs for common GitHub/OpenAI/AWS token formats, private-key headers, and
  credential-bearing URLs. No matches were found. This is a bounded pattern scan,
  not a guarantee that every possible secret format is absent.
- No files were staged. Repeat the scan against the actual staged bytes before
  committing after approval, and confirm the staged paths match the approved list.
- No proposed file exceeds 10 MB. The largest is the 1.67 MB selected recovery
  archive; raw 12.6 MB comparison output remains ignored and is compressed there.
- No required original-drive, user-profile or local model-storage path was found
  in application configuration, copied reports, or recovery data. Source paths and
  defaults resolve from the repository. A `C:\\path` test string is an intentional
  invalid-document-ID test, not a dependency. Standard Windows environment paths
  in troubleshooting are portable OS lookups.
- `.gitattributes` preserves DOCX/archive bytes and LF text checkout behavior.
  `.gitignore` adds local verification, credentials/private keys, numerical runtime
  files and GGUF weights; it cannot retroactively exclude tracked history.
- All dependency pins remain unchanged by this handoff. Qwen is only a future
  generation candidate and health identity; BGE-M3 is the integrated encoder.
- No chunking/ranking algorithms, model client, evidence labels, API schemas,
  clinical source content, or zero-vector validation were changed.

## Actual isolated recovery verification

The check used an isolated export of the proposed files, a **new Python 3.12.10
virtual environment**, and the existing machine's OS/GPU. It was not a GitHub
clone or a separate computer. No proposed-state commit exists yet to clone.

- Installed unchanged `requirements.txt` successfully; cached wheels were used.
  `pip check` passed. This is not an uncached internet/package-availability test.
- Restored the archive with the documented PowerShell `Expand-Archive` command.
- Rebuilt ingestion in that copy: nine documents, 153 tables, 113 chunks;
  213 informational issues, one existing KB-04 review issue, no structural errors.
- Every one of the 25 restored file hashes matched the original snapshot after
  rebuilding, including the ingestion JSON. Corpus fingerprint remained
  `3e4bb5da73e9b335e1e1bd15ed1bbfe971e262870d23336e0fd74aba8fa68888`.
- Complete offline suite: **213 passed, one optional live test skipped**, one
  existing Starlette/HTTPX warning, in 10.43 seconds. Existing API tests passed.
- Restored-index lexical lookup succeeded without Ollama inference.
- Setup helper prerequisite check and diagnostic help command ran successfully.
- The new live diagnostic reproduced the unresolved failure: dimension 1,024,
  nonzero count zero, norm zero, for single and English/Arabic batch requests.
  It exited with failure as intended. No new vector index was built.

Not tested here: a different GPU/driver, a true fresh Windows installation,
uncached package/model downloads, the optional CPU experiment, healthy live
dense/hybrid retrieval, or remote private-repository recovery. The new-machine
guide provides those commands. Do not claim that this verification fixed Ollama.

## Publication procedure after approval

The user request explicitly requires approval of this file list and confidential
assets before staging/committing/pushing. This is the only publication gate here;
no skill or additional policy approval is being inferred.

1. Confirm asset permission, exact proposed paths, repository name/account, and
   commit message. Install GitHub CLI manually if desired, then authenticate in
   the browser with `gh auth login --hostname github.com --git-protocol https --web`.
   Do not print or store a token in project files.
2. Recheck branch/status/remotes and the file inventory. Stage only approved paths,
   scan actual staged contents (including archive contents), review diff/large files,
   then create the approved commit on `master` using the existing history.
3. Recheck the proposed destination. Create it empty and **private** using
   `gh repo create mti-medical-rag --private --source=. --remote=origin`.
   Do not use `--push` before verifying `gh repo view --json isPrivate,url`.
   If a destination already exists, inspect its visibility and refs first; do not
   overwrite unrelated history or automatically change an existing remote.
4. Push normally with `git push -u origin master`; never force push. Compare
   `git rev-parse HEAD` with `git ls-remote origin refs/heads/master`, verify
   `isPrivate` again, then record URL, SHA and working-tree status.
5. Clone the published commit into a new local destination and verify file
   inventory/snapshot checksums. Document any residual differences before calling
   publication complete.

The private creation/verification flow follows the
[GitHub CLI repository documentation](https://cli.github.com/manual/gh_repo_create).
No GitHub authentication, system installation or remote mutation occurred during
this preparation. All current implementation progress remains local until approval.

## First task after migration

Follow [NEW_MACHINE_SETUP.md](NEW_MACHINE_SETUP.md), then run:

```powershell
.\.venv\Scripts\python.exe scripts/maintenance/probe_embeddings.py --output reports/evaluation/new_gpu_probe.json
```

Only after valid embeddings and a passing live check should you rebuild vectors
or proceed toward Stage 4. See [the preserved issue](OLLAMA_ZERO_VECTOR_ISSUE.md).
