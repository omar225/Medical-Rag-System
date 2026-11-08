# Fresh Windows machine setup

Run commands in PowerShell, from the cloned repository unless specified. Paths
are repository-relative; the original drive, user account, virtual environment,
and GPU are not required. Stage 3.1 is complete; Stage 4 is not implemented.
The existing zero-vector Ollama issue is unresolved. Do not rebuild embeddings
until the direct diagnostic below succeeds on the new GPU.

## Install prerequisites manually

Use Windows x64 and Python 3.12 x64. The recorded environment used Python 3.12.10;
use that patch for the closest reproduction. Other patch releases need their own
test run. No system software or model is installed by our helper script.

```powershell
# Explicit user actions, not commands run automatically by this project:
winget install --id Git.Git -e --source winget
Start-Process 'https://www.python.org/downloads/release/python-31210/'
Start-Process 'https://ollama.com/download/windows'
Start-Process 'https://www.nvidia.com/Download/index.aspx'
# Choose the Python x64 installer, install the Python launcher, and install
# Ollama and the appropriate supported NVIDIA driver for the new GPU.
# Reopen PowerShell after installation (restart Windows if the driver requires it).
git --version
py -3.12 --version
ollama --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Use [Git's Windows instructions](https://git-scm.com/install/windows), the
[Python 3.12.10 release](https://www.python.org/downloads/release/python-31210/),
[Ollama Windows requirements](https://docs.ollama.com/windows), and
[Ollama GPU compatibility](https://docs.ollama.com/gpu) when choosing a driver.
Do not assume the original driver's version is appropriate for the new GPU.
The project has no PyTorch/CUDA Python dependency. GPU execution is owned by
the separately installed Ollama runtime. Docker files remain placeholders.

## Clone the approved private repository

The remote URL is intentionally not invented; publication awaits approval.
After publication, substitute the confirmed private URL when prompted.

```powershell
$RepoUrl = Read-Host 'Confirmed private GitHub repository HTTPS URL'
git clone --branch master $RepoUrl medical-rag
Set-Location -LiteralPath .\medical-rag
git status --short --branch
git log -1 --oneline
```

Authenticate through Git Credential Manager's browser prompt, or optionally use
GitHub CLI (`winget install --id GitHub.cli -e`) followed by
`gh auth login --hostname github.com --git-protocol https --web`.
Never paste a token into a repository URL or commit credentials. After publication,
compare the checked-out SHA with the handoff report's confirmed remote SHA.

## Create Python environment and install exact dependencies

```powershell
# Optional prerequisite check:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\maintenance\setup_windows.ps1 -CheckOnly
# Optional simple setup helper (creates .venv and installs requirements only):
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\maintenance\setup_windows.ps1
```

Equivalent manual commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
.\.venv\Scripts\python.exe -m pip check
```

Use a new environment instead of copying the original `.venv`. No activation is
needed. Do not silently loosen pinned versions or upgrade packages if installation
fails. Keep the error and review availability/platform compatibility first.

## Configuration

```powershell
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:EMBEDDING_MODEL = 'bge-m3'
$env:OLLAMA_TIMEOUT_SECONDS = '180'
$env:EMBEDDING_BATCH_SIZE = '8'
$env:LLM_MODEL = 'qwen2.5:7b'
Copy-Item -LiteralPath .env.example -Destination .env
```

`LLM_MODEL` is currently a health-response identity only. Qwen is a future Stage 4
candidate, not an integrated generation model; **no Qwen download is required**.
BGE-M3 is the only integrated model required for dense/hybrid retrieval.
Index and alias paths default relative to the repository. Optional overrides are
`RETRIEVAL_INDEX_DIR` and `RETRIEVAL_ALIASES_FILE`; leave them unset for recovery.
`RUN_OLLAMA_TESTS=1` is optional and only enables the separate live test.
There are no required application secrets. Maintenance scripts do not load `.env`;
the session variables above work for them. Uvicorn can explicitly load `.env`.
None of these session variables configures an already running Ollama server.

## Restore the exact stopped state

The nine DOCX originals must be present at their approved KB paths. If upload
permission is denied, obtain them through an authorized secure transfer; a
history-preserving GitHub push is currently blocked because they already exist
in the two local commits. Do not substitute modified source files.

The reviewed recovery snapshot restores generated JSON, the last valid dense
index, and original experimental reports, without any inference:

```powershell
$Snapshot = Get-Content .\reproducibility\snapshot_manifest.json -Raw | ConvertFrom-Json
$ActualHash = (Get-FileHash .\reproducibility\stage31_recovery.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $Snapshot.archive_sha256) { throw 'Snapshot checksum mismatch' }
# Use in a fresh clone; no -Force, so later experiment files are not overwritten.
Expand-Archive -LiteralPath .\reproducibility\stage31_recovery.zip -DestinationPath .
foreach ($Entry in $Snapshot.entries) {
    $Hash = (Get-FileHash -LiteralPath $Entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Hash -ne $Entry.sha256) { throw "Snapshot entry mismatch: $($Entry.path)" }
}
```

This is the quickest way to resume the recorded corpus/index, and enables lexical
retrieval even if the new runtime also has the zero-vector problem. It contains
medical excerpts and requires the same explicit private-upload approval.

## Download and verify BGE-M3, then test it before rebuilding

These are explicit manual model-download commands. Never store weights in Git.
The [official BGE-M3 listing](https://ollama.com/library/bge-m3) reports roughly
1.2 GB for the current `latest` tag; actual transfer and runtime storage can differ.
Tags may change, so compare the full digest, not just the tag.

```powershell
ollama pull bge-m3:latest
ollama list
Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/version"
$Tags = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags"
$Model = $Tags.models | Where-Object { $_.name -eq 'bge-m3:latest' }
$Model | Select-Object name,digest,size
$RecordedDigest = '7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab'
if ($Model.digest -ne $RecordedDigest) { Write-Warning 'Different model digest: do not claim exact baseline reproduction or use the old index for dense/hybrid.' }
.\.venv\Scripts\python.exe scripts/maintenance/probe_embeddings.py --output reports/evaluation/new_machine_probe.json
if ($LASTEXITCODE -ne 0) { throw 'Resolve Ollama embeddings before any index build' }
ollama ps
```

Start the installed Ollama application if the endpoint is unavailable. Use
`ollama serve` only if there is no existing server; do not start a second server
on the same port. Record the new GPU, driver, Ollama version and digest. A different
digest requires a fresh index with that model and a new evaluation, even if the
tag and dimensions match. A successful small probe does not guarantee that all
source chunks fit the runtime context; builds retain `truncate: false` and must
fail rather than truncate. See [zero-vector troubleshooting](OLLAMA_ZERO_VECTOR_ISSUE.md).

## Run offline and optional live tests

Restore or rebuild ingestion outputs before the full suite: bootstrap tests read
the generated corpus, so source files alone are insufficient for those tests.

```powershell
Remove-Item Env:RUN_OLLAMA_TESTS -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --tb=short
# Recorded baseline: 213 passed, one optional live test skipped.
$env:RUN_OLLAMA_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_ollama_embeddings.py -p no:cacheprovider --tb=short
Remove-Item Env:RUN_OLLAMA_TESTS
```

The optional live check failed on the original machine after earlier successful
experiments. Do not accept zero vectors to make it pass.

## Regenerate artifacts when needed

Skip regeneration if resuming the restored snapshot. If deliberately rebuilding,
preserve old reports, validate the model first, then run:

```powershell
.\.venv\Scripts\python.exe scripts/ingestion/rebuild.py
if ($LASTEXITCODE -ne 0) { throw 'Ingestion failed' }
# Expected: nine documents, 153 tables, 113 chunks; KB-04 review flag remains.
.\.venv\Scripts\python.exe scripts/maintenance/probe_embeddings.py
if ($LASTEXITCODE -ne 0) { throw 'Embedding probe failed; do not build' }
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py build
if ($LASTEXITCODE -ne 0) { throw 'Index build failed' }
```

Ingestion uses pinned python-docx, original checksums, and reviewed corrections to
recreate build-specific IDs. It does not change DOCX originals. The vector builder
validates all embeddings and publishes the index last. Different GPUs/runtimes may
produce numerical differences; keep the old snapshot and label new measurements.
No upload, replacement, database or incremental-index workflow is implemented.

## Search, evaluate and start the API

```powershell
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode lexical --question 'Cataflam byetkawen men eh?' --top-k 5
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode dense --question 'What is the active ingredient in Cataflam?' --top-k 5
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py search --mode hybrid --question 'Cataflam byetkawen men eh?' --top-k 5
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate --mode dense --output reports/evaluation/new_machine_dense.json
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate --mode lexical --output reports/evaluation/new_machine_lexical.json
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate --mode hybrid --output reports/evaluation/new_machine_hybrid.json
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py compare --output reports/evaluation/new_machine_comparison.json
.\.venv\Scripts\python.exe -m uvicorn medical_rag.api.main:app --app-dir src --env-file .env --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
Start-Process 'http://127.0.0.1:8000/docs'
Invoke-RestMethod 'http://127.0.0.1:8000/health'
```

`/health` reports configured identities and API liveness, not model readiness.
A schema-valid `POST /chat` intentionally returns HTTP 500 with
`{"error":"Chat pipeline is not implemented yet"}`. There is no generation,
frontend, persistence, or safety classifier yet.

The [handoff review](GITHUB_HANDOFF_REVIEW.md) records publication status,
selected/excluded assets and actual isolated recovery results. The
[retrieval comparison](retrieval_comparison.md) preserves the experimental results;
new hardware measurements must be reported separately.
