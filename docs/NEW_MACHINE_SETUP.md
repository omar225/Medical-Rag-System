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

The existing repository is private. Use the account granted access to it.

```powershell
git clone --branch master https://github.com/omar225/Medical-Rag-System.git medical-rag
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

## Ingestion artifacts are included after cloning

The nine DOCX originals, `chunks/KB-01.json` through `KB-09.json`,
`processed/KB-01.json` through `KB-09.json`, and
`metadata/generated/ingestion_manifest.json` are directly tracked under
`data/knowledge_base/`. **Do not rebuild ingestion after a normal clone.**
They preserve the current Stage 2.1 build: nine documents, 153 tables, and
113 chunks, with source checksums, build IDs, and reviewed corrections intact.
`structure_corrections.json`, `retrieval_aliases.json`, and ingestion code are
also tracked. Leave temporary/candidate JSON and unrelated generated files ignored.

The immutable recovery snapshot also contains these exact ingestion files for
historical recovery. Their compressed copies occupy about 151 KB; it is retained
unchanged to preserve its original checksum and experimental evidence. The live
vector-store and reports directories remain ignored. If you want to resume
retrieval using the last valid dense index without rebuilding embeddings, restore
only the missing index/report files below. Existing files are checked, not overwritten:

```powershell
$Snapshot = Get-Content .\reproducibility\snapshot_manifest.json -Raw | ConvertFrom-Json
$ActualHash = (Get-FileHash .\reproducibility\stage31_recovery.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $Snapshot.archive_sha256) { throw 'Snapshot checksum mismatch' }
@'
import hashlib, json, zipfile
from pathlib import Path
root = Path.cwd().resolve()
manifest = json.loads(Path('reproducibility/snapshot_manifest.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(manifest['archive']) as archive:
    pending = []
    for entry in manifest['entries']:
        target = root / entry['path']
        if not target.resolve().is_relative_to(root):
            raise SystemExit('Unsafe archive destination')
        data = archive.read(entry['path'])
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise SystemExit('Archive entry checksum mismatch: ' + entry['path'])
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
                raise SystemExit('Existing file differs; review before restoring: ' + entry['path'])
        else:
            pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
print('Verified existing files; restored', len(pending), 'missing snapshot files.')
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Snapshot recovery failed' }
```

No ingestion regeneration is needed. Optional snapshot restoration enables lexical
retrieval even if the new runtime also has the zero-vector problem. Dense/hybrid
still require valid query embeddings from a compatible model/runtime.

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

The directly tracked ingestion outputs are available immediately after cloning;
no restoration, ingestion rebuild, model download, or model server is needed for
the offline suite. The optional live test separately requires healthy Ollama.

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

Skip ingestion regeneration after cloning: the required JSON is already tracked.
Only deliberately rebuild if approved sources or ingestion configuration change,
then review the resulting tracked JSON diff. If you need a new vector index,
validate embeddings first. These commands are optional maintenance, not setup:

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
