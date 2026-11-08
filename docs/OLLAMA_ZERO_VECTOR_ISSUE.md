# Unresolved BGE-M3 zero-vector issue

Original machine: Windows/PowerShell, GTX 1080 Ti (11 GB VRAM), 16 GB RAM,
Python 3.12.10, Ollama reporting `0.40.15`. Model tag `bge-m3:latest`, full digest
`7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
These are recorded observations, not an assertion about another installation.

The Stage 3 index successfully embedded 113 source chunks into 1,024-dimensional
vectors. Later Stage 3.1 comparison completed with frozen hybrid Hit@5 100% and
held-out Hit@5 95.83%. Subsequently the opt-in live integration check and hybrid
CLI failed: HTTP 200 embedding responses contained 1,024 zeros per input.
An English single input, neutral warm-up input, and English/Arabic batch reproduced
the failure. One manual idle-model unload/reload did not clear it. Cause unresolved.
There was no automatic Ollama upgrade, downgrade, reinstall, model download, or
validation relaxation as part of the handoff.

The existing validator raises `Zero-length embeddings cannot be used for cosine
similarity`. Keep it: a zero vector has no meaningful cosine direction. Lexical
retrieval can still use the restored published index without model inference.

## Preserved evidence

The recovery archive restores these exact reports:

- `reports/evaluation/stage3_retrieval.json`: original successful dense experiment.
- `reports/evaluation/stage31_comparison.json`: successful six-run comparison.
- `reports/evaluation/stage31_cold.json`: earlier unloaded-model cold/warm probe.
- `reports/evaluation/stage31_runtime_probe.json`: subsequent zero-vector probe.

The failing probe reported `llamacpp`, all GPU layers, and runtime context 4,096;
the model advertises training context 8,192. Neither context size nor GPU selection
is established as the cause. The complete default suite passed 213 tests with
one optional live test skipped; the separately enabled live test failed.

## First test on the new GPU

After installing dependencies and explicitly pulling/verifying BGE-M3:

```powershell
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:EMBEDDING_MODEL = 'bge-m3'
.\.venv\Scripts\python.exe scripts/maintenance/probe_embeddings.py --output reports/evaluation/new_gpu_probe.json
if ($LASTEXITCODE -ne 0) { throw 'Do not rebuild the vector index' }
```

The standalone diagnostic makes single-input and mixed English/Arabic batch
requests with truncation disabled, prints installed model identity, dimension,
nonzero count, finite-value status, vector norm, response timings and loaded-model
metadata, and exits nonzero for invalid BGE-M3 output. It requires only Python's
standard library and writes a report only when `--output` is supplied.

Equivalent minimal direct request and measurement:

```powershell
$Body = @{ model='bge-m3:latest'; input=@('Source document test'); truncate=$false } | ConvertTo-Json -Depth 4
$Response = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:11434/api/embed' -ContentType 'application/json' -Body $Body
$Vector = @($Response.embeddings[0])
$SumSquares = 0.0
$Nonzero = 0
foreach ($Value in $Vector) {
    $Number = [double]$Value
    $SumSquares += $Number * $Number
    if ($Number -ne 0) { $Nonzero++ }
}
[pscustomobject]@{ Dimension=$Vector.Count; Nonzero=$Nonzero; Norm=[Math]::Sqrt($SumSquares) }
```

Expected BGE-M3 dimension: 1,024; all numbers finite and norm greater than zero.
See the [Ollama embedding API](https://docs.ollama.com/api/embed).

## Collect runtime and GPU evidence

```powershell
ollama --version
Invoke-RestMethod 'http://127.0.0.1:11434/api/version'
(Invoke-RestMethod 'http://127.0.0.1:11434/api/tags').models | Select-Object name,digest,size
ollama ps
Invoke-RestMethod 'http://127.0.0.1:11434/api/ps' | ConvertTo-Json -Depth 8
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA 'Ollama\server.log') -Tail 100
```

`ollama ps` reports CPU/GPU placement; combine it with GPU memory observations
and server logs, not just model presence. Windows log locations are documented by
[Ollama troubleshooting](https://docs.ollama.com/troubleshooting). Review logs for
personal paths or secrets before sharing; raw logs are intentionally excluded.
The [NVIDIA monitoring reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
documents `nvidia-smi`.

For an **optional manual CPU comparison**, first quit the Ollama tray application
and ensure no server still owns port 11434. In a dedicated terminal:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
ollama serve
```

In another terminal run the same diagnostic and `ollama ps`. This CPU-selection
method follows [Ollama's GPU documentation](https://docs.ollama.com/gpu). Verify
actual CPU placement, since behavior may differ by runtime/backend. Stop the
temporary server with Ctrl+C, remove the variable in its terminal, then restart
the normal tray application:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
```

This comparison is not a pipeline configuration change or proven fix. It has not
been performed as part of this handoff. Do not run two servers simultaneously.

## Optional existing live test

```powershell
$env:RUN_OLLAMA_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_ollama_embeddings.py -p no:cacheprovider --tb=short
Remove-Item Env:RUN_OLLAMA_TESTS
```

A different GPU, driver, or Ollama build may behave differently. Record each
combination, model digest, and fresh probe; do not attribute a change to the GPU
without controlling other differences. The handoff does not modify the runtime
or mark the issue resolved. Only attempt an index rebuild after valid direct
embeddings, then pass the full build and live integration check.
