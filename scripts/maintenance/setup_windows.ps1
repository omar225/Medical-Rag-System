param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $ProjectRoot

foreach ($CommandName in @('git', 'py')) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Install $CommandName manually, reopen PowerShell, then rerun this script."
    }
}
& py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12); print('Python', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 x64 is required; see docs/NEW_MACHINE_SETUP.md.' }

foreach ($CommandName in @('ollama', 'nvidia-smi')) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        Write-Warning "$CommandName is unavailable. Offline Python setup can continue; install system prerequisites manually."
    }
}
if ($CheckOnly) { return }

$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & py -3.12 -m venv (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $VenvPython -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Use Python 3.12 x64'"
if ($LASTEXITCODE -ne 0) { throw 'Existing .venv is incompatible. Create a separate Python 3.12 x64 environment.' }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; preserve the pins and inspect the reported error.' }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Host 'Python setup complete. No system software, model, ingestion, or index was installed or rebuilt.'
