param(
    [string]$Python = '',
    [string]$RunDir = 'results/e000_20261027_collection_v2',
    [switch]$Model
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
if (-not $Python) {
    $taskPythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($taskPythonCommand) { $Python = $taskPythonCommand.Source }
    else { $Python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with the full path to Python 3.11 or 3.12.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE) { throw 'venv creation failed' }
}
$taskVenvPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path '.cache\tmp' -Force | Out-Null
$env:PIP_NO_CACHE_DIR = '1'
$env:TEMP = Join-Path $taskRoot '.cache\tmp'
$env:TMP = $env:TEMP
try {
    & $taskVenvPython -m pip install -e '.[dev]'
    if ($LASTEXITCODE) { throw 'CPU dependency installation failed' }
    if ($Model) {
        & $taskVenvPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE) { throw 'CUDA installation failed' }
        & $taskVenvPython -m pip install -e '.[dev,model]'
        if ($LASTEXITCODE) { throw 'Model dependency installation failed' }
    }
} finally {
    $taskWatch.Stop()
    if (Test-Path -LiteralPath '.venv\Lib\site-packages\yaml') {
        & $taskVenvPython -m video_compute_value.e000 account --config configs/e000.yaml --run-dir $RunDir --event-name 'Bootstrap including downloads, environment and installation' --event-seconds $taskWatch.Elapsed.TotalSeconds
    }
}
