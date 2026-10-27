param([string]$RunDir = 'results/e000_20261027_collection_v2')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
try {
    & '.\.venv\Scripts\python.exe' -m pytest -q --basetemp '.cache/pytest'
    if ($LASTEXITCODE) { throw 'Targeted validity tests failed' }
    & '.\.venv\Scripts\ruff.exe' check src tests
    if ($LASTEXITCODE) { throw 'Static checks failed' }
} finally {
    $taskWatch.Stop()
    & '.\.venv\Scripts\python.exe' -m video_compute_value.e000 account --config configs/e000.yaml --run-dir $RunDir --event-name 'CPU validity tests and static checks' --event-seconds $taskWatch.Elapsed.TotalSeconds
}
