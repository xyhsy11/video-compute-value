param(
    [string]$VidfitRoot = 'C:\Vidfit',
    [int]$Port = 7861
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $VidfitRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "VidFit's UI environment was not found at $pythonPath"
}
& $pythonPath (Join-Path $PSScriptRoot 'vidfit_experiments.py') --root $projectRoot --vidfit-root $VidfitRoot --serve --port $Port
exit $LASTEXITCODE
