# PowerShell Runner for RSNA 2025 1st Place Benchmark
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv"
}

Write-Host "Running RSNA 2025 (1st Place tomoon33) Benchmark..." -ForegroundColor Cyan
& $python run_benchmark.py --input-dir data/series --output results/predictions.csv
