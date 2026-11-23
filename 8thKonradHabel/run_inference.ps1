# PowerShell Runner for 8th Place RSNA Solution
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv"
}

Write-Host "Running 8th Place RSNA Aneurysm Detection Inference..." -ForegroundColor Cyan
& $python run_inference.py -i sample_data/series/case_001 -m model -f 0,1,2,5 -o results/predictions.csv -d cuda:0
