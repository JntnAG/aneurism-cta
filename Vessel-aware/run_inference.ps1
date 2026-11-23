# PowerShell Runner for Vessel-aware Aneurysm Detection
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv"
}

$env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"

Write-Host "Running Vessel-aware Inference..." -ForegroundColor Cyan
& $python run_inference.py --scan_dir sample_data/crop_0.4 --edt_dir sample_data/crop_0.4_vessel_edt --output_dir results --threshold 0.1
