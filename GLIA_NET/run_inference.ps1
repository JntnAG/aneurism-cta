# PowerShell Runner for GLIA-Net
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv"
}

Write-Host "Running GLIA-Net Segmentation Inference..." -ForegroundColor Cyan
& $python inference.py -c inference_GLIA-Net -i sample_data/case_001.nii.gz -t nii -o results -d 0
