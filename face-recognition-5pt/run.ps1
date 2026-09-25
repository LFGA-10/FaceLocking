# run.ps1 - one-shot launcher for the Falcon Eye / Eagle Eye recognition scanner.
# Usage:  .\run.ps1            (uses camera index 2 - working webcam on this machine)
#         .\run.ps1 -CamIndex 0   (built-in IR camera)
#         .\run.ps1 -CamIndex 1   (secondary camera)

param(
    [int]$CamIndex = 2
)

# Activate virtual environment
$VenvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
} else {
    Write-Warning "Virtual environment not found at .venv - using system python"
}

$env:FALCON_CAM_INDEX = "$CamIndex"
Write-Host "Using camera index $CamIndex (FALCON_CAM_INDEX=$CamIndex)"

python -m src.recognize
