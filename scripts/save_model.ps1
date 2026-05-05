$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$activateScript = Join-Path $projectRoot 'venv\Scripts\Activate.ps1'
if (Test-Path $activateScript) {
    . $activateScript
}

python scripts/train_first_model.py