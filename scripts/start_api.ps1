$projectRoot = Split-Path -Parent $PSScriptRoot
$activateScript = Join-Path $projectRoot 'venv\Scripts\Activate.ps1'

if (Test-Path $activateScript) {
    $apiCommand = ". '$activateScript'; python -m uvicorn api.main:app --reload"
} else {
    $apiCommand = "python -m uvicorn api.main:app --reload"
}

Start-Process powershell -ArgumentList @('-NoExit', '-Command', "Set-Location '$projectRoot'; $apiCommand")