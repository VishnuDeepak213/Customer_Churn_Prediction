$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$payload = @{
    SeniorCitizen = 0
    tenure = 24
    MonthlyCharges = 65.5
    TotalCharges = 1570.0
    Contract = 'One year'
    PhoneService = 'Yes'
    InternetService = 'Fiber optic'
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri 'http://localhost:8000/predict' `
    -ContentType 'application/json' `
    -Body $payload