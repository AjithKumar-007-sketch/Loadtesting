# Loads .env into this session and starts the call API server on port 8081.
# Usage: powershell -ExecutionPolicy Bypass -File .\run_server.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) { Write-Error ".env not found. Copy .env.example to .env and fill it in." }

Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $name, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
    }
}

$missing = @("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_SIP_TRUNK_ID") |
    Where-Object { -not [Environment]::GetEnvironmentVariable($_, "Process") }
if ($missing) {
    Write-Host "Missing values in .env: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "Starting API server on http://localhost:8081 (LiveKit: $env:LIVEKIT_URL)" -ForegroundColor Green
& ".\.venv\Scripts\python.exe" simple_livekit_audio_test.py --server --port 8081
