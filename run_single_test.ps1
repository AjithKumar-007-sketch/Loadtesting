# Sends exactly one call through the running API server and saves the result.
# Start run_server.ps1 in another terminal first.
# Usage: powershell -ExecutionPolicy Bypass -File .\run_single_test.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force "results" | Out-Null

# Any number works with SIPp; it auto-answers every call.
& ".\.venv\Scripts\python.exe" benchmark_livekit_calls.py `
    --sustained --target-cps 1 --duration 1 --max-concurrent 1 `
    --api-url http://localhost:8081 `
    --phone "+15551234567" `
    --wav-file uploads/sample.wav `
    --output results/single_user.json
