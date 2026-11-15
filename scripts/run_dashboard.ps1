# Run the Streamlit dashboard.
#   .\scripts\run_dashboard.ps1
# Requires the API to be running (scripts\run_api.ps1) in another terminal.

param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

# Fail fast with a clear message instead of a page full of error states.
try {
    $health = Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 3
    Write-Host "API reachable: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "WARNING: API not reachable on http://127.0.0.1:8000" -ForegroundColor Yellow
    Write-Host "         Start it first:  .\scripts\run_api.ps1" -ForegroundColor Yellow
}

Write-Host "Starting dashboard on http://127.0.0.1:$Port" -ForegroundColor Cyan
& $python -m streamlit run dashboard/streamlit_app.py --server.port $Port --server.headless true
