# Run the FastAPI backend.
#   .\scripts\run_api.ps1              -> http://127.0.0.1:8000
#   .\scripts\run_api.ps1 -Reload      -> auto-reload for development

param(
    [switch]$Reload,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$args = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) { $args += "--reload" }

Write-Host "Starting MediStock API on http://127.0.0.1:$Port (docs at /docs)" -ForegroundColor Cyan
& $python @args
