# Verify the current build: lint, import, tests.
# Run from the repository root:  .\scripts\verify.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
Write-Host "Interpreter: $python" -ForegroundColor Cyan

Write-Host "`n[1/3] Ruff lint" -ForegroundColor Cyan
& $python -m ruff check .
if ($LASTEXITCODE -ne 0) { Write-Host "Lint failed." -ForegroundColor Red; exit 1 }

Write-Host "`n[2/3] Import smoke test" -ForegroundColor Cyan
& $python -c "from app.main import create_app; app = create_app(); print('app OK, routes:', len(app.routes))"
if ($LASTEXITCODE -ne 0) { Write-Host "Import failed." -ForegroundColor Red; exit 1 }

Write-Host "`n[3/3] Pytest" -ForegroundColor Cyan
& $python -m pytest
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed." -ForegroundColor Red; exit 1 }

Write-Host "`nAll checks passed." -ForegroundColor Green
