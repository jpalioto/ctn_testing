<#
.SYNOPSIS
    Run all checks: linting + type checking + unit tests
.DESCRIPTION
    Runs ruff, pyright, and pytest. Fails fast on first error.
    Use before commits to ensure code quality.
.EXAMPLE
    .\scripts\check.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CTN TESTING - PRE-COMMIT CHECKS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Ruff lint
Write-Host "[1/4] RUFF CHECK - Linting" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
uv run ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: Linting errors found" -ForegroundColor Red
    exit 1
}
Write-Host "PASSED" -ForegroundColor Green
Write-Host ""

# 2. Ruff format check
Write-Host "[2/4] RUFF FORMAT - Format Check" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
uv run ruff format --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: Formatting issues found" -ForegroundColor Red
    exit 1
}
Write-Host "PASSED" -ForegroundColor Green
Write-Host ""

# 3. Type checking
Write-Host "[3/4] PYRIGHT - Type Checking" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
uv run pyright
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: Type errors found" -ForegroundColor Red
    exit 1
}
Write-Host "PASSED" -ForegroundColor Green
Write-Host ""

# 4. Unit tests
Write-Host "[4/4] PYTEST - Unit Tests" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
uv run pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: Test failures" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Success
Write-Host "============================================" -ForegroundColor Green
Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
