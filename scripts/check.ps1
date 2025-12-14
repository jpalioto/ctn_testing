<#
.SYNOPSIS
    Run all checks: type checking + unit tests
.DESCRIPTION
    Runs pyright and pytest. Fails fast on first error.
    Use before commits to ensure code quality.
.EXAMPLE
    .\scripts\check.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CTN TESTING - PRE-COMMIT CHECKS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Type checking
Write-Host "[1/2] PYRIGHT - Type Checking" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
pyright ctn_testing/
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: Type errors found" -ForegroundColor Red
    exit 1
}
Write-Host "PASSED" -ForegroundColor Green
Write-Host ""

# 2. Unit tests
Write-Host "[2/2] PYTEST - Unit Tests" -ForegroundColor Yellow
Write-Host "--------------------------------------------"
pytest tests/ -v --tb=short
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
