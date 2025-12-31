#!/bin/bash
# CTN Testing - Lint Script
# Runs all linting and type checking tools

set -e

echo "============================================"
echo "  CTN TESTING - LINT & TYPE CHECK"
echo "============================================"
echo ""

# 1. Ruff lint
echo "[1/4] RUFF CHECK - Linting"
echo "--------------------------------------------"
uv run ruff check .
echo "PASSED"
echo ""

# 2. Ruff format check
echo "[2/4] RUFF FORMAT - Format Check"
echo "--------------------------------------------"
uv run ruff format --check .
echo "PASSED"
echo ""

# 3. Pyright type checking
echo "[3/4] PYRIGHT - Type Checking"
echo "--------------------------------------------"
uv run pyright
echo "PASSED"
echo ""

# 4. Pytest
echo "[4/4] PYTEST - Unit Tests"
echo "--------------------------------------------"
uv run pytest tests/ -v --tb=short
echo ""

echo "============================================"
echo "  ALL CHECKS PASSED"
echo "============================================"
