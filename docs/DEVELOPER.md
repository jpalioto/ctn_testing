# CTN Developer Guide

Commands and workflows for testing, debugging, and development.

---

## SDK Server (ctn_sdk)

```powershell
# Start SDK server
ctn serve --log-level info
ctn serve --log-level debug

# Test dry-run (see what kernel would be sent)
ctn send "@analytical hello" --strategy ctn --dry-run
ctn send "@analytical hello" --strategy operational --dry-run

# Actual send
ctn send "@analytical hello" --strategy ctn
ctn send "@terse Explain recursion" --strategy operational
```

---

## SDK HTTP Testing (PowerShell)

```powershell
# Set UTF-8 encoding for Greek symbols
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Test /send endpoint
$body = @{ input = "@analytical hello"; provider = "anthropic"; strategy = "ctn" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:14380/send" -Method Post -ContentType "application/json" -Body $body
$response | ConvertTo-Json -Depth 5
$response.kernel

# Test dry-run via HTTP
$body = @{ input = "@analytical hello"; provider = "anthropic"; strategy = "ctn"; dryRun = $true } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:14380/send" -Method Post -ContentType "application/json" -Body $body
$response | ConvertTo-Json -Depth 5

# Health check
Invoke-RestMethod -Uri "http://localhost:14380/health"
```

---

## CTN Testing (ctn_testing)

### Run Full Evaluation

```powershell
uv run python -c "
from pathlib import Path
from ctn_testing.runners.evaluation import ConstraintEvaluator, format_status

def progress(stage, current, total, success=True, error_msg=None):
    status = format_status(success)
    print(f'{stage}: {current}/{total} {status}')
    if error_msg:
        print(f'  Error: {error_msg}')

evaluator = ConstraintEvaluator(Path('domains/constraint_adherence/configs/phase1.yaml'))
result = evaluator.run(progress_callback=progress)
print(f'\nCompleted: {result.run_dir}')
print(f'SDK calls: {len(result.run_results)}')
print(f'Comparisons: {len(result.comparisons)}')
"
```

### Quick SDK Runner Test

```powershell
uv run python -c "
from ctn_testing.runners.http_runner import SDKRunner
runner = SDKRunner(strategy='ctn')
result = runner.send_with_dry_run('@analytical test', provider='anthropic')
print(f'kernel_match: {result.kernel_match}')
print(f'kernel: {result.dry_run.kernel[:200]}')
print(f'user_prompt: {result.dry_run.user_prompt}')
print(f'parameters: {result.dry_run.parameters}')
"
```

### Results Browser

```powershell
# Start Streamlit browser
uv run streamlit run ctn_testing/browser/app.py -- --results-dir domains/constraint_adherence/results

# Or via CLI
uv run python -m ctn_testing.cli browse domains/constraint_adherence/results
```

### Clean Up Results

```powershell
Remove-Item -Recurse -Force domains/constraint_adherence/results/* -ErrorAction SilentlyContinue
```

---

## Testing & Linting

```powershell
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ctn_testing --cov-report=term-missing --cov-report=html

# Run specific test file
uv run pytest tests/test_http_runner.py -v

# Linting
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
uv run ruff format --check .

# Type checking
uv run pyright
```

---

## Inspecting Results (PowerShell)

```powershell
# Find latest run
$latest = (Get-ChildItem domains/constraint_adherence/results/ | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

# Check response file
Get-Content "$latest/responses/recursion_analytical.json" | ConvertFrom-Json | ConvertTo-Json -Depth 5

# Check manifest
Get-Content "$latest/manifest.json" | ConvertFrom-Json | ConvertTo-Json -Depth 3

# Check analysis
Get-Content "$latest/analysis/summary.json" | ConvertFrom-Json | ConvertTo-Json -Depth 3

# Check judging
Get-ChildItem "$latest/judging" | Select-Object -First 1 | Get-Content | ConvertFrom-Json | ConvertTo-Json -Depth 5

# Verify dry-run data populated
$response = Get-Content "$latest/responses/recursion_analytical.json" | ConvertFrom-Json
$response.dry_run | ConvertTo-Json -Depth 3
$response.invariant_check
```

---

## Git Commands

```powershell
# Restore all tracked files to last commit
git restore .

# Undo last commit, keep changes staged
git reset --soft HEAD~1

# Remove file from staging
git reset HEAD <filename>

# Force push (when you've rewritten local history)
git push --force origin master
```

---

## SDK Build & Test (ctn_sdk)

```powershell
# Build all packages
pnpm build

# Run all tests
pnpm -r test

# Run specific package tests
pnpm --filter @ctn/language test
pnpm --filter @ctn/anthropic test

# Type check
pnpm typecheck
```

---

## Config Files

| File | Purpose |
|------|---------|
| `domains/constraint_adherence/configs/phase1.yaml` | Operational strategy config |
| `domains/constraint_adherence/configs/phase1_ctn.yaml` | CTN strategy config |
| `domains/constraint_adherence/prompts.yaml` | Test prompts |
| `domains/constraint_adherence/traits.yaml` | Judging trait definitions |

---

## Response Data Format

After running an evaluation, response files contain:

```json
{
  "prompt_id": "recursion",
  "prompt_text": "Explain recursion",
  "constraint_name": "analytical",
  "input_sent": "@analytical Explain recursion",
  "output": "...",
  "provider": "anthropic",
  "model": "claude-sonnet-4-5-20250929",
  "tokens": { "input": 44, "output": 495 },
  "timestamp": "2025-12-31T12:34:56.789",
  "error": null,
  "dry_run": {
    "kernel": "<behavioral_constraints>...",
    "system_prompt": "<behavioral_constraints>...",
    "user_prompt": "Explain recursion",
    "parameters": { "temperature": 0.84, "top_k": 47.2 }
  },
  "kernel": "<behavioral_constraints>...",
  "invariant_check": {
    "kernel_match": true
  }
}
```

### Key Fields

- `dry_run`: What was captured BEFORE the API call
- `kernel`: What was returned with the response (should match dry_run.kernel)
- `invariant_check.kernel_match`: Verification that both kernels are identical

---

## Troubleshooting

### UTF-8 Encoding Issues

If Greek symbols (Σ, Ψ, Ω, τ) display as garbage:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### SDK Server Not Responding

```powershell
# Check if server is running
Invoke-RestMethod -Uri "http://localhost:14380/health"

# Check server stats
Invoke-RestMethod -Uri "http://localhost:14380/stats"
```

### Timeouts During Evaluation

Increase timeout in config:

```yaml
runner:
  type: http
  base_url: http://localhost:14380
  timeout: 120  # seconds
```
