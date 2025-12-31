$latest = "domains/constraint_adherence/results/2025-12-31T12-34-33-120447"

# Check response has dry_run data populated
$response = Get-Content "$latest/responses/recursion_analytical.json" | ConvertFrom-Json
Write-Host "=== DRY-RUN DATA ==="
Write-Host "kernel (first 100 chars): $($response.dry_run.kernel.Substring(0,100))"
Write-Host "system_prompt populated: $($response.dry_run.system_prompt.Length -gt 0)"
Write-Host "user_prompt: $($response.dry_run.user_prompt)"
Write-Host "parameters: $($response.dry_run.parameters | ConvertTo-Json -Compress)"
Write-Host ""
Write-Host "=== INVARIANT CHECK ==="
Write-Host "kernel_match: $($response.invariant_check.kernel_match)"
Write-Host ""
Write-Host "=== RUN DIR ==="
Write-Host "run_dir: $latest"