# scripts/fix_field_rename.ps1
# Run from repo root

$files = @(
    "ctn_testing/core/types.py",
    "ctn_testing/metrics/scorer.py",
    "ctn_testing/runners/results.py",
    "ctn_testing/runners/runner.py",
    "ctn_testing/runners/kernel.py",
    "scripts/smoke_test.py"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # ext.field -> ext.field_name
        $content = $content -replace '\bext\.field\b', 'ext.field_name'
        
        # e.field -> e.field_name (in dict comprehensions)
        $content = $content -replace '\be\.field\b', 'e.field_name'
        
        # f.field -> f.field_name (FieldResult access)
        $content = $content -replace '\bf\.field\b', 'f.field_name'
        
        # gt.field -> gt.field_name
        $content = $content -replace '\bgt\.field\b', 'gt.field_name'
        
        # field= kwarg in constructors -> field_name=
        $content = $content -replace '\bfield=([^,\)]+)', 'field_name=$1'
        
        Set-Content $file $content -NoNewline
        Write-Host "Fixed: $file" -ForegroundColor Green
    }
}

Write-Host "`nDone. Run: pyright ctn_testing/" -ForegroundColor Cyan