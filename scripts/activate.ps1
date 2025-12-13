# CTN Testing Environment Activation
# Usage: . .\scripts\activate.ps1

# Activate venv
.\.venv\Scripts\Activate.ps1

# Load API keys from SecretStore
if (Get-Module -ListAvailable Microsoft.PowerShell.SecretManagement) {
    Import-Module Microsoft.PowerShell.SecretManagement
    Import-Module Microsoft.PowerShell.SecretStore

    $keys = @("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
    
    foreach ($keyName in $keys) {
        try {
            $secret = Get-Secret -Name $keyName -AsPlainText -ErrorAction Stop
            Set-Item -Path "env:$keyName" -Value $secret
            $secret = $null
            Write-Host " [CTN] $keyName loaded." -ForegroundColor Cyan
        }
        catch {
            Write-Host " [CTN] $keyName not found in SecretStore." -ForegroundColor Yellow
        }
    }
}
else {
    Write-Host " [CTN] SecretManagement module not installed. Keys not loaded." -ForegroundColor Yellow
    Write-Host "       Install with: Install-Module Microsoft.PowerShell.SecretManagement" -ForegroundColor Gray
}

Write-Host " [CTN] Environment ready." -ForegroundColor Green
