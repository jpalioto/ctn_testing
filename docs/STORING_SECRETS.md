# STORING_SECRETS.md

## Overview

API keys go in Windows SecretStore, not in files. This keeps keys out of git history and environment dumps.

## Setup (One Time)
```powershell
# Install modules if needed
Install-Module Microsoft.PowerShell.SecretManagement -Scope CurrentUser
Install-Module Microsoft.PowerShell.SecretStore -Scope CurrentUser

# Configure SecretStore (first time only - sets master password)
Set-SecretStoreConfiguration -Scope CurrentUser -Authentication Password
```

## Adding Keys
```powershell
Set-Secret -Name "ANTHROPIC_API_KEY" -SecureStringSecret (Read-Host -AsSecureString)
Set-Secret -Name "GEMINI_API_KEY" -SecureStringSecret (Read-Host -AsSecureString)
```

## Verifying Keys
```powershell
# List stored keys
Get-SecretInfo

# Test connectivity (after loading into env)
Test-ApiKeys
```

## Loading Keys

Keys are loaded into environment variables by `scripts/activate.ps1` or your PowerShell profile. They exist only in memory for the session.
```powershell
# Project activation
. .\scripts\activate.ps1

# Or if using profile script, keys load automatically
```

## Key Rotation
```powershell
# Overwrite existing
Set-Secret -Name "ANTHROPIC_API_KEY" -SecureStringSecret (Read-Host -AsSecureString)
```

## Troubleshooting
```powershell
# Unlock SecretStore if locked
Unlock-SecretStore

# Remove and re-add if corrupted
Remove-Secret -Name "ANTHROPIC_API_KEY"
Set-Secret -Name "ANTHROPIC_API_KEY" -SecureStringSecret (Read-Host -AsSecureString)
```