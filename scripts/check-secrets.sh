#!/bin/bash
#
# check-secrets.sh - Scan staged files for API keys and secrets
#
# This script is run by the pre-commit hook to prevent accidental
# commits of API keys, secrets, or credentials.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Scanning staged files for secrets..."

# Get list of staged files (excluding deleted files)
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || echo "")

if [ -z "$STAGED_FILES" ]; then
    echo -e "${GREEN}No staged files to check.${NC}"
    exit 0
fi

# Patterns to detect (extended regex)
PATTERNS=(
    # Anthropic API keys
    'sk-ant-[a-zA-Z0-9_-]{20,}'

    # OpenAI API keys
    'sk-proj-[a-zA-Z0-9_-]{20,}'
    'sk-[a-zA-Z0-9]{48,}'

    # Google API keys
    'AIzaSy[a-zA-Z0-9_-]{33}'

    # AWS keys
    'AKIA[0-9A-Z]{16}'
    'aws_secret_access_key\s*=\s*["\047][^"\047]+'

    # Hardcoded env assignments (in code, not .env files)
    'ANTHROPIC_API_KEY\s*=\s*["\047][a-zA-Z0-9_-]+'
    'OPENAI_API_KEY\s*=\s*["\047][a-zA-Z0-9_-]+'
    'GEMINI_API_KEY\s*=\s*["\047][a-zA-Z0-9_-]+'
    'GOOGLE_API_KEY\s*=\s*["\047][a-zA-Z0-9_-]+'

    # Generic secret patterns
    'api[_-]?key\s*=\s*["\047][a-zA-Z0-9_-]{20,}["\047]'
    'secret\s*=\s*["\047][a-zA-Z0-9_-]{20,}["\047]'
    'password\s*=\s*["\047][^"\047]{8,}["\047]'
    'token\s*=\s*["\047][a-zA-Z0-9_-]{20,}["\047]'
)

FOUND_SECRETS=0

for file in $STAGED_FILES; do
    # Skip binary files and this script
    if [ ! -f "$file" ] || [[ "$file" == "scripts/check-secrets.sh" ]]; then
        continue
    fi

    # Skip common non-text files
    case "$file" in
        *.png|*.jpg|*.jpeg|*.gif|*.ico|*.woff|*.woff2|*.ttf|*.eot)
            continue
            ;;
    esac

    for pattern in "${PATTERNS[@]}"; do
        # Use grep with extended regex, case insensitive
        if git show ":$file" 2>/dev/null | grep -iE "$pattern" > /dev/null 2>&1; then
            if [ $FOUND_SECRETS -eq 0 ]; then
                echo ""
                echo -e "${RED}========================================${NC}"
                echo -e "${RED}  POTENTIAL SECRETS DETECTED!${NC}"
                echo -e "${RED}========================================${NC}"
                echo ""
            fi

            echo -e "${YELLOW}File:${NC} $file"
            echo -e "${YELLOW}Pattern:${NC} $pattern"
            echo -e "${YELLOW}Matches:${NC}"
            git show ":$file" 2>/dev/null | grep -inE "$pattern" | head -5 | while read line; do
                echo "  $line"
            done
            echo ""

            FOUND_SECRETS=1
        fi
    done
done

if [ $FOUND_SECRETS -eq 1 ]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  COMMIT BLOCKED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Secrets or API keys were detected in staged files."
    echo "Please remove them before committing."
    echo ""
    echo "If this is a false positive, you can:"
    echo "  1. Add the pattern to .gitignore"
    echo "  2. Use 'git commit --no-verify' (not recommended)"
    echo ""
    exit 1
fi

echo -e "${GREEN}No secrets detected in staged files.${NC}"
exit 0
