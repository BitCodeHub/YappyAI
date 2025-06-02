#!/bin/bash

# Auto-commit and push script
# Usage: ./auto_commit.sh "commit message"

if [ -z "$1" ]; then
    echo "Please provide a commit message"
    exit 1
fi

# Add all changes
git add -A

# Commit with provided message
git commit -m "$1

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to origin
git push origin main

echo "✅ Changes committed and pushed to GitHub"