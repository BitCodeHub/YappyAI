#!/bin/bash
# Simple git push script

echo "🚀 Git Push Helper"
echo "=================="

# Check git status
echo "📊 Current status:"
git status --short

# Add all changes
echo ""
echo "📦 Adding all changes..."
git add .

# Get commit message
echo ""
echo "💬 Enter commit message (or press Enter for default):"
read -r commit_msg

# Use default message if empty
if [ -z "$commit_msg" ]; then
    commit_msg="update: changes from Claude"
fi

# Commit
echo ""
echo "📝 Committing with message: $commit_msg"
git commit -m "$commit_msg"

# Push
echo ""
echo "⬆️ Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Done! Changes pushed to GitHub."