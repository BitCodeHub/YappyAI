#!/bin/bash
# Double-clickable git push for Mac

cd "$(dirname "$0")"

echo "🚀 Quick Git Push"
echo "================"
echo ""

# Show what's changed
echo "📊 Changes to push:"
git status --short
echo ""

# Add all changes
git add .

# Create commit with timestamp
commit_msg="update: changes from $(date '+%Y-%m-%d %H:%M')"
git commit -m "$commit_msg"

# Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push origin main

echo ""
echo "✅ All done! Press Enter to close."
read