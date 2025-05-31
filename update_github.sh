#!/bin/bash

echo "📦 Updating GitHub with Yappy AI Complete..."

# Add the complete version files
git add app_complete.py
git add static/index.html
git add requirements_complete.txt
git add Dockerfile
git add render.yaml

# Commit the changes
git commit -m "feat: complete Yappy AI with full features - auth, multi-LLM, chat UI"

# Push to GitHub
git push origin main

echo "✅ GitHub updated successfully!"
echo "🚀 Render should automatically deploy the new version"