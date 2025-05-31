#!/bin/bash

echo "🚀 Deploying Yappy AI Complete to GitHub..."

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files to git..."
git add app_complete.py static/index.html requirements_complete.txt Dockerfile render.yaml

# Create commit
echo "Creating commit..."
git commit -m "feat: complete Yappy AI with full features - auth, multi-LLM, chat UI"

# Check if remote exists
if ! git remote | grep -q origin; then
    echo ""
    echo "⚠️  No remote repository found!"
    echo "Please add your GitHub repository:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo ""
    echo "Then run: git push -u origin main"
else
    echo "Pushing to GitHub..."
    git push origin main
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "Next steps:"
    echo "1. Go to https://dashboard.render.com/"
    echo "2. Create a new Web Service"
    echo "3. Connect your GitHub repo"
    echo "4. Deploy will start automatically!"
fi