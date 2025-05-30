#!/bin/bash

echo "🐾 Preparing Yappy for deployment..."

# Build frontend
echo "📦 Building frontend..."
cd frontend/agentic-seek-front
npm install
npm run build

# Copy built frontend to static directory
echo "📁 Copying frontend build to static directory..."
cd ../..
rm -rf static
cp -r frontend/agentic-seek-front/build static

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Yappy is ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Push to GitHub: git add . && git commit -m 'Deploy Yappy' && git push"
echo "2. Deploy on Railway: https://railway.app"
echo "3. Or use any Docker hosting service with the provided Dockerfile"
echo ""
echo "🚀 Happy deploying!"