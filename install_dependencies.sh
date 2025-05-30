#!/bin/bash

echo "📦 Installing dependencies for AgenticSeek with PDF support..."
echo "================================================"

# Change to the project directory
cd "$(dirname "$0")"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "📋 Installing from requirements.txt..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "📄 PDF support dependencies installed:"
    echo "  - pypdf (PDF text extraction)"
    echo "  - matplotlib (Chart generation)"
    echo "  - seaborn (Advanced visualizations)"
    echo "  - pandas (Data analysis)"
    echo ""
    echo "🚀 You can now start the API with:"
    echo "  python api_clean.py"
    echo "  OR"
    echo "  python start_neo_api.py"
else
    echo "❌ Failed to install some dependencies"
    echo "Please check the error messages above"
fi