#!/bin/bash

echo "Starting AgenticSeek without Docker dependencies..."
echo "Note: Web search functionality will be limited"
echo ""

# Check if virtual environment exists
if [ ! -d "agentic_seek_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv agentic_seek_env
fi

# Activate virtual environment
source agentic_seek_env/bin/activate

# Install core dependencies only
echo "Installing core dependencies..."
pip install fastapi uvicorn pydantic aiofiles pyjwt bcrypt requests

echo ""
echo "To start the API server, run:"
echo "python3 api.py"
echo ""
echo "To start the frontend, run:"
echo "cd frontend/agentic-seek-front && npm start"