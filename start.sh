#!/bin/bash
echo "Starting Yappy application..."
echo "PORT environment variable: ${PORT:-not set}"
echo "Using port: ${PORT:-8000}"

# Try to start the minimal API first (for deployment testing)
if [ -f "api_minimal.py" ]; then
    echo "Starting minimal API for deployment..."
    exec uvicorn api_minimal:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
else
    echo "Starting full API..."
    exec uvicorn api:api --host 0.0.0.0 --port ${PORT:-8000} --log-level info
fi