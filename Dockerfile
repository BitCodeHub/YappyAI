# Minimal Dockerfile for Railway deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only essential files first
COPY requirements_minimal.txt .
COPY api_minimal.py .

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Skip frontend for now - focusing on getting API working first

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Railway will override with $PORT)
EXPOSE 8000

# Direct command - Railway will inject PORT env variable
CMD python -m uvicorn api_minimal:app --host 0.0.0.0 --port ${PORT:-8000}