# Minimal Dockerfile for Railway deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only essential files first
COPY requirements_minimal.txt .
COPY api_minimal.py .
COPY app.py .

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Use shell form to allow variable expansion
CMD ["/bin/sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]