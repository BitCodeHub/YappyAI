# Production Dockerfile - Fixed version
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements first (better caching)
COPY requirements_complete.txt ./
RUN pip install --no-cache-dir -r requirements_complete.txt

# Copy application code
COPY . .

# Ensure static directory exists with files
RUN mkdir -p .screenshots .logs && \
    echo "Contents of /app:" && ls -la /app/ && \
    echo "Contents of /app/static:" && ls -la /app/static/ || echo "Static directory not found"

# Environment
ENV PYTHONUNBUFFERED=1
# Don't hardcode PORT - let Railway set it

# Remove HEALTHCHECK - Railway handles this externally

EXPOSE 8000

# Start command - use complete version
CMD ["sh", "-c", "python app_complete.py"]