# Production Dockerfile for Yappy
FROM node:18-alpine AS frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY frontend/agentic-seek-front/package*.json ./
RUN npm ci
COPY frontend/agentic-seek-front/ ./
ENV REACT_APP_API_URL=""
RUN npm run build

# Python backend
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt && \
    pip install openai anthropic google-generativeai groq

# Copy application files
COPY api_deploy.py .
COPY app.py .
COPY sources sources/
COPY prompts prompts/
COPY config.ini .

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/build ./static

# Create necessary directories
RUN mkdir -p .screenshots .logs

# Environment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start the production API
CMD ["python", "api_deploy.py"]