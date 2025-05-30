# Simple production Dockerfile without frontend build
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt && \
    pip install openai anthropic google-generativeai groq

# Copy application files
COPY api.py .
COPY sources sources/
COPY config.ini .

# Create directories
RUN mkdir -p .screenshots .logs static

# Environment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use the working API
CMD ["python", "api.py"]