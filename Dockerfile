# Production Dockerfile - Fixed version
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements first (better caching)
COPY requirements_minimal.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements_minimal.txt && \
    pip install openai anthropic google-generativeai groq || \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static .screenshots .logs

# Environment
ENV PYTHONUNBUFFERED=1
# Don't hardcode PORT - let Railway set it

# Remove HEALTHCHECK - Railway handles this externally

EXPOSE 8000

# Start command
CMD ["sh", "-c", "python api.py"]