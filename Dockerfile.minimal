# Minimal Dockerfile for Railway deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only essential files first
COPY requirements_minimal.txt .
COPY app.py .
COPY entrypoint.sh .

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run Python directly - app.py handles PORT internally
CMD ["python", "app.py"]