# Hugging Face Spaces Dockerfile for SolBridge AI ML Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (C compiler for Prophet)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs

# Expose port
EXPOSE 7860

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
ENV HOST=0.0.0.0
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=5)"

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
