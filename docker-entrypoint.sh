#!/bin/bash
# Docker Entrypoint Script for ML Service
# Used when deploying to cloud services (Render, Railway, Heroku, etc)

set -e

echo "=============================================================="
echo "ML SERVICE DOCKER STARTUP"
echo "=============================================================="
echo "Time: $(date)"
echo "User: $(whoami)"
echo "Python: $(python3 --version)"
echo "Working Directory: $(pwd)"
echo ""

# Set environment variables if not already set
export ENVIRONMENT=${ENVIRONMENT:-production}
export ML_HOST=${ML_HOST:-0.0.0.0}
export ML_PORT=${ML_PORT:-8001}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export PYTHONUNBUFFERED=1

echo "Environment Configuration:"
echo "  ENVIRONMENT: $ENVIRONMENT"
echo "  ML_HOST: $ML_HOST"
echo "  ML_PORT: $ML_PORT"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo "  PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
echo ""

# Create logs directory
mkdir -p logs
echo "✓ Logs directory created"

# Install requirements (in case they're missing)
echo "Installing Python dependencies..."
pip install -q --no-cache-dir -r requirements.txt 2>&1 | tail -5 || true
echo "✓ Dependencies ready"
echo ""

# Start service
echo "=============================================================="
echo "Starting ML Service..."
echo "=============================================================="
exec python3 run.py
