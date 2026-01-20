# ML SERVICE DEPLOYMENT GUIDE

## Overview

This guide covers deploying the Solar Sharing ML Service to production environments. The service provides machine learning models for:
- Solar generation forecasting
- Energy demand prediction
- Dynamic pricing
- Anomaly detection
- Equipment failure prediction

## Deployment Files Created

### 1. `run.py` - Main Service Runner
**Purpose:** Python entry point for the ML service
**Features:**
- Graceful signal handling (SIGINT, SIGTERM)
- Automatic model loading verification
- Comprehensive error logging
- Service lifecycle management

**Usage:**
```bash
python3 run.py
```

### 2. `start.sh` - Bash Startup Script
**Purpose:** Automated environment setup and service startup
**Features:**
- Python version verification
- Virtual environment management
- Dependency installation
- Environment configuration loading
- Logging setup

**Usage:**
```bash
chmod +x start.sh
./start.sh
```

### 3. `health-check.sh` - Health Monitoring Script
**Purpose:** Monitor service health and display diagnostics
**Features:**
- Service availability check
- Health endpoint verification
- Log tail viewer
- Error log filtering

**Usage:**
```bash
chmod +x health-check.sh
./health-check.sh
```

### 4. `docker-entrypoint.sh` - Docker Entry Point
**Purpose:** Container startup script for cloud deployment
**Features:**
- Environment variable setup
- Dependency installation in container
- Log directory creation
- PYTHONUNBUFFERED for real-time logs

**Usage:** Specified in Dockerfile

---

## Environment Requirements

### Python Version
- **Minimum:** Python 3.10
- **Recommended:** Python 3.10.x or 3.11.x

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Key packages:
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `tensorflow==2.14.0` - Neural networks
- `xgboost==2.0.1` - Gradient boosting
- `scikit-learn==1.3.2` - ML utilities
- `pandas==2.0.3` - Data processing

---

## Local Deployment

### Method 1: Using Bash Script (Recommended for Local)

```bash
cd ml-service
chmod +x start.sh
./start.sh
```

The script will:
1. Check Python version
2. Create virtual environment (if needed)
3. Install dependencies
4. Start the service

### Method 2: Manual Startup

```bash
cd ml-service

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENVIRONMENT=development
export ML_HOST=0.0.0.0
export ML_PORT=8001
export LOG_LEVEL=INFO

# Start service
python3 run.py
```

### Verification

Check service health:
```bash
# In another terminal
./health-check.sh

# Or manually:
curl http://localhost:8001/health

# View API documentation:
# Open browser to: http://localhost:8001/docs
```

---

## Cloud Deployment

### Option 1: Railway (Recommended for Simple Deployment)

#### Step 1: Connect GitHub
1. Go to https://railway.app
2. Create new project from GitHub repo
3. Select `solar-sharing-platform` repository

#### Step 2: Configure Service
In Railway dashboard:
- Set root directory: `ml-service`
- Set start command: `python3 run.py`

#### Step 3: Environment Variables
Add these variables in Railway:
```
ENVIRONMENT=production
ML_HOST=0.0.0.0
ML_PORT=8001
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

#### Step 4: Model Files
Ensure models are in repository:
- `models/solar_xgboost_model.pkl`
- `models/demand_xgboost_model.pkl`

#### Step 5: Deploy
Push to GitHub:
```bash
git add .
git commit -m "Deploy ML service to Railway"
git push origin main
```

Railway will auto-deploy. URL format: `https://ml-service-<random>.railway.app`

---

### Option 2: Render

#### Step 1: Create Service
1. Go to https://render.com
2. Create new "Web Service"
3. Connect GitHub repository

#### Step 2: Configuration
```
Name: solar-ml-service
Root Directory: ml-service
Build Command: pip install -r requirements.txt
Start Command: python3 run.py
Environment: Python 3.10
```

#### Step 3: Environment Variables
```
ENVIRONMENT=production
ML_HOST=0.0.0.0
ML_PORT=8001
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

#### Step 4: Deploy
Push changes to trigger auto-deployment.

---

### Option 3: Fly.io

#### Step 1: Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

#### Step 2: Authentication
```bash
fly auth login
```

#### Step 3: Initialize Project
```bash
cd ml-service
fly launch
```

Choose:
- Region closest to you
- Python 3.10
- Don't set up Postgres

#### Step 4: Set Secrets
```bash
fly secrets set ENVIRONMENT=production
fly secrets set LOG_LEVEL=INFO
fly secrets set PYTHONUNBUFFERED=1
```

#### Step 5: Deploy
```bash
fly deploy
```

Your service will be available at: `https://<app-name>.fly.dev`

---

## Docker Deployment

### Build Docker Image

```bash
cd ml-service
docker build -t solar-ml-service:latest .
```

### Run Container

```bash
docker run -d \
  --name solar-ml \
  -p 8001:8001 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  solar-ml-service:latest
```

### Check Logs
```bash
docker logs -f solar-ml
```

---

## Logging & Monitoring

### Log Files
Logs are stored in `logs/` directory:
- `ml-service.log` - Main application log
- JSON formatted with timestamp, level, service info

### Log Format
```json
{
  "timestamp": "2024-01-18T14:30:45.123456",
  "level": "INFO",
  "logger": "src.api.main",
  "service": "solar-ml-service",
  "version": "1.0.0",
  "environment": "production",
  "message": "Model loaded successfully"
}
```

### View Logs
```bash
# Last 100 lines
tail -100 logs/ml-service.log

# Follow real-time
tail -f logs/ml-service.log

# Filter errors only
grep ERROR logs/ml-service.log

# JSON pretty-print
cat logs/ml-service.log | python3 -m json.tool
```

### Log Levels
- `DEBUG` - Detailed debugging information
- `INFO` - General informational messages
- `WARNING` - Warning messages (recoverable issues)
- `ERROR` - Error messages (service continues)
- `CRITICAL` - Critical errors (service may stop)

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-18T14:30:45",
  "models": {
    "solar_xgboost": "loaded",
    "demand_xgboost": "loaded"
  },
  "uptime_seconds": 3600
}
```

### Documentation
- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

### Example Predictions
See `src/api/main.py` for all available endpoints:
- `/predict/solar-generation` - Solar forecast
- `/predict/consumption` - Demand forecast
- `/predict/pricing` - Dynamic pricing
- `/predict/anomalies` - Anomaly detection
- `/predict/equipment-failure` - Equipment health

---

## Troubleshooting

### Service Won't Start

**Error: "Python 3.10+ required"**
```bash
python3 --version
# Install Python 3.10 or higher
```

**Error: "ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

### Models Not Loading

**Check model files exist:**
```bash
ls -la models/
# Should show: solar_xgboost_model.pkl, demand_xgboost_model.pkl
```

**Check logs for details:**
```bash
tail -50 logs/ml-service.log | grep -i "error\|model"
```

### Service Unresponsive

**Check if service is running:**
```bash
lsof -i :8001  # Check port 8001
ps aux | grep run.py
```

**Restart service:**
```bash
# Kill existing process
pkill -f "python3 run.py"

# Start again
./start.sh
```

### High Memory Usage

**Check memory:**
```bash
# Linux
free -h
ps aux | grep run.py

# macOS
top -l 1 | grep "PhysMem"
```

**Reduce model precision or batch size in environment:**
```bash
export BATCH_SIZE=32  # Default 64
export PRECISION=float16  # Default float32
```

---

## Performance Optimization

### For Cloud Deployment

1. **Set LOG_LEVEL=WARNING** to reduce I/O
   ```bash
   export LOG_LEVEL=WARNING
   ```

2. **Increase uvicorn workers** (in run.py)
   ```python
   config = uvicorn.Config(
       workers=4,  # Adjust based on vCPU
       loop="uvloop"  # Faster event loop
   )
   ```

3. **Enable model caching**
   ```bash
   export MODEL_CACHE_SIZE=1000  # Cache predictions
   ```

4. **Monitor resource usage**
   ```bash
   # Use your platform's monitoring tools
   # Railway: Built-in metrics
   # Render: Built-in monitoring
   # Fly.io: fly status
   ```

---

## Maintenance

### Update Models

1. Train new models locally
2. Save to `models/` directory
3. Commit to git
4. Push to GitHub
5. Service auto-deploys with new models

### Database Backups

If using persistent storage:
```bash
docker cp solar-ml:/app/data/ ./backup/
```

### Rolling Updates

For zero-downtime updates:

**Railway/Render:** Automatic with deployment
**Fly.io:**
```bash
fly deploy --strategy rolling
```

---

## Production Checklist

- [ ] Python 3.10+ installed
- [ ] All dependencies in requirements.txt
- [ ] Models trained and saved in `models/` directory
- [ ] .env file configured with production values
- [ ] Logs directory created and writable
- [ ] Service starts without errors
- [ ] Health endpoint responds: `/health`
- [ ] API documentation accessible: `/docs`
- [ ] Error logging working properly
- [ ] Environment variables set in cloud platform
- [ ] Monitoring/alerting configured
- [ ] Backup strategy in place

---

## Support & Documentation

- **FastAPI:** https://fastapi.tiangolo.com/
- **Uvicorn:** https://www.uvicorn.org/
- **TensorFlow:** https://www.tensorflow.org/
- **XGBoost:** https://xgboost.readthedocs.io/

---

## Quick Start Commands

```bash
# Local development
cd ml-service
./start.sh

# Check health
./health-check.sh

# View logs
tail -f logs/ml-service.log

# Docker build & run
docker build -t solar-ml .
docker run -p 8001:8001 solar-ml

# Deploy to Railway
git push origin main  # Auto-deploys

# Deploy to Fly.io
fly deploy

# View cloud logs
fly logs  # Fly.io
railway logs  # Railway
```

---

**Last Updated:** January 2026
**Version:** 1.0.0
