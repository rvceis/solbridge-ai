# ML Service Environment Configuration Guide

## File Location
`ml-service/.env`

## Development Configuration

```bash
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Service
SERVICE_NAME=solar-ml-service
SERVICE_VERSION=1.0.0

# Server
ML_HOST=0.0.0.0
ML_PORT=8001
ML_WORKERS=4

# Database (Local PostgreSQL)
DATABASE_URL=postgresql://solar_user:solar_password@localhost:5434/solar_sharing
TIMESCALE_URL=postgresql://solar_user:solar_password@localhost:5434/timescale

# Redis (Local)
REDIS_URL=redis://localhost:6380

# Logging
LOG_DIR=logs
LOG_FILE=logs/ml-service.log

# Model Configuration
MODEL_CACHE_SIZE=100
BATCH_SIZE=64
PREDICTION_CONFIDENCE_THRESHOLD=0.7

# Feature Flags
ENABLE_ANOMALY_DETECTION=true
ENABLE_PRICING_MODEL=true
ENABLE_EQUIPMENT_FAILURE_PREDICTION=true
```

## Production Configuration

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Service
SERVICE_NAME=solar-ml-service
SERVICE_VERSION=1.0.0

# Server (Cloud Platform)
ML_HOST=0.0.0.0
ML_PORT=8001
ML_WORKERS=4

# Database (Cloud PostgreSQL)
DATABASE_URL=postgresql://user:password@cloud-db.railway.app:5432/solar_sharing
TIMESCALE_URL=postgresql://user:password@cloud-db.railway.app:5432/timescale

# Redis (Cloud Redis)
REDIS_URL=redis://:password@cloud-redis.railway.app:6379

# Logging
LOG_DIR=logs
LOG_FILE=logs/ml-service.log

# Model Configuration
MODEL_CACHE_SIZE=500
BATCH_SIZE=128
PREDICTION_CONFIDENCE_THRESHOLD=0.8

# Feature Flags
ENABLE_ANOMALY_DETECTION=true
ENABLE_PRICING_MODEL=true
ENABLE_EQUIPMENT_FAILURE_PREDICTION=true

# Monitoring (Optional)
SENTRY_DSN=https://key@sentry.io/project
DATADOG_API_KEY=your-datadog-key
```

## Configuration Parameters

### Environment Variables

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| ENVIRONMENT | development | production | Deployment environment |
| DEBUG | true | false | Enable debug mode |
| LOG_LEVEL | DEBUG | INFO | Logging verbosity |
| ML_PORT | 8001 | 8001 | Service port |
| ML_WORKERS | 4 | 4-8 | Uvicorn worker threads |

### Database Configuration

- **Development:** Local Docker PostgreSQL (port 5434)
- **Production:** Cloud-hosted PostgreSQL (Railway, AWS RDS, etc.)

Get connection string from your cloud provider:
```
postgresql://username:password@host:port/database
```

### Redis Configuration

Used for caching model predictions and session management.

- **Development:** Local Docker Redis (port 6380)
- **Production:** Cloud Redis (Railway, Redis Cloud, etc.)

Format: `redis://:password@host:port`

### Logging

**Log Level Hierarchy:**
```
DEBUG > INFO > WARNING > ERROR > CRITICAL
```

**Log Directory:** Must be writable by the application
- Local: `logs/` (auto-created)
- Cloud: Logs to stdout (captured by platform)

### Model Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| MODEL_CACHE_SIZE | 100 | Cache predictions (faster responses) |
| BATCH_SIZE | 64 | Predictions per batch (memory trade-off) |
| CONFIDENCE_THRESHOLD | 0.7 | Min confidence for predictions |

---

## Setting Environment Variables

### Local Development

**Option 1: .env file (Recommended)**
```bash
# Edit .env file in ml-service directory
cd ml-service
nano .env

# Variables are automatically loaded by settings.py
python3 run.py
```

**Option 2: Command Line**
```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export ML_PORT=8001
python3 run.py
```

**Option 3: Bash Script**
```bash
# In start.sh (already configured)
./start.sh
```

### Cloud Deployment

#### Railway
1. Go to Railway dashboard
2. Select project
3. Go to "Variables" tab
4. Add each variable:
   - ENVIRONMENT=production
   - LOG_LEVEL=INFO
   - etc.

#### Render
1. Go to Render dashboard
2. Select service
3. Go to "Environment" tab
4. Add all variables

#### Fly.io
```bash
fly secrets set ENVIRONMENT=production
fly secrets set LOG_LEVEL=INFO
fly secrets set DATABASE_URL="postgresql://..."
```

---

## Database Connection Strings

### Local Development (Docker)
```
# PostgreSQL
postgresql://solar_user:solar_password@localhost:5434/solar_sharing

# Connection from within Docker Compose
postgresql://solar_user:solar_password@postgres:5432/solar_sharing
```

### Production Examples

**Railway:**
```
postgresql://postgres:password@containers-us-west-###.railway.app:5432/railway
```

**AWS RDS:**
```
postgresql://admin:password@mydb.c9akciq32.us-east-1.rds.amazonaws.com:5432/solar_sharing
```

**Heroku:**
```
postgresql://user:password@ec2-###-compute.amazonaws.com:5432/database
```

**DigitalOcean:**
```
postgresql://user:password@db-###.db.ondigitalocean.com:25060/solar_sharing
```

---

## Logging Configuration

### Log Output

**Development:** Console + File
- Console: All levels (colorized)
- File: `logs/ml-service.log`

**Production:** File Only
- File: `logs/ml-service.log` or stdout
- Format: JSON (machine-readable)

### Log Rotation

- **Max file size:** 10MB
- **Backup count:** 10 (keeps 100MB total)
- **Auto cleanup:** Oldest files deleted when limit exceeded

### Reading Logs

```bash
# Real-time monitoring
tail -f logs/ml-service.log

# Last 100 lines
tail -100 logs/ml-service.log

# Filter by level
grep ERROR logs/ml-service.log
grep WARNING logs/ml-service.log

# JSON format (production)
cat logs/ml-service.log | python3 -m json.tool | head -50

# Statistics
echo "Total errors:" $(grep -c ERROR logs/ml-service.log)
echo "Total warnings:" $(grep -c WARNING logs/ml-service.log)
```

---

## Feature Flags

Control what features are enabled:

```bash
ENABLE_ANOMALY_DETECTION=true      # Detect abnormal patterns
ENABLE_PRICING_MODEL=true           # Dynamic pricing calculations
ENABLE_EQUIPMENT_FAILURE_PREDICTION=true  # Predict equipment health
```

Disable features to reduce resource usage if needed:
```bash
ENABLE_ANOMALY_DETECTION=false
ENABLE_PRICING_MODEL=false
```

---

## Performance Tuning

### For Limited Resources

```bash
ENVIRONMENT=production
LOG_LEVEL=WARNING           # Reduce I/O
MODEL_CACHE_SIZE=50         # Smaller cache
BATCH_SIZE=32               # Smaller batches
ML_WORKERS=2                # Fewer workers
```

### For High Traffic

```bash
MODEL_CACHE_SIZE=1000       # Large cache
BATCH_SIZE=256              # Larger batches
ML_WORKERS=8                # More workers (if CPU available)
LOG_LEVEL=INFO              # Moderate logging
```

---

## Troubleshooting Configuration Issues

### Service Won't Start

**Check .env syntax:**
```bash
# Valid
KEY=value
KEY="value with spaces"
KEY=123

# Invalid (don't do this)
KEY = value  (spaces around =)
KEY=value    (comment without #)
```

**Verify environment variables loaded:**
```bash
python3 -c "from src.config.settings import get_settings; print(get_settings())"
```

### Database Connection Failed

**Check connection string:**
```bash
# Use psql to test
psql "postgresql://user:password@host:port/database"

# Or Python
python3 -c "import psycopg2; psycopg2.connect('postgresql://...')"
```

### Log File Permission Denied

```bash
# Check permissions
ls -la logs/

# Fix permissions
chmod 777 logs/
```

### Models Not Loading

**Verify model files exist:**
```bash
ls -la models/
# Should show: solar_xgboost_model.pkl, demand_xgboost_model.pkl
```

**Check .env has correct paths:**
```bash
grep -i "model" .env
```

---

## Security Best Practices

### Sensitive Information

**Never commit to Git:**
- Passwords
- API keys
- Database URLs

**Use `.gitignore`:**
```
.env
*.log
models/
.venv/
```

**For Cloud Deployment:**
- Use platform's secrets management
- Railway: Variables tab (encrypted)
- Render: Environment variables
- Fly.io: Secrets command

### Database Credentials

**Development:** Use weak credentials locally
```
ENVIRONMENT=development
DATABASE_URL=postgresql://dev:dev@localhost:5434/solar_sharing
```

**Production:** Use strong credentials
```
ENVIRONMENT=production
DATABASE_URL=postgresql://strong_user:extremely_secure_password_123!@host:5432/db
```

---

## Quick Reference

### Start Service
```bash
cd ml-service
./start.sh
```

### Set Variable
```bash
export VAR_NAME=value
```

### View All Variables
```bash
env | grep -E "ML_|DATABASE_|REDIS_|LOG_"
```

### Check Health
```bash
./health-check.sh
```

### View Logs
```bash
tail -f logs/ml-service.log
```

---

**Last Updated:** January 2026
**Version:** 1.0.0
