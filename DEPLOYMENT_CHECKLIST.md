# ML Service Deployment Checklist

## Pre-Deployment (Local Development)

### Environment Setup
- [ ] Python 3.10 or higher installed
- [ ] Virtual environment created: `.venv/`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] `.env` file configured with correct values
- [ ] Models trained and saved in `models/` directory

### Code Quality
- [ ] No syntax errors: `python3 -m py_compile src/**/*.py`
- [ ] Unit tests pass: `python3 -m pytest tests/`
- [ ] Linting passes: `pylint src/`
- [ ] Type checking: `mypy src/`

### Local Testing
- [ ] Service starts without errors: `./start.sh`
- [ ] Health endpoint responds: `curl http://localhost:8001/health`
- [ ] API documentation loads: Visit `http://localhost:8001/docs`
- [ ] Models load successfully (check logs for no ERROR)
- [ ] Test API calls work

### Files Ready
- [ ] `run.py` - Service runner (created ✓)
- [ ] `start.sh` - Startup script (created ✓)
- [ ] `health-check.sh` - Health check (created ✓)
- [ ] `docker-entrypoint.sh` - Docker entry (created ✓)
- [ ] `Dockerfile` - Docker config
- [ ] `requirements.txt` - Dependencies
- [ ] `ML_DEPLOYMENT_GUIDE.md` - Documentation (created ✓)
- [ ] `ENV_CONFIGURATION.md` - Environment guide (created ✓)

### Model Files
- [ ] `models/solar_xgboost_model.pkl` exists
- [ ] `models/demand_xgboost_model.pkl` exists
- [ ] Models load within 5 seconds
- [ ] Model predictions work correctly

---

## Cloud Platform Setup

### Choose Platform
- [ ] Railway (Recommended for simplicity)
- [ ] Render (Recommended for free tier)
- [ ] Fly.io (Recommended for global)
- [ ] Other: ___________

### Platform-Specific Steps

#### For Railway:
- [ ] Create Railway account: https://railway.app
- [ ] Create project
- [ ] Connect GitHub repository
- [ ] Set root directory: `ml-service`
- [ ] Add environment variables (see below)
- [ ] Deploy

#### For Render:
- [ ] Create Render account: https://render.com
- [ ] Create Web Service
- [ ] Connect GitHub repository
- [ ] Set root directory: `ml-service`
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `python3 run.py`
- [ ] Add environment variables (see below)
- [ ] Deploy

#### For Fly.io:
- [ ] Install Fly CLI
- [ ] Run: `fly launch`
- [ ] Set secrets: `fly secrets set VARIABLE=value`
- [ ] Deploy: `fly deploy`

---

## Environment Variables

### Required (All Platforms)
```
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

### Database Connection
```
DATABASE_URL=postgresql://user:pass@host:port/database
TIMESCALE_URL=postgresql://user:pass@host:port/timescale
```

### Redis Connection
```
REDIS_URL=redis://:password@host:port
```

### Service Configuration
```
ML_HOST=0.0.0.0
ML_PORT=8001
ML_WORKERS=4
```

### Checklist
- [ ] All required variables set in platform
- [ ] Database credentials correct
- [ ] Redis connection working
- [ ] Port set to 8001 (or platform default)

---

## Pre-Deployment Testing

### Local Docker Test
```bash
cd ml-service
docker build -t solar-ml-test .
docker run -p 8001:8001 -e ENVIRONMENT=production solar-ml-test
```

- [ ] Docker build succeeds
- [ ] Container starts without errors
- [ ] Health check passes
- [ ] API responds to requests

### Cloud Staging (Optional)
- [ ] Deploy to staging environment first
- [ ] Test all endpoints
- [ ] Check logs for errors
- [ ] Verify models load
- [ ] Test with real data

---

## Production Deployment

### Deployment Steps

1. **Code Push**
   ```bash
   git add .
   git commit -m "Deploy ML service: [description]"
   git push origin main
   ```
   - [ ] Changes pushed to GitHub
   - [ ] Platform detects new commit

2. **Build Phase**
   - [ ] Dependencies install successfully
   - [ ] No compilation errors
   - [ ] Build completes within time limit

3. **Deployment Phase**
   - [ ] Service starts successfully
   - [ ] Health checks pass
   - [ ] No startup errors in logs

4. **Post-Deployment Verification**
   - [ ] Service URL is accessible
   - [ ] Health endpoint responds
   - [ ] API documentation loads
   - [ ] Models loaded correctly
   - [ ] Test predictions work

---

## Post-Deployment Verification

### Health Checks
```bash
# Check service is running
curl https://your-ml-service.railway.app/health

# Should return:
{
  "status": "healthy",
  "models": {
    "solar_xgboost": "loaded",
    "demand_xgboost": "loaded"
  }
}
```

- [ ] Health endpoint returns 200
- [ ] Status shows "healthy"
- [ ] Models show "loaded"

### API Verification
- [ ] Documentation accessible: `/docs`
- [ ] ReDoc available: `/redoc`
- [ ] Test GET request to `/health`
- [ ] Test POST request to `/predict/solar-generation`

### Log Monitoring
- [ ] Check logs for errors
- [ ] No exceptions visible
- [ ] Service processing requests
- [ ] Model predictions working

### Performance Monitoring
- [ ] Response times acceptable (< 5 seconds)
- [ ] CPU usage reasonable
- [ ] Memory usage stable
- [ ] No memory leaks

---

## Ongoing Operations

### Daily Tasks
- [ ] Check logs for errors
- [ ] Monitor response times
- [ ] Verify model accuracy
- [ ] Check resource usage

### Weekly Tasks
- [ ] Review error rates
- [ ] Check disk usage (logs)
- [ ] Verify backups
- [ ] Update documentation if needed

### Monthly Tasks
- [ ] Review model performance
- [ ] Analyze prediction accuracy
- [ ] Update models if needed
- [ ] Security audit
- [ ] Cost review

### As Needed
- [ ] Scale workers if needed
- [ ] Optimize database queries
- [ ] Update dependencies
- [ ] Fix bugs/issues
- [ ] Deploy updates

---

## Troubleshooting During Deployment

### Build Fails
- [ ] Check Python version compatibility
- [ ] Verify requirements.txt is valid
- [ ] Check for syntax errors in code
- [ ] Review build logs for details

### Service Won't Start
- [ ] Check environment variables set
- [ ] Verify database connection
- [ ] Check model files exist
- [ ] Review startup logs

### Models Won't Load
- [ ] Verify model files in repository
- [ ] Check file paths in code
- [ ] Check model file permissions
- [ ] Review error logs

### Health Check Fails
- [ ] Check service is actually running
- [ ] Verify port configuration
- [ ] Check firewall/network rules
- [ ] Review logs for startup errors

---

## Rollback Plan

### If Deployment Goes Wrong

1. **Immediate Action**
   ```bash
   # For Railway/Render: Revert to previous deployment
   # Platform shows previous versions
   ```

2. **Check Status**
   - [ ] Service is down or error 500
   - [ ] Models not loading
   - [ ] Database connection failed

3. **Revert**
   - [ ] Click "Rollback" in platform (if available)
   - [ ] OR: Revert last commit and push again
   - [ ] OR: Manually restart from previous version

4. **Investigation**
   - [ ] Check error logs
   - [ ] Identify root cause
   - [ ] Fix locally and test
   - [ ] Deploy again

---

## Success Criteria

Your ML service is production-ready when:

- [x] All files created successfully
- [ ] Service starts without errors
- [ ] Health endpoint returns 200
- [ ] Models load within 5 seconds
- [ ] API responds to predictions
- [ ] Logs show no ERROR/CRITICAL
- [ ] Performance acceptable (< 5s responses)
- [ ] Database connected
- [ ] Service survives restart
- [ ] Auto-deploys from GitHub

---

## Support Links

### Documentation
- [ML Deployment Guide](ML_DEPLOYMENT_GUIDE.md)
- [Environment Configuration](ENV_CONFIGURATION.md)
- [Backend Integration](../backend/docs/ml-integration.md)

### External Resources
- FastAPI: https://fastapi.tiangolo.com/
- Railway: https://railway.app/docs
- Render: https://render.com/docs
- Fly.io: https://fly.io/docs/

### Quick Commands

```bash
# Local start
./start.sh

# Health check
./health-check.sh

# View logs
tail -f logs/ml-service.log

# Docker build
docker build -t solar-ml .

# Docker run
docker run -p 8001:8001 solar-ml
```

---

**Template Version:** 1.0.0
**Last Updated:** January 2026

Use this checklist for each deployment to ensure nothing is missed!
