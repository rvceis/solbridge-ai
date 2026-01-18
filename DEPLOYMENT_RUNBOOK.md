# Solar Sharing AIML Deployment & Operations Runbook

## Table of Contents
1. [Data Intake Workflow](#data-intake-workflow)
2. [Model Training Pipeline](#model-training-pipeline)
3. [MQTT Broker Setup](#mqtt-broker-setup)
4. [Service Deployment](#service-deployment)
5. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
6. [Retraining Cadence](#retraining-cadence)
7. [Rollback Procedures](#rollback-procedures)

---

## Data Intake Workflow

### 1. Prepare Raw Data

Raw datasets should be placed in `ml-service/data/raw/`:

```bash
# Solar data (NSRDB format)
ml-service/data/raw/solar_site_1_2024.csv
ml-service/data/raw/solar_site_2_2024.csv

# Consumption data (meter format)
ml-service/data/raw/meter_consumption_2024.csv

# Weather data
ml-service/data/raw/weather_2024.csv
```

**Expected formats:**

- **NSRDB Solar**: Year, Month, Day, Hour, Minute, GHI, DNI, DHI, Temperature, Humidity, WindSpeed, ... (Note: First 2 rows are metadata)
- **Meter Consumption**: timestamp, active_power_kw, reactive_power_kvar, voltage_v, current_a, frequency_hz, ...
- **Weather**: timestamp, temperature_c, humidity_percent, wind_speed_ms, pressure_mb, cloud_cover_percent, ...

### 2. Run Batch Preprocessing

```bash
cd /home/akash/Desktop/SOlar_Sharing

# Single-threaded processing with validation
python3 ml-service/scripts/batch_preprocess.py \
  --input-dir ml-service/data/raw \
  --output-dir ml-service/data/processed \
  --capacity 5.0 \
  --workers 1

# Multi-threaded (faster) without strict validation
python3 ml-service/scripts/batch_preprocess.py \
  --input-dir ml-service/data/raw \
  --output-dir ml-service/data/processed \
  --capacity 5.0 \
  --workers 4 \
  --no-validate
```

**Output:**
- Processed files appear in `ml-service/data/processed/`
- File naming: `{source}_processed_{timestamp}_{original_name}.csv`
- Logs show rows removed due to validation failures

### 3. Verify Processed Data

```bash
# List processed files
ls -lh ml-service/data/processed/

# Quick inspection
head -20 ml-service/data/processed/solar_processed_*.csv
```

**Expected outputs:**
- Solar: Year, Month, Day, Hour, Minute, GHI, DNI, DHI, + 30+ engineered features
- Meter: timestamp, active_power_kw, + engineered features
- All NaN/outliers removed; 80-100% of input rows retained typically

---

## Model Training Pipeline

### 1. Start MLflow Tracking Server (if not running in Docker)

```bash
cd ml-service

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000 &

# Access at: http://localhost:5000
```

### 2. Run Training Service

```bash
cd /home/akash/Desktop/SOlar_Sharing

# Train all models (LSTM, XGBoost, Risk, Anomaly)
python3 ml-service/src/services/training_service.py

# Or train specific models
python3 -c "
from ml_service.src.services.training_service import SolarGenerationTrainer
import pandas as pd

df = pd.read_csv('ml-service/data/processed/solar_processed_*.csv')
trainer = SolarGenerationTrainer()
lstm_result = trainer.train_lstm(df, epochs=50, batch_size=32)
xgb_result = trainer.train_xgboost(df)
"
```

**Outputs:**
- Models saved to `ml-service/models/`
- Metrics logged to MLflow at http://localhost:5000
- Artifacts (model files) persisted in MLflow backend

### 3. Evaluate Models

```bash
python3 -c "
from ml_service.src.services.evaluation_service import ModelEvaluator, ModelSelector
import json

# Load training results from MLflow or disk
model_results = {
    'LSTM': {'metrics': {'MAE': 50, 'RMSE': 75, 'MAPE': 0.15}},
    'XGBoost': {'metrics': {'MAE': 45, 'RMSE': 70, 'MAPE': 0.12}}
}

# Compare
evaluator = ModelEvaluator()
comparison = evaluator.compare_models(model_results)
print(comparison)

# Select best
selector = ModelSelector()
best_model, best_result = selector.select_best_regression_model(model_results)
print(f'Best model: {best_model}')

# Generate report
report = evaluator.generate_report(model_results, 'evaluation_report.txt')
"
```

### 4. Register Models to MLflow

```bash
python3 -c "
from ml_service.src.services.evaluation_service import MLflowHelper

# Register best LSTM model to registry (moves from exp runs to model registry)
MLflowHelper.register_model_to_registry(
    run_id='run_uuid_here',
    model_uri='runs:/run_uuid_here/lstm_model',
    model_name='solar_generation_lstm',
    stage='Staging'
)

# Transition to Production after validation
# (done manually in MLflow UI or via client.transition_model_version_stage())
"
```

---

## MQTT Broker Setup

### 1. Start Mosquitto Broker

**Option A: System Service**
```bash
# On Linux
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Check status
sudo systemctl status mosquitto

# View logs
sudo journalctl -u mosquitto -f
```

**Option B: Docker**
```bash
# Use the backend docker-compose (if configured)
docker-compose -f backend/docker-compose.yml up mosquitto

# Or standalone
docker run -d --name mosquitto -p 1883:1883 -p 8883:8883 eclipse-mosquitto
```

### 2. Configure Backend

Update `backend/.env`:
```bash
MQTT_URL=mqtt://localhost:1883
ML_SERVICE_URL=http://ml-service:8000
REDIS_URL=redis://localhost:6380
```

### 3. Test MQTT Connectivity

```bash
# Subscribe to device data topic (terminal 1)
mosquitto_sub -h localhost -p 1883 -t 'solar/+/data'

# Subscribe to forecasts (terminal 2)
mosquitto_sub -h localhost -p 1883 -t 'solar/+/forecast'

# Publish test device data (terminal 3)
mosquitto_pub -h localhost -p 1883 -t 'solar/device_01/data' \
  -m '{"ghi":500,"temperature":25,"timestamp":"2024-01-17T12:00:00Z"}'

# Should see forecast response in terminal 2 after backend processes it
```

### 4. Device Registration

```bash
# Register a new solar device
curl -X POST http://localhost:3000/api/iot/devices/register \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "solar_01",
    "device_type": "solar_inverter",
    "location": {"lat": 37.7749, "lng": -122.4194},
    "system_capacity_kw": 5.0,
    "mqtt_topic": "solar/solar_01/data"
  }'

# Get device details
curl http://localhost:3000/api/iot/devices/solar_01

# List all devices
curl http://localhost:3000/api/iot/devices
```

---

## Service Deployment

### 1. Start ML Service (Docker Compose)

```bash
cd ml-service

# Build and start
docker-compose up --build -d

# Check logs
docker-compose logs -f ml-service

# Verify health
curl http://localhost:8000/health
```

### 2. Start Backend (Docker Compose)

```bash
cd backend

# Start
docker-compose up -d

# Check logs
docker-compose logs -f nodejs-backend

# Verify health
curl http://localhost:3000/health
```

### 3. Full Stack (from root)

```bash
# Start all services (backend + ML)
docker-compose -f backend/docker-compose.yml -f ml-service/docker-compose.yml up -d

# Verify all services
docker ps
```

### 4. Integration Check

```bash
# 1. Check ML service health
curl http://localhost:8000/health

# 2. Check backend health
curl http://localhost:3000/health

# 3. Check backend IoT/ML integration
curl http://localhost:3000/api/iot/health

# 4. Submit sample prediction request
curl -X POST http://localhost:8000/api/v1/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "ghi": 500,
    "temperature": 25,
    "hour": 12,
    "system_capacity_kw": 5.0
  }'

# Expected response:
# {
#   "prediction": 3.5,
#   "model": "solar_lstm",
#   "confidence": 0.92,
#   "timestamp": "2024-01-17T12:00:00Z"
# }
```

---

## Monitoring & Troubleshooting

### ML Service Logs

```bash
# Real-time logs
docker logs -f ml-service

# Last 100 lines
docker logs --tail 100 ml-service

# Look for errors
docker logs ml-service | grep ERROR
```

### Backend Logs

```bash
# Real-time logs
docker logs -f nodejs-backend

# Check IoT manager startup
docker logs nodejs-backend | grep -i "iot\|mqtt"
```

### MQTT Troubleshooting

```bash
# Check mosquitto is running
pgrep mosquitto

# Verify port listening
netstat -an | grep 1883

# Test connectivity
mosquitto_pub -h localhost -p 1883 -t test -m "hello" && echo "✓ MQTT OK"

# View subscription topics
# (Use mosquitto_sub with verbose flag)
mosquitto_sub -h localhost -p 1883 -t '$SYS/#' -v
```

### Redis Troubleshooting

```bash
# Connect to Redis
redis-cli -p 6380

# Check keys
redis-cli -p 6380 KEYS '*'

# Monitor real-time commands
redis-cli -p 6380 MONITOR
```

### Common Issues

| Issue | Symptoms | Fix |
|-------|----------|-----|
| **MQTT connection failed** | Backend logs: "MQTT error" every 5s | Check `MQTT_URL` in `.env`; verify mosquitto is running; check firewall |
| **ML service timeout** | Backend logs: "ML prediction timeout" | Increase timeout in `mlClient.js`; check ML service logs; verify GPU/memory |
| **Data validation errors** | Preprocessing fails on schema check | Check column names match NSRDB_SCHEMA; use `--no-validate` flag; inspect raw CSV |
| **Out of memory** | OOM kill observed; services restart | Reduce batch size in training; disable validation; use sparse data format |
| **Model predictions NaN** | Forecast returns `null` or `NaN` | Check input features for missing values; verify model serialization; retrain if corrupted |

### Health Check Dashboard

```bash
# Create a simple monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
  echo "=== $(date) ==="
  echo "ML Service: $(curl -s http://localhost:8000/health | jq -r .status // 'UNKNOWN')"
  echo "Backend: $(curl -s http://localhost:3000/health | jq -r .status // 'UNKNOWN')"
  echo "MQTT: $(pgrep mosquitto > /dev/null && echo 'Running' || echo 'Stopped')"
  echo "Devices: $(curl -s http://localhost:3000/api/iot/devices | jq '.length')"
  echo ""
  sleep 30
done
EOF

chmod +x monitor.sh
./monitor.sh
```

---

## Retraining Cadence

### Weekly Retraining

```bash
#!/bin/bash
# ml-service/scripts/weekly_retrain.sh

set -e

echo "Starting weekly model retraining at $(date)"

# 1. Collect last week's data from MQTT/meters into data/raw/
# (This step depends on your data collection infrastructure)
# Example: Download from data warehouse or pull from live sources

# 2. Preprocess
python3 ml-service/scripts/batch_preprocess.py \
  --input-dir ml-service/data/raw \
  --output-dir ml-service/data/processed \
  --workers 4

# 3. Train models
python3 ml-service/src/services/training_service.py

# 4. Evaluate and register
python3 ml-service/scripts/evaluate_and_register.py

# 5. Optional: Notify team
# curl -X POST https://hooks.slack.com/... -d "Models retrained successfully"

echo "Retraining complete at $(date)"
```

**Schedule with cron:**
```bash
crontab -e

# Add line:
0 2 * * 0 cd /home/akash/Desktop/SOlar_Sharing && bash ml-service/scripts/weekly_retrain.sh
```

### Model Refresh Decision Tree

```
Is new data available?
  ├─ YES: Retrain
  │   ├─ New model MAE < Old model MAE by >5%?
  │   │   ├─ YES: Deploy to Staging, run A/B test
  │   │   └─ NO: Keep current Production
  │   └─ New model has data drift detected?
  │       ├─ YES: Alert data science team
  │       └─ NO: Continue normal rotation
  └─ NO: Skip retraining, monitor drift metrics
```

---

## Rollback Procedures

### Rollback Model to Previous Version

```bash
# 1. List available versions in MLflow
mlflow models list --registry-uri http://localhost:5000

# 2. Get version numbers
mlflow models versions.list solar_generation_lstm

# 3. Transition previous version back to Production
python3 -c "
import mlflow

client = mlflow.tracking.MlflowClient(tracking_uri='http://localhost:5000')

# Transition version 2 to Production (was v3)
client.transition_model_version_stage(
    name='solar_generation_lstm',
    version='2',
    stage='Production',
    archive_existing_versions=True
)
"

# 4. Restart backend to load new model
docker restart nodejs-backend

# 5. Verify with test request
curl -X POST http://localhost:8000/api/v1/forecast/solar \
  -d '{"ghi": 500, "temperature": 25, "hour": 12, "system_capacity_kw": 5.0}' \
  | jq .
```

### Rollback Service Version

```bash
# 1. Check current image version
docker inspect ml-service | grep Image

# 2. Rollback to previous tag (assuming image versioning)
docker pull ml-service:v1.2.0

# 3. Update docker-compose.yml
sed -i 's/image: ml-service:v1.2.1/image: ml-service:v1.2.0/' ml-service/docker-compose.yml

# 4. Restart
docker-compose -f ml-service/docker-compose.yml up -d

# 5. Verify
docker logs -f ml-service
```

### Rollback MQTT Configuration

```bash
# If MQTT broker config changed:

# 1. Stop backend (prevent stale connections)
docker-compose stop nodejs-backend

# 2. Restart mosquitto with previous config
docker restart mosquitto
# Or: sudo systemctl restart mosquitto

# 3. Restart backend
docker-compose start nodejs-backend

# 4. Monitor logs for successful reconnection
docker logs -f nodejs-backend | grep -i "mqtt\|connected"
```

---

## Disaster Recovery

### Data Loss (Processed Data)

```bash
# Recreate processed data from raw
python3 ml-service/scripts/batch_preprocess.py \
  --input-dir ml-service/data/raw \
  --output-dir ml-service/data/processed \
  --no-validate

# Re-train models
python3 ml-service/src/services/training_service.py
```

### Model Registry Corruption

```bash
# Export current models to disk
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
for model in ['solar_generation_lstm', 'demand_forecast', 'risk_scoring']:
    versions = client.search_model_versions(f\"name='{model}'\")
    for v in versions:
        print(f'{model} v{v.version}: {v.status}')
"

# Restore from backup (if available)
# Or rebuild from training data
```

### Complete Service Restart

```bash
# 1. Stop all services
docker-compose -f backend/docker-compose.yml -f ml-service/docker-compose.yml down

# 2. Verify all stopped
docker ps

# 3. Clean volumes (WARNING: data loss if no backup)
# docker volume prune -f

# 4. Start fresh
docker-compose -f backend/docker-compose.yml -f ml-service/docker-compose.yml up -d

# 5. Verify health
curl http://localhost:3000/health && echo "✓ Backend"
curl http://localhost:8000/health && echo "✓ ML Service"
```

---

## Performance Tuning

### ML Service

```bash
# Edit ml-service/docker-compose.yml

services:
  ml-service:
    environment:
      ML_WORKERS: 4           # Increase workers for throughput
      ML_BATCH_SIZE: 64       # Larger batches = more throughput, more latency
      REDIS_MAX_CONNECTIONS: 100
```

### Backend

```bash
# backend/.env

NODE_ENV=production
NODE_OPTIONS="--max-old-space-size=2048"  # Increase if OOM
NODE_WORKERS=4                              # Worker processes
```

### Database (PostgreSQL)

```sql
-- Increase memory for query planning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';

-- Restart
docker-compose restart timescaledb
```

---

## Support & Escalation

**Issues to escalate:**
- MQTT broker persistent connection errors
- ML model predictions NaN/invalid
- Data validation failures on >50% of rows
- OOM or CPU throttling
- Redis connection pool exhaustion

**Contact:**
- ML Ops: ml-ops@company.com
- Data Science: ds@company.com
- DevOps: devops@company.com

---

**Last Updated:** 2024-01-17  
**Version:** 1.0  
**Owner:** ML Engineering Team
