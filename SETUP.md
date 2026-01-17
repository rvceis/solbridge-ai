# ML Service Setup Guide

Complete step-by-step setup instructions for the Solar Energy ML Service.

## Prerequisites

- Docker & Docker Compose (latest version)
- Python 3.10+ (for local development)
- 8GB+ RAM (minimum for training)
- 20GB+ disk space (for models and data)

## Quick Start (Docker)

### 1. Start Services

```bash
cd /home/akash/Desktop/SOlar_Sharing/ml-service
docker-compose up -d
```

### 2. Verify Services

```bash
# Wait 30 seconds for services to start
sleep 30

# Check health
curl http://localhost:8001/health

# Check logs
docker-compose logs ml-service
```

### 3. Access Services

- **ML Service API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Grafana**: http://localhost:3000 (admin:admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

## Data Preparation

### Step 1: Generate Synthetic Data

```bash
# Generate 10,000 samples each for solar, consumption, and weather
python scripts/generate_synthetic_data.py --samples 10000

# Output files:
# - data/raw/solar_synthetic.csv (solar generation)
# - data/raw/consumption_synthetic.csv (consumption)
# - data/raw/weather_synthetic.csv (weather data)
```

### Step 2: Preprocess Data

```bash
# Preprocess solar data
python scripts/preprocess_data.py \
  --input data/raw/solar_synthetic.csv \
  --output data/processed/solar_processed.csv \
  --type solar \
  --system-capacity 5.0 \
  --validate

# Preprocess consumption data
python scripts/preprocess_data.py \
  --input data/raw/consumption_synthetic.csv \
  --output data/processed/consumption_processed.csv \
  --type consumption
```

### Step 3: Verify Preprocessing

Check the processed files:

```bash
ls -lah data/processed/

# Should see:
# - solar_processed.csv (~2 MB)
# - consumption_processed.csv (~2 MB)
```

## Model Training

### 1. Prepare Training Script

Create `scripts/train_models.py`:

```python
#!/usr/bin/env python3
"""Train all models"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.solar_forecast import SolarLSTMModel, SolarXGBoostModel
from src.models.demand_forecast import DemandLSTMModel, DemandXGBoostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_solar_models():
    """Train solar forecasting models"""
    logger.info("Loading solar data")
    data = pd.read_csv("data/processed/solar_processed.csv")
    
    # Get feature columns (exclude target if present)
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'device_id', 'power_kw']]
    target_col = 'power_kw'
    
    X = data[feature_cols].fillna(0).values
    y = data[target_col].fillna(0).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Reshape for LSTM (samples, timesteps, features)
    lookback = 48
    X_train_lstm = np.array([X_train[i:i+lookback] for i in range(len(X_train)-lookback)])
    y_train_lstm = np.array([y_train[i+lookback] for i in range(len(y_train)-lookback)])
    
    # Train LSTM
    logger.info("Training Solar LSTM")
    lstm_model = SolarLSTMModel(input_size=X.shape[1], forecast_horizon=48)
    lstm_model.build_model(lookback_hours=lookback)
    lstm_model.train(X_train_lstm, y_train_lstm.reshape(-1, 1), 
                     X_train_lstm[:100], y_train_lstm[:100].reshape(-1, 1),
                     epochs=10, batch_size=32)
    lstm_model.save("models/solar_lstm.h5")
    logger.info("Solar LSTM saved")
    
    # Train XGBoost
    logger.info("Training Solar XGBoost")
    xgb_model = SolarXGBoostModel()
    xgb_model.train(pd.DataFrame(X_train), pd.Series(y_train))
    xgb_model.save("models/solar_xgboost.bin")
    logger.info("Solar XGBoost saved")

def train_demand_models():
    """Train demand forecasting models"""
    logger.info("Loading consumption data")
    data = pd.read_csv("data/processed/consumption_processed.csv")
    
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'user_id', 'power_kw']]
    target_col = 'power_kw'
    
    X = data[feature_cols].fillna(0).values
    y = data[target_col].fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    logger.info("Training Demand XGBoost")
    xgb_model = DemandXGBoostModel()
    xgb_model.train(pd.DataFrame(X_train), pd.Series(y_train))
    xgb_model.save("models/demand_xgboost.bin")
    logger.info("Demand XGBoost saved")

if __name__ == "__main__":
    train_solar_models()
    train_demand_models()
    logger.info("Model training complete")
```

### 2. Run Training

```bash
# Create models directory
mkdir -p models

# Run training
python scripts/train_models.py

# This will take 5-10 minutes depending on data size
```

### 3. Verify Models

```bash
ls -lah models/

# Should see:
# - solar_lstm.h5
# - solar_xgboost.bin
# - demand_xgboost.bin
```

## Backend Integration

Once ML service is running, integrate with main backend:

### 1. Update Backend Configuration

In backend `.env`:

```env
ML_SERVICE_URL=http://localhost:8001
ML_SERVICE_API_PREFIX=/api/v1
ML_REQUEST_TIMEOUT=30
ML_ENABLE_CACHING=true
```

### 2. Create Backend Client

Create `backend/src/services/MLServiceClient.js`:

```javascript
const axios = require('axios');
const logger = require('../utils/logger');

class MLServiceClient {
  constructor(baseURL = process.env.ML_SERVICE_URL || 'http://localhost:8001') {
    this.client = axios.create({
      baseURL: `${baseURL}${process.env.ML_SERVICE_API_PREFIX || '/api/v1'}`,
      timeout: parseInt(process.env.ML_REQUEST_TIMEOUT || 30000)
    });
  }

  async forecastSolar(hostId, panelCapacity, historicalData, weatherForecast) {
    try {
      const response = await this.client.post('/forecast/solar', {
        host_id: hostId,
        panel_capacity_kw: panelCapacity,
        historical_data: historicalData,
        weather_forecast: weatherForecast,
        forecast_hours: 48
      });
      return response.data;
    } catch (error) {
      logger.error('Solar forecast failed:', error);
      throw error;
    }
  }

  async forecastDemand(userId, historicalData, weatherForecast) {
    try {
      const response = await this.client.post('/forecast/demand', {
        user_id: userId,
        historical_data: historicalData,
        weather_forecast: weatherForecast,
        forecast_hours: 48
      });
      return response.data;
    } catch (error) {
      logger.error('Demand forecast failed:', error);
      throw error;
    }
  }

  async calculatePricing(supply, demand, gridTariff, timeOfDay) {
    try {
      const response = await this.client.post('/pricing/calculate', {
        timestamp: new Date(),
        total_supply_kwh: supply,
        total_demand_kwh: demand,
        grid_tariff: gridTariff,
        time_of_day: timeOfDay
      });
      return response.data;
    } catch (error) {
      logger.error('Pricing calculation failed:', error);
      throw error;
    }
  }

  async scoreRisk(latitude, longitude, financialScore, systemAge, capacity) {
    try {
      const response = await this.client.post('/risk/score', {
        location_latitude: latitude,
        location_longitude: longitude,
        financial_history_score: financialScore,
        system_age_years: systemAge,
        system_capacity_kw: capacity,
        installation_quotes_received: 0
      });
      return response.data;
    } catch (error) {
      logger.error('Risk scoring failed:', error);
      throw error;
    }
  }

  async detectAnomalies(deviceId, readings) {
    try {
      const response = await this.client.post('/anomaly/detect', {
        device_id: deviceId,
        readings: readings
      });
      return response.data;
    } catch (error) {
      logger.error('Anomaly detection failed:', error);
      throw error;
    }
  }

  async getHealth() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      logger.error('Health check failed:', error);
      return { status: 'unhealthy' };
    }
  }
}

module.exports = new MLServiceClient();
```

### 3. Use in Backend Routes

```javascript
const mlClient = require('../services/MLServiceClient');

router.get('/api/v1/insights/solar-forecast/:hostId', async (req, res) => {
  try {
    // Get historical data
    const historicalData = await getHistoricalData(req.params.hostId);
    const weatherForecast = await getWeatherForecast();

    // Call ML service
    const forecast = await mlClient.forecastSolar(
      req.params.hostId,
      5.0,  // panel capacity
      historicalData,
      weatherForecast
    );

    res.json(forecast);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## Troubleshooting

### Issue: Services not starting

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Full reset
docker-compose down -v
docker-compose up -d
```

### Issue: ML service unhealthy

```bash
# Check service logs
docker-compose logs ml-service

# Verify Python dependencies
docker-compose exec ml-service pip list

# Check model status
curl http://localhost:8001/api/v1/models/status
```

### Issue: Database connection failed

```bash
# Check PostgreSQL
docker-compose exec postgres psql -U solar_user -d solar_sharing -c "\dt"

# Check Redis
docker-compose exec redis redis-cli ping
```

### Issue: Out of memory during training

Reduce batch size and samples:

```bash
python scripts/train_models.py --batch-size 16 --samples 5000
```

## Performance Monitoring

### View Metrics in Prometheus

```
http://localhost:9090/graph
```

Key metrics to monitor:
- `ml_prediction_latency_ms`
- `ml_prediction_errors_total`
- `ml_model_accuracy`
- `api_request_duration_seconds`

### View Dashboards in Grafana

```
http://localhost:3000
```

Default dashboard shows:
- Model performance trends
- API latency
- System resources
- Error rates

## Production Deployment

### 1. Build Production Image

```bash
docker build -f Dockerfile.prod -t solar-ml-service:production .
```

### 2. Push to Registry

```bash
docker tag solar-ml-service:production registry.example.com/solar-ml-service:production
docker push registry.example.com/solar-ml-service:production
```

### 3. Deploy to Kubernetes (Optional)

```bash
kubectl apply -f k8s/ml-service-deployment.yaml
```

## Next Steps

1. **Provide Real Data**: Replace synthetic data with your NREL/Pecan Street data
2. **Retrain Models**: Use real data for production-quality predictions
3. **Performance Tuning**: Optimize hyperparameters for your specific use case
4. **Integration Testing**: Test all backend-ML service integrations
5. **Load Testing**: Verify performance under expected load
6. **Production Deployment**: Deploy to production infrastructure

## Support & Documentation

- **API Docs**: http://localhost:8001/docs (Swagger UI)
- **Logs**: `ml-service/logs/ml-service.log`
- **Models**: `ml-service/models/`
- **Data**: `ml-service/data/`
