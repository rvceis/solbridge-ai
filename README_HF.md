---
title: SolBridge AI - ML Service
emoji: ☀️
colorFrom: yellow
colorTo: orange
sdk: docker
pinned: false
license: mit
---

# ☀️ SolBridge AI - ML Service

**AI-Powered Solar Energy Forecasting & Community Energy Sharing Platform**

This is the ML service powering SolBridge AI, providing:
- 🌞 **Solar Generation Forecasting** - Prophet-based time series prediction
- ⚡ **Energy Demand Forecasting** - XGBoost consumption predictions
- 🌤️ **Real-time Weather Integration** - Free Open-Meteo API
- 💰 **Dynamic Pricing** - Supply-demand optimization
- 🎯 **Risk Scoring** - Investment analysis for solar installations
- 🚨 **Anomaly Detection** - Equipment failure prediction

## 🚀 Quick Start

### API Endpoints

**Health Check:**
```bash
curl https://your-space.hf.space/health
```

**Solar Forecast (24 hours):**
```bash
curl -X POST https://your-space.hf.space/api/v1/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "host_id": "H-1",
    "panel_capacity_kw": 5,
    "forecast_hours": 24,
    "historical_data": [],
    "weather_forecast": []
  }'
```

**Demand Forecast:**
```bash
curl -X POST https://your-space.hf.space/api/v1/forecast/demand \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U-1",
    "forecast_hours": 24,
    "historical_data": [],
    "weather_forecast": []
  }'
```

**Dynamic Pricing:**
```bash
curl -X POST https://your-space.hf.space/api/v1/pricing/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-01-21T12:00:00Z",
    "total_supply_kwh": 120,
    "total_demand_kwh": 100,
    "time_of_day": "afternoon",
    "grid_tariff": 7.5
  }'
```

## 🤖 ML Models

- **Prophet** - Facebook's pre-trained time series forecaster (no training needed!)
- **XGBoost** - Gradient boosting for solar/demand forecasting
- **Random Forest** - Risk scoring and classification
- **Isolation Forest** - Anomaly detection

## 📊 Features

✅ Pre-trained models (no historical data required)  
✅ Real-time weather data from free APIs  
✅ RESTful API with FastAPI  
✅ Auto-scaling on Hugging Face infrastructure  
✅ 95%+ confidence intervals  
✅ Production-ready logging and monitoring

## 🌐 Integration

Use this ML service with your frontend application:

```javascript
// Example: Fetch solar forecast
const response = await fetch('https://your-space.hf.space/api/v1/forecast/solar', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    host_id: 'H-1',
    panel_capacity_kw: 5,
    forecast_hours: 24,
    historical_data: [],
    weather_forecast: []
  })
});
const forecast = await response.json();
console.log(forecast.predictions);
```

## 📖 API Documentation

Interactive API docs available at:
- **Swagger UI**: `https://your-space.hf.space/docs`
- **ReDoc**: `https://your-space.hf.space/redoc`

## 🛠️ Tech Stack

- **FastAPI** - Modern Python web framework
- **Prophet** - Time series forecasting by Facebook
- **XGBoost** - Gradient boosting
- **Pandas** - Data processing
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Deployed on** [Hugging Face Spaces](https://huggingface.co/spaces) 🤗
