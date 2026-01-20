# ML Service Integration Guide

## 📊 Data Requirements for Training Models

### 1. **Solar Forecasting Models** (LSTM + XGBoost)

#### Required Features:
Your processed CSV has these columns ✅:
- **Time features**: Year, Month, Day, Hour, Minute
- **Meteorological**: Temperature, Dew Point, Relative Humidity, Pressure, Precipitable Water
- **Solar irradiance**: GHI (Global Horizontal Irradiance), DHI (Diffuse), DNI (Direct), Clearsky variants
- **Environmental**: Cloud Type, Surface Albedo, Solar Zenith Angle

#### Target Variable:
- **Power Output (kW)**: Generate from GHI using formula: `power_kw = GHI * panel_efficiency * area / 1000`
  - Current implementation: `power_kw = GHI * 0.005` (equivalent to 5kW system)

#### Data Location:
```
/ml-service/data/processed/solar_processed_*.csv
- Shape: 8760 rows × 2035 columns
- Coverage: Full year hourly data
- Status: ✅ Ready for training
```

---

### 2. **Demand Forecasting Models** (LSTM + XGBoost)

#### Required Features:
- **Time features**: hour, day_of_week, month, is_weekend
- **Weather**: temperature, humidity
- **User profile**: household_size, has_ac, has_ev
- **Historical consumption**: avg_monthly_consumption_kwh

#### Target Variable:
- **Demand (kW)**: Actual consumption at each timestamp

#### Current Status:
- ❌ **Real demand data not available yet**
- ✅ Synthetic data generator exists in training pipeline
- 📝 **Need to collect**: User consumption history from backend DB

#### How to Get Real Demand Data:
```sql
-- Query backend PostgreSQL database
SELECT 
  user_id,
  timestamp,
  energy_consumed_kwh,
  household_size,
  temperature,
  humidity
FROM consumption_readings
WHERE timestamp >= NOW() - INTERVAL '90 days'
ORDER BY timestamp;
```

---

### 3. **Dynamic Pricing Model**

#### Required Features:
- Current supply (available energy in area)
- Current demand (buyer requests)
- Supply/demand ratio
- Base tariff rate
- Time of day
- Historical prices

#### Target Variable:
- **Optimal price per kWh**

#### Current Status:
- ✅ Trained on synthetic data
- 📝 **Need real marketplace transaction data**

---

### 4. **Risk Scoring Model** (Investor Risk Assessment)

#### Required Features:
- Financial credit score
- Solar system age (years)
- System capacity (kW)
- Number of installation quotes
- Location risk factors
- Historical performance

#### Target Variable:
- **Risk score (0-100)**: Lower = safer investment

#### Current Status:
- ✅ Trained on synthetic data
- 📝 **Need investor/project data from backend**

---

### 5. **Anomaly Detection Model**

#### Required Features:
- Real-time power output (kW)
- Voltage (V)
- Frequency (Hz)
- Current (A)
- Expected vs actual output ratio

#### Target Variable:
- **Anomaly flag (0/1)**: Unsupervised clustering

#### Current Status:
- ✅ Trained on synthetic data
- 📝 **Need real-time IoT device data**

---

### 6. **Equipment Failure Predictor**

#### Required Features:
- Temperature
- Voltage
- Current
- Runtime hours
- Days since last maintenance
- Historical failure events

#### Target Variable:
- **Failure probability (0-1)**

#### Current Status:
- ✅ Trained on synthetic data
- 📝 **Need device maintenance logs**

---

## 🧪 Testing the Models

### 1. Health Check
```bash
curl http://localhost:8001/health | jq .
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-17T...",
  "version": "1.0.0",
  "models_loaded": {
    "solar_xgboost": true,
    "solar_lstm": false,
    ...
  }
}
```

### 2. Test Solar Forecast (Single Device)
```bash
curl -X POST http://localhost:8001/api/v1/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "host_id": "SOLAR_001",
    "panel_capacity_kw": 5.0,
    "historical_data": [
      {
        "device_id": "SOLAR_001",
        "timestamp": "2026-01-17T12:00:00Z",
        "power_kw": 4.2,
        "temperature": 28.0,
        "voltage": 230.0,
        "current": 18.3,
        "frequency": 50.0,
        "system_capacity_kw": 5.0
      }
    ],
    "weather_forecast": [
      {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "temperature": 30.0,
        "humidity": 60.0,
        "wind_speed": 3.5,
        "cloud_cover": 20.0,
        "irradiance": 800.0
      }
    ],
    "forecast_hours": 24
  }' | jq .
```

### 3. Test Demand Forecast
```bash
curl -X POST http://localhost:8001/api/v1/forecast/demand \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "historical_consumption": [
      {
        "user_id": "user_123",
        "timestamp": "2026-01-17T12:00:00Z",
        "power_kw": 2.5,
        "temperature": 28.0,
        "household_size": 4
      }
    ],
    "weather_forecast": [...],
    "forecast_hours": 48
  }' | jq .
```

### 4. Test Dynamic Pricing
```bash
curl -X POST http://localhost:8001/api/v1/pricing/dynamic \
  -H "Content-Type: application/json" \
  -d '{
    "current_supply_kwh": 1000.0,
    "current_demand_kwh": 800.0,
    "base_price": 5.0,
    "time_of_day": "peak",
    "location": {
      "latitude": 12.9716,
      "longitude": 77.5946
    }
  }' | jq .
```

### 5. Test Model Inference Script
```bash
cd /home/akash/Desktop/SOlar_Sharing/ml-service
source .venv/bin/activate
python test_model_inference.py
```

---

## 🔗 Backend Integration

### Current Backend Structure
- **Backend API**: Node.js/Express on port 3000
- **ML Service**: Python/FastAPI on port 8001
- **Integration**: Backend calls ML service via HTTP

### Backend Routes Already Set Up ✅

#### 1. Solar Predictions
```javascript
// GET /api/v1/devices/:deviceId/prediction
// Calls ML service and returns solar forecast
```

#### 2. Consumption Forecast
```javascript
// GET /api/v1/users/consumption-forecast
// Returns user consumption prediction
```

#### 3. Pricing Recommendations
```javascript
// GET /api/v1/pricing/recommendation
// POST /api/v1/pricing/calculate
```

#### 4. Anomaly Detection
```javascript
// GET /api/v1/anomaly-alerts
// GET /api/v1/devices/:deviceId/health/failure
```

### Integration Steps

#### Step 1: Configure ML Service URL in Backend

Edit `/backend/src/config/index.js`:
```javascript
module.exports = {
  // ... existing config
  mlService: {
    url: process.env.ML_SERVICE_URL || 'http://localhost:8001',
    timeout: 30000, // 30 seconds
    retries: 3
  }
};
```

#### Step 2: Update Backend PredictionService

Edit `/backend/src/services/PredictionService.js`:
```javascript
const axios = require('axios');
const config = require('../config');

class PredictionService {
  async predictPanelOutput(deviceId, userId, days = 7) {
    try {
      // Get device historical data
      const readings = await this.getDeviceReadings(deviceId, 30);
      
      // Call ML service
      const response = await axios.post(
        `${config.mlService.url}/api/v1/forecast/solar`,
        {
          host_id: deviceId,
          panel_capacity_kw: 5.0,
          historical_data: readings,
          forecast_hours: days * 24
        },
        { timeout: config.mlService.timeout }
      );
      
      return response.data;
    } catch (error) {
      logger.error('ML service call failed:', error);
      // Fallback to rule-based prediction
      return this._fallbackPrediction(deviceId, days);
    }
  }
  
  async getDeviceReadings(deviceId, days) {
    const result = await db.query(`
      SELECT 
        device_id,
        time as timestamp,
        power_kw,
        temperature,
        voltage,
        current,
        frequency
      FROM energy_readings
      WHERE device_id = $1
        AND time >= NOW() - INTERVAL '${days} days'
      ORDER BY time DESC
      LIMIT 1000
    `, [deviceId]);
    
    return result.rows;
  }
}
```

#### Step 3: Add Error Handling & Fallback

```javascript
// In PredictionService.js
_fallbackPrediction(deviceId, days) {
  // Use simple rule-based prediction if ML service fails
  return {
    deviceId,
    forecasts: Array.from({ length: days }, (_, i) => ({
      date: new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000),
      predicted: 4.5, // Default average
      confidence: 0.5,
      source: 'fallback'
    }))
  };
}
```

---

## 📱 Frontend Integration

### Frontend API Service Location
`/frontend/src/api/`

### 1. Create ML Service API Client

Create `/frontend/src/api/mlService.ts`:
```typescript
import apiClient from './client';

export const mlService = {
  // Get solar forecast for device
  getSolarForecast: async (deviceId: string, days: number = 7) => {
    const response = await apiClient.get(
      `/devices/${deviceId}/prediction?days=${days}`
    );
    return response.data;
  },

  // Get consumption forecast for user
  getConsumptionForecast: async (days: number = 7) => {
    const response = await apiClient.get(
      `/users/consumption-forecast?days=${days}`
    );
    return response.data;
  },

  // Get pricing recommendation
  getPricingRecommendation: async (energyAmount: number, location: any) => {
    const response = await apiClient.get('/pricing/recommendation', {
      params: { energy_amount: energyAmount, ...location }
    });
    return response.data;
  },

  // Get anomaly alerts
  getAnomalyAlerts: async () => {
    const response = await apiClient.get('/anomaly-alerts');
    return response.data;
  },

  // Get equipment health
  getEquipmentHealth: async (deviceId: string) => {
    const response = await apiClient.get(
      `/devices/${deviceId}/health/failure`
    );
    return response.data;
  }
};

export default mlService;
```

### 2. Create React Hook for ML Data

Create `/frontend/src/hooks/useSolarForecast.ts`:
```typescript
import { useState, useEffect } from 'react';
import mlService from '../api/mlService';

export const useSolarForecast = (deviceId: string, days: number = 7) => {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoading(true);
        const data = await mlService.getSolarForecast(deviceId, days);
        setForecast(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (deviceId) {
      fetchForecast();
    }
  }, [deviceId, days]);

  return { forecast, loading, error };
};
```

### 3. Display Forecast in Dashboard

Update `/frontend/src/screens/Dashboard/DashboardScreen.tsx`:
```typescript
import React from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { useSolarForecast } from '../../hooks/useSolarForecast';
import { LineChart } from 'react-native-chart-kit';

const DashboardScreen = () => {
  const { forecast, loading, error } = useSolarForecast('DEVICE_001', 7);

  if (loading) return <ActivityIndicator size="large" />;
  if (error) return <Text>Error: {error}</Text>;

  const chartData = {
    labels: forecast?.forecasts.map(f => 
      new Date(f.date).toLocaleDateString('en-US', { weekday: 'short' })
    ),
    datasets: [{
      data: forecast?.forecasts.map(f => f.predicted) || []
    }]
  };

  return (
    <View>
      <Text style={{ fontSize: 20, fontWeight: 'bold' }}>
        7-Day Solar Forecast
      </Text>
      
      <LineChart
        data={chartData}
        width={350}
        height={220}
        chartConfig={{
          backgroundColor: '#022173',
          backgroundGradientFrom: '#1E3A8A',
          backgroundGradientTo: '#3B82F6',
          decimalPlaces: 1,
          color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
        }}
        bezier
      />

      <View style={{ marginTop: 20 }}>
        {forecast?.forecasts.map((day, idx) => (
          <View key={idx} style={{ flexDirection: 'row', padding: 10 }}>
            <Text>{new Date(day.date).toLocaleDateString()}</Text>
            <Text style={{ marginLeft: 'auto' }}>
              {day.predicted.toFixed(2)} kW
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
};
```

---

## 🚀 Complete Integration Workflow

### Phase 1: Data Collection (Week 1-2)
1. ✅ Collect solar data (DONE - CSVs available)
2. ❌ Collect user consumption data from backend DB
3. ❌ Collect marketplace transaction data
4. ❌ Collect device health/maintenance logs

### Phase 2: Model Training (Week 2-3)
1. ✅ Train Solar XGBoost on real data (DONE)
2. ⚠️ Train Solar LSTM (fix sequence preprocessing)
3. ❌ Train Demand models on real consumption data
4. ✅ Advanced models trained on synthetic data

### Phase 3: API Integration (Week 3-4)
1. ✅ ML service endpoints available
2. ❌ Update backend to call ML service
3. ❌ Add error handling & fallbacks
4. ❌ Frontend components to display predictions

### Phase 4: Testing & Validation (Week 4)
1. ❌ End-to-end testing
2. ❌ Accuracy validation
3. ❌ Performance optimization
4. ❌ User acceptance testing

---

## 📋 Quick Checklist

### For Training:
- [x] Solar processed data available
- [x] Solar XGBoost trained
- [ ] Solar LSTM trained properly
- [ ] Collect real demand data
- [ ] Train demand models
- [ ] Add model persistence for all models

### For Testing:
- [x] ML service health check works
- [ ] Test solar forecast endpoint
- [ ] Test demand forecast endpoint
- [ ] Test pricing endpoint
- [ ] Test anomaly detection
- [ ] Load testing (100+ concurrent requests)

### For Backend Integration:
- [ ] Configure ML_SERVICE_URL in backend
- [ ] Update PredictionService to call ML API
- [ ] Add fallback logic for ML failures
- [ ] Add request/response logging
- [ ] Set up monitoring/alerting

### For Frontend:
- [ ] Create mlService API client
- [ ] Create useSolarForecast hook
- [ ] Create useConsumptionForecast hook
- [ ] Update Dashboard with charts
- [ ] Add loading/error states
- [ ] Show confidence intervals

---

## 🔧 Next Steps (Priority Order)

### 1. **Fix LSTM Training** (High Priority)
```bash
cd /home/akash/Desktop/SOlar_Sharing/ml-service
# Fix timestamp columns in sequence generation
# Re-train on real data
```

### 2. **Export Real Demand Data** (High Priority)
```bash
# From backend PostgreSQL
psql -U postgres -d solar_sharing -c "
  COPY (
    SELECT * FROM consumption_readings
    WHERE timestamp >= NOW() - INTERVAL '90 days'
  ) TO '/tmp/demand_data.csv' CSV HEADER;
"

# Copy to ML service
cp /tmp/demand_data.csv /home/akash/Desktop/SOlar_Sharing/ml-service/data/processed/
```

### 3. **Backend Integration** (Medium Priority)
- Update backend/src/services/PredictionService.js
- Test ML service calls
- Add fallback logic

### 4. **Frontend Dashboard** (Medium Priority)
- Create forecast display components
- Add charts (react-native-chart-kit)
- Show real-time predictions

### 5. **Model Persistence** (Low Priority)
- Add save/load to DemandXGBoost
- Auto-load models on service startup
- Version control for models

---

## 📞 Support & Troubleshooting

### Common Issues:

**Issue 1: ML service not responding**
```bash
# Check if service is running
curl http://localhost:8001/health

# Restart service
cd /home/akash/Desktop/SOlar_Sharing/ml-service
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**Issue 2: Model predictions are incorrect**
- Check input data format
- Verify feature engineering matches training
- Check model is loaded correctly
- Review preprocessing pipeline

**Issue 3: Backend can't reach ML service**
- Verify ML_SERVICE_URL configuration
- Check firewall/network settings
- Test with curl from backend server

---

## 📚 Additional Resources

- ML Service API Documentation: `http://localhost:8001/docs`
- Training Scripts: `/ml-service/src/services/training_*.py`
- Test Scripts: `/ml-service/test_*.py`
- Backend Integration: `/backend/src/services/PredictionService.js`
- Frontend Hooks: `/frontend/src/hooks/`

---

**Last Updated**: January 17, 2026
**Version**: 1.0.0
**Status**: In Progress
