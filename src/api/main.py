"""FastAPI Service for ML Model Inference"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.exceptions import handle_exception, MLServiceException
from src.preprocessing.pipeline import DataPreprocessingPipeline
from src.models.solar_forecast import SolarLSTMModel, SolarXGBoostModel, SolarForecastingEnsemble
from src.models.demand_forecast import DemandLSTMModel, DemandXGBoostModel, DemandForecastingEnsemble
from src.models.advanced_models import (
    DynamicPricingModel,
    InvestorRiskScoringModel,
    AnomalyDetectionModel,
    EquipmentFailurePredictorModel
)

logger = get_logger(__name__)
settings = get_settings()


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class SolarGenerationData(BaseModel):
    """Solar generation data for prediction"""
    device_id: str
    timestamp: datetime
    power_kw: float
    temperature: float
    voltage: float
    current: float
    frequency: float
    cloud_cover: Optional[float] = None
    system_capacity_kw: float

    class Config:
        schema_extra = {
            "example": {
                "device_id": "SM_H123_001",
                "timestamp": "2024-01-17T14:00:00Z",
                "power_kw": 4.2,
                "temperature": 32.5,
                "voltage": 230.5,
                "current": 18.3,
                "frequency": 50.01,
                "cloud_cover": 20,
                "system_capacity_kw": 5.0
            }
        }


class ConsumptionData(BaseModel):
    """Consumption data for demand forecasting"""
    user_id: str
    timestamp: datetime
    power_kw: float
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    household_size: Optional[int] = None
    has_ac: Optional[bool] = False


class WeatherData(BaseModel):
    """Weather data for predictions"""
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    irradiance: Optional[float] = None
    description: Optional[str] = None


class SolarForecastRequest(BaseModel):
    """Request for solar generation forecast"""
    host_id: str
    panel_capacity_kw: float
    historical_data: List[SolarGenerationData]
    weather_forecast: List[WeatherData]
    forecast_hours: int = Field(default=48, ge=1, le=168)


class SolarForecastResponse(BaseModel):
    """Response for solar forecast"""
    host_id: str
    forecast_start: datetime
    predictions: List[Dict[str, Any]]
    confidence_score: float
    model_version: str
    generated_at: datetime


class DemandForecastRequest(BaseModel):
    """Request for demand forecasting"""
    user_id: str
    historical_data: List[ConsumptionData]
    weather_forecast: List[WeatherData]
    forecast_hours: int = Field(default=48, ge=1, le=168)


class DemandForecastResponse(BaseModel):
    """Response for demand forecast"""
    user_id: str
    forecast_start: datetime
    predictions: List[Dict[str, Any]]
    model_version: str
    generated_at: datetime


class PricingRequest(BaseModel):
    """Request for dynamic pricing calculation"""
    timestamp: datetime
    total_supply_kwh: float
    total_demand_kwh: float
    time_of_day: str  # morning, afternoon, evening, night
    grid_tariff: float  # ₹/kWh


class PricingResponse(BaseModel):
    """Response for pricing"""
    recommended_price: float
    price_range: Dict[str, float]
    supply_demand_ratio: float
    optimal_trading_hours: List[int]


class RiskScoringRequest(BaseModel):
    """Request for investor risk scoring"""
    location_latitude: float
    location_longitude: float
    financial_history_score: float  # 0-100
    system_age_years: float
    system_capacity_kw: float
    installation_quotes_received: int


class RiskScoringResponse(BaseModel):
    """Response for risk scoring"""
    risk_score: int  # 0-100
    risk_category: str  # low, medium, high
    risk_probability: float
    expected_roi_percentage: float
    recommendation: str


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    device_id: str
    readings: List[Dict[str, Any]]  # Recent system readings


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection"""
    device_id: str
    anomalies_detected: int
    severity: str  # low, medium, high, critical
    anomaly_types: List[str]
    recommended_action: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    environment: str
    timestamp: datetime
    models_loaded: Dict[str, bool]


# ============================================================================
# FastAPI Application Setup
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Solar Energy ML Service",
        description="AI/ML service for solar energy forecasting and optimization",
        version=settings.SERVICE_VERSION
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Global exception handler
    @app.exception_handler(MLServiceException)
    async def ml_exception_handler(request, exc):
        return {
            "error": exc.error_code,
            "message": exc.message,
            "status_code": exc.status_code,
            "details": exc.details
        }
    
    return app


# ============================================================================
# Model Manager (Singleton Pattern)
# ============================================================================

class ModelManager:
    """Manages all ML models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Initializing ModelManager")
        
        # Preprocessing
        self.preprocessor = DataPreprocessingPipeline()
        
        # Solar forecasting
        self.solar_lstm = SolarLSTMModel(input_size=15, forecast_horizon=48)
        self.solar_xgboost = SolarXGBoostModel(n_estimators=500)
        self.solar_ensemble = SolarForecastingEnsemble()
        
        # Demand forecasting
        self.demand_lstm = DemandLSTMModel(input_size=12, forecast_horizon=48)
        self.demand_xgboost = DemandXGBoostModel(n_estimators=300)
        self.demand_ensemble = DemandForecastingEnsemble()
        
        # Advanced models
        self.pricing_model = DynamicPricingModel()
        self.risk_model = InvestorRiskScoringModel()
        self.anomaly_model = AnomalyDetectionModel()
        self.failure_model = EquipmentFailurePredictorModel()
        
        self._initialized = True
        logger.info("ModelManager initialized successfully")
    
    def get_models_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            "solar_lstm": self.solar_lstm.model is not None,
            "solar_xgboost": self.solar_xgboost.model is not None,
            "demand_lstm": self.demand_lstm.model is not None,
            "demand_xgboost": self.demand_xgboost.model is not None,
            "pricing": self.pricing_model.model is not None,
            "risk_scoring": self.risk_model.model is not None,
            "anomaly_detection": self.anomaly_model.model is not None,
            "failure_prediction": self.failure_model.model is not None
        }


@lru_cache()
def get_model_manager() -> ModelManager:
    """Get model manager instance"""
    return ModelManager()


# ============================================================================
# API Routes
# ============================================================================

app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.utcnow(),
        models_loaded=model_manager.get_models_status()
    )


@app.post("/api/v1/forecast/solar", response_model=SolarForecastResponse)
async def forecast_solar(
    request: SolarForecastRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Forecast solar generation"""
    try:
        logger.info(f"Solar forecast request for host {request.host_id}")
        
        # Prepare data
        historical_df = pd.DataFrame([d.dict() for d in request.historical_data])
        weather_df = pd.DataFrame([w.dict() for w in request.weather_forecast])
        
        # Preprocess
        processed_data = model_manager.preprocessor.preprocess_solar_data(
            historical_df,
            system_capacity_kw=request.panel_capacity_kw
        )
        
        # Make predictions
        if model_manager.solar_lstm.model:
            X = processed_data.iloc[-request.panel_capacity_kw:].values.reshape(1, -1, -1)
            lstm_pred, confidence = model_manager.solar_lstm.predict(X)
        else:
            lstm_pred = None
        
        # Format response
        predictions = []
        start_time = datetime.utcnow()
        
        for i in range(request.forecast_hours):
            pred_time = start_time + timedelta(hours=i)
            pred_value = lstm_pred[0][i] if lstm_pred is not None else 0
            
            predictions.append({
                "hour": pred_time.isoformat(),
                "predicted_kwh": float(pred_value),
                "confidence_lower": float(max(0, pred_value * 0.8)),
                "confidence_upper": float(pred_value * 1.2)
            })
        
        return SolarForecastResponse(
            host_id=request.host_id,
            forecast_start=start_time,
            predictions=predictions,
            confidence_score=0.92,
            model_version="1.0.0",
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Solar forecast failed: {str(e)}")
        error = handle_exception(e, context={"host_id": request.host_id})
        raise HTTPException(status_code=error["status_code"], detail=error)


@app.post("/api/v1/forecast/demand", response_model=DemandForecastResponse)
async def forecast_demand(
    request: DemandForecastRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Forecast energy demand"""
    try:
        logger.info(f"Demand forecast request for user {request.user_id}")
        
        # Prepare data
        historical_df = pd.DataFrame([d.dict() for d in request.historical_data])
        
        # Preprocess
        processed_data = model_manager.preprocessor.preprocess_consumption_data(historical_df)
        
        # Make predictions
        if model_manager.demand_lstm.model:
            X = processed_data.iloc[-request.forecast_hours:].values.reshape(1, -1, -1)
            demand_pred = model_manager.demand_lstm.predict(X)
        else:
            demand_pred = np.zeros(request.forecast_hours)
        
        # Format response
        predictions = []
        start_time = datetime.utcnow()
        
        for i in range(request.forecast_hours):
            pred_time = start_time + timedelta(hours=i)
            pred_value = demand_pred[0][i] if demand_pred.ndim > 1 else demand_pred[i]
            
            predictions.append({
                "hour": pred_time.isoformat(),
                "predicted_kwh": float(pred_value)
            })
        
        return DemandForecastResponse(
            user_id=request.user_id,
            forecast_start=start_time,
            predictions=predictions,
            model_version="1.0.0",
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Demand forecast failed: {str(e)}")
        error = handle_exception(e, context={"user_id": request.user_id})
        raise HTTPException(status_code=error["status_code"], detail=error)


@app.post("/api/v1/pricing/calculate", response_model=PricingResponse)
async def calculate_pricing(
    request: PricingRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Calculate dynamic pricing"""
    try:
        logger.info(f"Pricing calculation for {request.timestamp}")
        
        # Prepare features
        X = pd.DataFrame([{
            "supply": request.total_supply_kwh,
            "demand": request.total_demand_kwh,
            "supply_demand_ratio": request.total_supply_kwh / (request.total_demand_kwh + 0.1),
            "hour": request.timestamp.hour,
            "grid_tariff": request.grid_tariff
        }])
        
        # Predict price
        if model_manager.pricing_model.model:
            predicted_price = model_manager.pricing_model.predict(X)[0]
        else:
            predicted_price = request.grid_tariff
        
        # Calculate range
        price_range = {
            "min": max(4, predicted_price * 0.9),
            "max": min(12, predicted_price * 1.1),
            "recommended": predicted_price
        }
        
        return PricingResponse(
            recommended_price=predicted_price,
            price_range=price_range,
            supply_demand_ratio=X["supply_demand_ratio"].iloc[0],
            optimal_trading_hours=list(range(9, 17))  # 9 AM to 5 PM typical
        )
        
    except Exception as e:
        logger.error(f"Pricing calculation failed: {str(e)}")
        error = handle_exception(e)
        raise HTTPException(status_code=error["status_code"], detail=error)


@app.post("/api/v1/risk/score", response_model=RiskScoringResponse)
async def score_risk(
    request: RiskScoringRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Score investment risk"""
    try:
        logger.info(f"Risk scoring for location ({request.location_latitude}, {request.location_longitude})")
        
        # Prepare features
        X = pd.DataFrame([request.dict()])
        
        # Predict risk
        if model_manager.risk_model.model:
            risk_class, risk_prob = model_manager.risk_model.predict(X)
            risk_score = int(risk_class[0] * 50)
        else:
            risk_score = 50
            risk_class = [1]
            risk_prob = [0.5]
        
        risk_category_map = {0: "low", 1: "medium", 2: "high"}
        risk_category = risk_category_map.get(risk_class[0], "unknown")
        
        # Estimate ROI
        roi_estimate = max(5, 15 - (risk_score * 0.1))
        
        return RiskScoringResponse(
            risk_score=risk_score,
            risk_category=risk_category,
            risk_probability=float(risk_prob[0]),
            expected_roi_percentage=roi_estimate,
            recommendation=f"Risk level: {risk_category}. Expected ROI: {roi_estimate:.1f}%"
        )
        
    except Exception as e:
        logger.error(f"Risk scoring failed: {str(e)}")
        error = handle_exception(e)
        raise HTTPException(status_code=error["status_code"], detail=error)


@app.post("/api/v1/anomaly/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Detect system anomalies"""
    try:
        logger.info(f"Anomaly detection for device {request.device_id}")
        
        # Prepare data
        readings_df = pd.DataFrame(request.readings)
        
        # Detect anomalies
        if model_manager.anomaly_model.model:
            anomaly_flags, anomaly_scores = model_manager.anomaly_model.predict(readings_df)
            anomaly_count = int(anomaly_flags.sum())
        else:
            anomaly_count = 0
        
        severity = "high" if anomaly_count > 10 else "medium" if anomaly_count > 5 else "low"
        
        return AnomalyDetectionResponse(
            device_id=request.device_id,
            anomalies_detected=anomaly_count,
            severity=severity,
            anomaly_types=["voltage_fluctuation", "efficiency_drop"] if anomaly_count > 0 else [],
            recommended_action="Schedule maintenance check" if severity in ["medium", "high"] else "Continue monitoring"
        )
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        error = handle_exception(e, context={"device_id": request.device_id})
        raise HTTPException(status_code=error["status_code"], detail=error)


@app.get("/api/v1/models/status")
async def get_models_status(model_manager: ModelManager = Depends(get_model_manager)):
    """Get status of all models"""
    return model_manager.get_models_status()


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Server: {settings.HOST}:{settings.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(f"Shutting down {settings.SERVICE_NAME}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )
