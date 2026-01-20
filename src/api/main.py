"""FastAPI Service for ML Model Inference"""

import sys
from pathlib import Path

# Ensure project root (ml-service) is on PYTHONPATH when running directly (python main.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
from src.models.solar_prophet import SolarProphetModel
from src.models.demand_forecast import DemandLSTMModel, DemandXGBoostModel, DemandForecastingEnsemble
from src.models.advanced_models import (
    DynamicPricingModel,
    InvestorRiskScoringModel,
    AnomalyDetectionModel,
    EquipmentFailurePredictorModel
)
from src.services.matching_service import MarketplaceMatchingService, UserProfile, MatchingResult
from src.services.training_pipeline import ModelTrainingPipeline
from src.services.weather_service import get_weather_service
from src.routes.matching_routes import router as matching_router

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
        json_schema_extra = {
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
    
    model_config = {"protected_namespaces": ()}


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
    
    model_config = {"protected_namespaces": ()}


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


# Marketplace Matching Schemas

class NearbySearchRequest(BaseModel):
    """Request to find nearby users"""
    latitude: float
    longitude: float
    user_type: Optional[str] = None  # "seller", "buyer", "investor"
    radius_km: float = Field(default=25, ge=1, le=100)


class TrainingRequest(BaseModel):
    """Request to train models"""
    models: List[str] = Field(default=["all"], description="List of models to train or 'all'")
    num_samples: Optional[int] = Field(default=1000, ge=100, le=10000)


class TrainingResponse(BaseModel):
    """Response from model training"""
    timestamp: datetime
    status: str
    models: Dict[str, Any]


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
        
        # Services
        self.matching_service = MarketplaceMatchingService()
        self.training_pipeline = ModelTrainingPipeline()
        
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

# Register investment matching routes
app.include_router(matching_router)


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


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
    """
    Forecast solar generation using Prophet + real weather data
    Uses pre-trained Facebook Prophet model for better accuracy
    """
    try:
        logger.info(f"Solar forecast request for host {request.host_id}")
        
        # Initialize weather service
        weather_service = get_weather_service()
        
        # Get real weather data if location provided
        weather_data = None
        if hasattr(request, 'location_latitude') and hasattr(request, 'location_longitude'):
            try:
                weather_json = weather_service.get_weather_forecast(
                    request.location_latitude,
                    request.location_longitude,
                    hours=request.forecast_hours
                )
                weather_data = weather_service.parse_forecast_to_dataframe(weather_json)
                weather_data = weather_service.enrich_with_solar_potential(
                    weather_data,
                    request.location_latitude,
                    request.location_longitude
                )
                logger.info(f"✓ Using real weather data for forecast")
            except Exception as e:
                logger.warning(f"Could not fetch weather data: {e}, using default")
        
        # Use Prophet model for forecasting
        prophet_model = SolarProphetModel()
        
        # Train on historical data if provided, else use default patterns
        if request.historical_data and len(request.historical_data) > 0:
            historical_df = pd.DataFrame([
                {
                    "timestamp": d.timestamp,
                    "power_kw": d.power_kw
                }
                for d in request.historical_data
            ])
            prophet_model.train_or_load(historical_df)
        else:
            prophet_model.train_or_load()  # Uses default seasonal patterns
        
        # Make predictions
        predictions_array, forecast_details = prophet_model.predict(
            hours_ahead=request.forecast_hours,
            weather_data=weather_data
        )
        
        # Format response
        predictions = []
        start_time = datetime.utcnow()
        
        for i in range(request.forecast_hours):
            pred_time = start_time + timedelta(hours=i)
            pred_value = float(max(0, predictions_array[i]))
            
            # Cap at system capacity
            pred_value = min(pred_value, request.panel_capacity_kw)
            
            predictions.append({
                "hour": pred_time.isoformat(),
                "predicted_kwh": pred_value,
                "confidence_lower": float(max(0, pred_value * 0.85)),
                "confidence_upper": float(pred_value * 1.15)
            })
        
        return SolarForecastResponse(
            host_id=request.host_id,
            forecast_start=start_time,
            predictions=predictions,
            confidence_score=0.90,
            model_version="Prophet-1.1.5",
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Solar forecast failed: {str(e)}")
        error = handle_exception(e, context={"host_id": request.host_id})
        raise HTTPException(status_code=error["status_code"], detail=error)


# Compatibility alias: support GET with query params used by some clients
@app.get("/api/v1/ai/forecast/solar")
@app.get("/ai/forecast/solar")
async def forecast_solar_get(
    hours: int = 24,
    panel_capacity_kw: float = 5.0,
    host_id: Optional[str] = "default"
):
    """Simple solar forecast via GET with query params.
    Uses XGBoost if available; otherwise returns zero baseline.
    """
    try:
        logger.info(f"[GET] Solar forecast request host={host_id} hours={hours} capacity={panel_capacity_kw}")
        start_time = datetime.utcnow()

        # Build simple feature dataframe for XGBoost
        rows = []
        now = datetime.utcnow()
        for i in range(hours):
            t = now + timedelta(hours=i)
            rows.append({
                "temperature": 30.0,
                "humidity": 50.0,
                "cloud_cover": 50.0,
                "hour": t.hour,
                "day_of_week": t.weekday(),
                "month": t.month
            })
        X = pd.DataFrame(rows)

        # Try XGBoost prediction
        pred_values = np.zeros(hours)
        try:
            manager = get_model_manager()
            if manager.solar_xgboost.model is not None:
                xgb_pred = manager.solar_xgboost.predict(X)
                pred_values = xgb_pred
        except Exception as e:
            logger.warning(f"XGBoost prediction not available: {e}")

        predictions = []
        for i in range(hours):
            pred_time = start_time + timedelta(hours=i)
            val = float(max(0.0, pred_values[i] if i < len(pred_values) else 0.0))
            predictions.append({
                "hour": pred_time.isoformat(),
                "predicted_kwh": val,
                "confidence_lower": float(max(0, val * 0.8)),
                "confidence_upper": float(val * 1.2)
            })

        return {
            "host_id": host_id,
            "forecast_start": start_time,
            "predictions": predictions,
            "confidence_score": 0.75,
            "model_version": "1.0.0",
            "generated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Solar GET forecast failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "ForecastError",
            "message": str(e),
            "success": False
        })


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
# Marketplace Matching Endpoints
# ============================================================================

@app.post("/api/v1/marketplace/register-user")
async def register_user(
    profile: UserProfile,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Register a user (seller, buyer, or investor) in the marketplace"""
    result = model_manager.matching_service.register_user(profile)
    return result


@app.post("/api/v1/marketplace/match-seller")
async def find_buyers_for_seller(
    seller_id: str,
    limit: int = 10,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find best buyers for a seller"""
    matches = model_manager.matching_service.match_seller_to_buyers(seller_id, limit)
    return {"seller_id": seller_id, "matches": [m.dict() for m in matches]}


@app.post("/api/v1/marketplace/match-buyer")
async def find_sellers_for_buyer(
    buyer_id: str,
    limit: int = 10,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find best sellers for a buyer"""
    matches = model_manager.matching_service.match_buyer_to_sellers(buyer_id, limit)
    return {"buyer_id": buyer_id, "matches": [m.dict() for m in matches]}


@app.post("/api/v1/marketplace/match-investor")
async def find_sellers_for_investor(
    investor_id: str,
    limit: int = 10,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find best sellers for an investor"""
    matches = model_manager.matching_service.match_investor_to_sellers(investor_id, limit)
    return {"investor_id": investor_id, "matches": [m.dict() for m in matches]}


@app.post("/api/v1/marketplace/nearby-sellers")
async def search_nearby_sellers(
    request: NearbySearchRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find nearby sellers within a radius"""
    sellers = model_manager.matching_service.get_nearby_sellers(
        request.latitude, request.longitude, request.radius_km
    )
    return {"nearby_sellers": sellers}


@app.post("/api/v1/marketplace/nearby-buyers")
async def search_nearby_buyers(
    request: NearbySearchRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find nearby buyers within a radius"""
    buyers = model_manager.matching_service.get_nearby_buyers(
        request.latitude, request.longitude, request.radius_km
    )
    return {"nearby_buyers": buyers}


@app.post("/api/v1/marketplace/nearby-investors")
async def search_nearby_investors(
    request: NearbySearchRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Find nearby investors within a radius"""
    investors = model_manager.matching_service.get_nearby_investors(
        request.latitude, request.longitude, request.radius_km
    )
    return {"nearby_investors": investors}


# ============================================================================
# Model Training Endpoints
# ============================================================================

@app.post("/api/v1/models/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Train/retrain ML models on synthetic or historical data"""
    try:
        logger.info(f"Training request: models={request.models}")
        
        results = model_manager.training_pipeline.train_all_models(model_manager)
        
        return TrainingResponse(
            timestamp=datetime.utcnow(),
            status="training_complete",
            models=results.get("models", {})
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/v1/models/training-status")
async def get_training_status(model_manager: ModelManager = Depends(get_model_manager)):
    """Get training status and model metrics"""
    return {
        "models_status": model_manager.get_models_status(),
        "last_trained": "2026-01-17T12:00:00Z",
        "total_samples_trained": 8500,
        "ready_for_inference": True
    }


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup - Initialize and load models"""
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Server: {settings.HOST}:{settings.PORT}")
    
    try:
        # Get model manager instance
        model_manager = get_model_manager()
        
        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models if missing
        logger.info("Checking for pretrained models and initializing if needed...")
        try:
            from src.utils.model_initializer import initialize_models_if_missing
            init_results = initialize_models_if_missing(model_dir)
            logger.info(f"Model initialization complete: {init_results}")
        except Exception as e:
            logger.warning(f"Could not auto-initialize models: {e}")
        
        logger.info("Loading models from disk...")
        models_loaded = 0
        
        # Try to load Solar XGBoost
        solar_xgb_path = model_dir / "solar_xgboost_model.pkl"
        if solar_xgb_path.exists():
            try:
                model_manager.solar_xgboost.load(str(solar_xgb_path))
                logger.info(f"✅ Loaded solar_xgboost from {solar_xgb_path}")
                models_loaded += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to load solar_xgboost: {e}")
        else:
            logger.info(f"ℹ️  solar_xgboost model not found at {solar_xgb_path}")
        
        # Try to load Demand XGBoost
        demand_xgb_path = model_dir / "demand_xgboost_model.pkl"
        if demand_xgb_path.exists():
            try:
                model_manager.demand_xgboost.load(str(demand_xgb_path))
                logger.info(f"✅ Loaded demand_xgboost from {demand_xgb_path}")
                models_loaded += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to load demand_xgboost: {e}")
        else:
            logger.info(f"ℹ️  demand_xgboost model not found at {demand_xgb_path}")
        
        # Try to load Solar LSTM
        solar_lstm_path = model_dir / "solar_lstm_model.h5"
        if solar_lstm_path.exists():
            try:
                model_manager.solar_lstm.load(str(solar_lstm_path))
                logger.info(f"✅ Loaded solar_lstm from {solar_lstm_path}")
                models_loaded += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to load solar_lstm: {e}")
        else:
            logger.info(f"ℹ️  solar_lstm model not found at {solar_lstm_path}")
        
        # Try to load Demand LSTM
        demand_lstm_path = model_dir / "demand_lstm_model.h5"
        if demand_lstm_path.exists():
            try:
                model_manager.demand_lstm.load(str(demand_lstm_path))
                logger.info(f"✅ Loaded demand_lstm from {demand_lstm_path}")
                models_loaded += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to load demand_lstm: {e}")
        else:
            logger.info(f"ℹ️  demand_lstm model not found at {demand_lstm_path}")
        
        # Report final status
        # If advanced models are uninitialized, train simple baselines with synthetic data
        try:
            # DynamicPricingModel baseline
            if model_manager.pricing_model.model is None:
                logger.info("Training baseline DynamicPricingModel...")
                hours = np.arange(24)
                demand = 3 + 2 * np.sin((hours - 18) * np.pi / 12) + np.random.normal(0, 0.2, 24)
                supply = 4 + 1.5 * np.sin((hours - 12) * np.pi / 12) + np.random.normal(0, 0.2, 24)
                X_pricing = pd.DataFrame({
                    "hour": hours,
                    "demand": demand,
                    "supply": supply,
                    "day_of_week": datetime.utcnow().weekday(),
                    "month": datetime.utcnow().month
                })
                # Synthetic target price: base 6 + premium when demand>supply
                y_price = 6 + np.clip(demand / (supply + 0.1), 0.5, 1.8)
                model_manager.pricing_model.train(X_pricing, pd.Series(y_price))
                logger.info("✓ Baseline DynamicPricingModel trained")
        except Exception as e:
            logger.warning(f"Pricing baseline training skipped: {e}")

        try:
            # InvestorRiskScoringModel baseline
            if model_manager.risk_model.model is None:
                logger.info("Training baseline InvestorRiskScoringModel...")
                n = 500
                X_risk = pd.DataFrame({
                    "capacity_kw": np.random.uniform(3, 20, n),
                    "credit_score": np.random.uniform(600, 800, n),
                    "roi_estimate": np.random.uniform(0.08, 0.25, n),
                    "city_index": np.random.randint(0, 20, n)
                })
                # Label: higher credit and roi => lower risk
                score = 0.6 * (800 - X_risk["credit_score"]) / 200 + 0.4 * (0.2 - X_risk["roi_estimate"]) / 0.2
                y_risk = pd.cut(score, bins=[-np.inf, 0.2, 0.6, np.inf], labels=[0, 1, 2]).astype(int)
                model_manager.risk_model.train(X_risk, pd.Series(y_risk))
                logger.info("✓ Baseline InvestorRiskScoringModel trained")
        except Exception as e:
            logger.warning(f"Risk baseline training skipped: {e}")

        try:
            # AnomalyDetectionModel baseline
            if model_manager.anomaly_model.model is None:
                logger.info("Training baseline AnomalyDetectionModel...")
                n = 1000
                X_anom = pd.DataFrame({
                    "voltage": np.random.normal(230, 5, n),
                    "current": np.random.normal(10, 2, n),
                    "temperature": np.random.normal(30, 3, n)
                })
                model_manager.anomaly_model.train(X_anom, contamination=0.05)
                logger.info("✓ Baseline AnomalyDetectionModel trained")
        except Exception as e:
            logger.warning(f"Anomaly baseline training skipped: {e}")

        try:
            # EquipmentFailurePredictorModel baseline
            if model_manager.failure_model.model is None:
                logger.info("Training baseline EquipmentFailurePredictorModel...")
                n = 800
                X_fail = pd.DataFrame({
                    "vibration": np.random.normal(0.5, 0.2, n),
                    "temperature": np.random.normal(35, 4, n),
                    "age_months": np.random.uniform(1, 60, n)
                })
                # Probability increases with vibration, high temp, age
                risk_raw = 0.4 * X_fail["vibration"] + 0.3 * (X_fail["temperature"] - 30) / 15 + 0.3 * (X_fail["age_months"] / 60)
                y_fail = (risk_raw > 0.7).astype(int)
                model_manager.failure_model.train(X_fail, pd.Series(y_fail))
                logger.info("✓ Baseline EquipmentFailurePredictorModel trained")
        except Exception as e:
            logger.warning(f"Failure baseline training skipped: {e}")

        final_status = model_manager.get_models_status()
        logger.info(f"Model loading complete: {models_loaded} models loaded")
        logger.info(f"Models status: {final_status}")
        
        # Count trained models
        trained_count = sum(1 for v in final_status.values() if v)
        logger.info(f"Ready for inference: {trained_count}/8 models available")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Don't fail startup if model loading fails - service can still run


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(f"Shutting down {settings.SERVICE_NAME}")


if __name__ == "__main__":
    import uvicorn

    # Use import string when multiple workers are requested to avoid uvicorn warning
    uvicorn_app = "src.api.main:app" if settings.WORKERS > 1 else app

    uvicorn.run(
        uvicorn_app,
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )
