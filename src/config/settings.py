import os
from functools import lru_cache
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="allow")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Service
    SERVICE_NAME: str = "solar-ml-service"
    SERVICE_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = os.getenv("ML_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("ML_PORT", 8001))
    WORKERS: int = int(os.getenv("ML_WORKERS", 4))
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/solar_sharing"
    )
    TIMESCALE_URL: str = os.getenv(
        "TIMESCALE_URL",
        "postgresql://user:password@localhost:5432/timescale"
    )
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Model Configuration
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
    
    # Feature Store
    FEATURE_STORE_TYPE: str = os.getenv("FEATURE_STORE_TYPE", "redis")  # redis, s3, local
    S3_BUCKET: str = os.getenv("S3_BUCKET", "solar-features")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "solar-forecasting")
    
    # Model Performance Targets
    SOLAR_FORECAST_MAPE_TARGET: float = 0.12  # 12%
    DEMAND_FORECAST_MAPE_TARGET: float = 0.15  # 15%
    
    # Batch Processing
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    HISTORICAL_HOURS: int = int(os.getenv("HISTORICAL_HOURS", 168))  # 7 days
    FORECAST_HOURS: int = int(os.getenv("FORECAST_HOURS", 48))  # 2 days
    
    # API Configuration
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 30))
    MAX_CONCURRENT_PREDICTIONS: int = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", 100))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    
    # Weather API
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    WEATHER_UPDATE_INTERVAL: int = int(os.getenv("WEATHER_UPDATE_INTERVAL", 3600))  # 1 hour
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json, text
    LOG_FILE: str = os.path.join(LOG_DIR, "ml-service.log")
    
    # Feature Engineering
    USE_LAG_FEATURES: bool = True
    USE_ROLLING_FEATURES: bool = True
    USE_WEATHER_FEATURES: bool = True
    USE_TIME_FEATURES: bool = True
    
    # Model Ensemble
    ENABLE_ENSEMBLE: bool = True
    ENSEMBLE_WEIGHTS: dict = {
        "lstm": 0.6,
        "xgboost": 0.4
    }
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", 9090))
    ENABLE_DRIFT_DETECTION: bool = True
    DRIFT_THRESHOLD: float = 0.15  # 15%
    
    # Training
    EPOCHS: int = int(os.getenv("EPOCHS", 100))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 0.001))
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
