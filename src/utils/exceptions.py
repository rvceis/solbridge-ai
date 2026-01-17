"""Custom exceptions and error handling"""
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLServiceException(Exception):
    """Base exception for ML Service"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "ML_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        
        logger.error(
            f"{error_code}: {message}",
            extra={
                "error_code": error_code,
                "status_code": status_code,
                "details": self.details
            }
        )
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }


class DataValidationError(MLServiceException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        error_code = f"VALIDATION_ERROR_{field}" if field else "VALIDATION_ERROR"
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=400,
            details={"field": field, **kwargs}
        )


class DataProcessingError(MLServiceException):
    """Raised when data preprocessing fails"""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            status_code=500,
            details={"stage": stage, **kwargs}
        )


class ModelNotFoundError(MLServiceException):
    """Raised when model not found"""
    
    def __init__(self, model_name: str, version: Optional[str] = None):
        super().__init__(
            message=f"Model '{model_name}' version '{version}' not found",
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            details={"model": model_name, "version": version}
        )


class ModelLoadError(MLServiceException):
    """Raised when model fails to load"""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            status_code=500,
            details={"model": model_name, "reason": reason}
        )


class PredictionError(MLServiceException):
    """Raised when prediction fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            status_code=500,
            details={"model": model_name, **kwargs}
        )


class FeatureEngineeringError(MLServiceException):
    """Raised when feature engineering fails"""
    
    def __init__(self, message: str, feature_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="FEATURE_ENGINEERING_ERROR",
            status_code=500,
            details={"feature": feature_name, **kwargs}
        )


class DatabaseError(MLServiceException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={"operation": operation, **kwargs}
        )


class CacheError(MLServiceException):
    """Raised when cache operations fail"""
    
    def __init__(self, message: str, key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details={"cache_key": key, **kwargs}
        )


class ConfigError(MLServiceException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            status_code=500,
            details={"config_key": config_key}
        )


class TimeoutError(MLServiceException):
    """Raised when operation times out"""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout}s",
            error_code="TIMEOUT_ERROR",
            status_code=504,
            details={"operation": operation, "timeout": timeout}
        )


class RateLimitError(MLServiceException):
    """Raised when rate limit exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details={}
        )


class InsufficientDataError(MLServiceException):
    """Raised when insufficient data for model"""
    
    def __init__(self, required: int, available: int, **kwargs):
        super().__init__(
            message=f"Insufficient data: required {required}, available {available}",
            error_code="INSUFFICIENT_DATA_ERROR",
            status_code=400,
            details={"required": required, "available": available, **kwargs}
        )


def handle_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generic exception handler
    
    Args:
        exc: Exception to handle
        context: Additional context information
    
    Returns:
        Error response dictionary
    """
    if isinstance(exc, MLServiceException):
        response = exc.to_dict()
    else:
        logger.exception(f"Unhandled exception: {str(exc)}", extra={"context": context})
        response = {
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "details": {"error_type": type(exc).__name__}
        }
    
    return response
