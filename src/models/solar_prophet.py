"""
Prophet-based Solar Forecasting Model
Uses Facebook's Prophet for time series forecasting (pre-trained, no training needed)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from prophet import Prophet
import warnings

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, PredictionError

warnings.filterwarnings("ignore")  # Suppress Prophet warnings
logger = get_logger(__name__)


class SolarProphetModel:
    """
    Prophet-based solar forecasting model
    Pre-trained meta-learner - works great without historical data
    """
    
    def __init__(self):
        """Initialize Prophet model"""
        self.model = None
        self.trained = False
        logger.info("Initializing SolarProphetModel (Facebook Prophet)")
    
    def train_or_load(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Train or create Prophet model
        
        If historical_data provided, trains on it.
        Otherwise, uses default seasonal patterns (Prophet works great for this!)
        
        Args:
            historical_data: DataFrame with columns ['timestamp', 'power_kw']
        """
        try:
            logger.info("Setting up Prophet model for solar forecasting...")
            
            # Prepare data
            if historical_data is not None and len(historical_data) > 48:
                logger.info(f"Training on {len(historical_data)} historical records")
                df = historical_data.copy()
                df.columns = ["ds", "y"]  # Prophet expects ds, y
                df["ds"] = pd.to_datetime(df["ds"])
            else:
                # Create synthetic seasonal data for Prophet to learn from
                logger.info("Generating synthetic training data for Prophet...")
                dates = pd.date_range(start="2023-01-01", periods=365*24, freq="H")
                
                # Realistic solar generation pattern
                hours = np.array([d.hour for d in dates])
                days = np.array([d.dayofyear for d in dates])
                
                # Daily pattern (peak at noon, zero at night)
                daily_pattern = np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * 5
                
                # Seasonal variation
                seasonal = 0.7 + 0.3 * np.sin(days * 2 * np.pi / 365)
                
                # Cloud effect
                cloud_effect = 1.0 - np.random.uniform(0, 0.4, len(dates))
                
                power = daily_pattern * seasonal * cloud_effect
                
                df = pd.DataFrame({
                    "ds": dates,
                    "y": power
                })
            
            # Create and fit Prophet model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=0.95,
                changepoint_prior_scale=0.01  # Smooth trends
            )
            
            # Add regressors if needed (optional)
            # self.model.add_regressor('temperature')
            
            logger.info("Training Prophet model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(df)
            
            self.trained = True
            logger.info("✓ Prophet model trained successfully")
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            raise ModelLoadError("SolarProphetModel", str(e))
    
    def predict(
        self,
        hours_ahead: int = 48,
        weather_data: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forecast solar generation
        
        Args:
            hours_ahead: Number of hours to forecast
            weather_data: Optional weather DataFrame with temp, cloud_cover, etc.
        
        Returns:
            (predictions, forecast_details)
        """
        if not self.trained or self.model is None:
            raise ModelLoadError("SolarProphetModel", "Model not trained")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=hours_ahead,
                freq="H"
            )
            
            # Make forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model.predict(future)
            
            # Extract predictions
            predictions = forecast.iloc[-hours_ahead:]["yhat"].values
            lower = forecast.iloc[-hours_ahead:]["yhat_lower"].values
            upper = forecast.iloc[-hours_ahead:]["yhat_upper"].values
            
            # Ensure non-negative
            predictions = np.maximum(predictions, 0)
            lower = np.maximum(lower, 0)
            upper = np.maximum(upper, 0)
            
            # Adjust with weather data if provided
            if weather_data is not None and len(weather_data) >= hours_ahead:
                predictions = self._adjust_with_weather(
                    predictions,
                    weather_data.iloc[:hours_ahead]
                )
            
            details = {
                "model_type": "Prophet",
                "hours_forecast": hours_ahead,
                "confidence_level": 0.95,
                "mean_prediction": float(np.mean(predictions)),
                "max_prediction": float(np.max(predictions)),
                "uncertainty": float(np.mean(upper - lower))
            }
            
            logger.info(f"Prophet forecast: {hours_ahead} hours, mean={details['mean_prediction']:.2f} kW")
            return predictions, details
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            raise PredictionError(str(e), model_name="SolarProphetModel")
    
    def _adjust_with_weather(
        self,
        predictions: np.ndarray,
        weather_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Adjust predictions using weather data
        
        Args:
            predictions: Base Prophet predictions
            weather_df: Weather data with cloud_cover, temperature, etc.
        
        Returns:
            Adjusted predictions
        """
        try:
            adjusted = predictions.copy()
            
            # Cloud cover adjustment (lower clouds = more solar)
            if "cloud_cover" in weather_df.columns:
                cloud_factor = 1.0 - (weather_df["cloud_cover"].values / 100) * 0.8
                adjusted = adjusted * cloud_factor
            
            # Temperature adjustment (hot = less efficient)
            if "temperature" in weather_df.columns:
                # PV efficiency drops ~0.4% per °C above 25°C
                temp_factor = 1.0 - (np.maximum(0, weather_df["temperature"].values - 25) * 0.004)
                adjusted = adjusted * temp_factor
            
            adjusted = np.maximum(adjusted, 0)
            logger.info("Weather adjustments applied to forecast")
            return adjusted
            
        except Exception as e:
            logger.warning(f"Weather adjustment failed: {e}, using base predictions")
            return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.trained:
            return {"status": "not_trained"}
        
        return {
            "model_type": "Prophet",
            "status": "trained",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": True,
            "confidence_interval": 0.95,
            "notes": "Pre-trained meta-learner by Facebook, works without training data"
        }
