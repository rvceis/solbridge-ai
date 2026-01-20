"""
Weather Data Service - Fetch real weather data from free APIs
Uses Open-Meteo (free, no API key needed) and NASA POWER as fallback
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherService:
    """Fetch real weather data from free APIs"""
    
    # Open-Meteo API (free, no key needed)
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    
    # NASA POWER API (free, no key needed)
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/v1/solar/hourly/point"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        logger.info("WeatherService initialized")
    
    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 48
    ) -> Dict[str, any]:
        """
        Get weather forecast from Open-Meteo (free API)
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            hours: Number of hours to forecast (max 240)
        
        Returns:
            Dict with temperature, humidity, cloud_cover, wind_speed, etc.
        """
        try:
            logger.info(f"Fetching weather forecast for ({latitude}, {longitude})")
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,direct_radiation",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "forecast_hours": min(hours, 240),  # Max 240 hours
                "timezone": "auto"
            }
            
            response = self.session.get(self.OPEN_METEO_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                raise ValueError("Invalid API response")
            
            logger.info(f"✓ Weather forecast fetched successfully")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to fetch weather forecast: {e}")
            return self._get_default_weather(hours)
    
    def get_solar_irradiance(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get solar irradiance data from NASA POWER API (free)
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYYMMDD format)
            end_date: End date (YYYYMMDD format)
        
        Returns:
            DataFrame with solar irradiance data or None
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            
            logger.info(f"Fetching solar irradiance from NASA POWER ({start_date} to {end_date})")
            
            params = {
                "start": start_date,
                "end": end_date,
                "latitude": latitude,
                "longitude": longitude,
                "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN",
                "community": "sb",
                "format": "json",
                "user": "solbridge-ai"
            }
            
            response = self.session.get(self.NASA_POWER_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "properties" not in data or "parameter" not in data["properties"]:
                logger.warning("Invalid NASA POWER response")
                return None
            
            # Parse response
            irradiance = data["properties"]["parameter"].get("ALLSKY_SFC_SW_DWN", {})
            
            records = []
            for date_str, value in irradiance.items():
                records.append({
                    "date": date_str,
                    "irradiance_w_m2": value
                })
            
            df = pd.DataFrame(records)
            logger.info(f"✓ Solar irradiance fetched: {len(df)} records")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch solar irradiance: {e}")
            return None
    
    def parse_forecast_to_dataframe(self, weather_data: Dict) -> pd.DataFrame:
        """
        Convert Open-Meteo forecast to usable DataFrame
        
        Args:
            weather_data: Response from get_weather_forecast()
        
        Returns:
            DataFrame with hourly weather data
        """
        try:
            hourly = weather_data.get("hourly", {})
            
            df = pd.DataFrame({
                "timestamp": hourly.get("time", []),
                "temperature": hourly.get("temperature_2m", []),
                "humidity": hourly.get("relative_humidity_2m", []),
                "cloud_cover": hourly.get("cloud_cover", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "irradiance": hourly.get("direct_radiation", [0] * len(hourly.get("time", [])))
            })
            
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Parsed {len(df)} hours of weather data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse forecast: {e}")
            return pd.DataFrame()
    
    def _get_default_weather(self, hours: int) -> Dict:
        """Return default/fallback weather data"""
        logger.info("Using default weather data (API fallback)")
        
        now = datetime.utcnow()
        times = [(now + timedelta(hours=i)).isoformat() for i in range(hours)]
        
        # Realistic defaults: 25°C, 60% humidity, 30% cloud
        return {
            "hourly": {
                "time": times,
                "temperature_2m": [25.0] * hours,
                "relative_humidity_2m": [60.0] * hours,
                "cloud_cover": [30.0] * hours,
                "wind_speed_10m": [5.0] * hours,
                "direct_radiation": [100.0] * hours  # Low baseline
            }
        }
    
    def enrich_with_solar_potential(
        self,
        weather_df: pd.DataFrame,
        latitude: float,
        longitude: float
    ) -> pd.DataFrame:
        """
        Add solar potential calculations to weather data
        
        Args:
            weather_df: Weather DataFrame
            latitude: Location latitude
            longitude: Location longitude
        
        Returns:
            Enhanced DataFrame with solar potential
        """
        try:
            # Simple solar potential model based on:
            # - Cloud cover (lower = more solar)
            # - Hour of day (peak at noon)
            # - Latitude-based seasonal variation
            
            weather_df = weather_df.copy()
            
            # Extract hour
            weather_df["hour"] = weather_df["timestamp"].dt.hour
            weather_df["day_of_year"] = weather_df["timestamp"].dt.dayofyear
            
            # Solar intensity model
            # Peak at noon (hour 12), zero at night
            solar_factor = np.maximum(0, np.sin((weather_df["hour"] - 6) * np.pi / 12))
            
            # Cloud cover reduces solar (0% clouds = 1.0, 100% clouds = 0.1)
            cloud_factor = 1.0 - (weather_df["cloud_cover"] / 100) * 0.9
            
            # Seasonal variation (simplified)
            seasonal_factor = 0.7 + 0.3 * np.abs(np.sin(weather_df["day_of_year"] * 2 * np.pi / 365))
            
            # Combine factors
            weather_df["solar_potential"] = solar_factor * cloud_factor * seasonal_factor * 1000  # W/m²
            weather_df["solar_potential"] = weather_df["solar_potential"].clip(lower=0)
            
            logger.info("Added solar potential calculations")
            return weather_df
            
        except Exception as e:
            logger.error(f"Failed to enrich weather data: {e}")
            return weather_df


def get_weather_service() -> WeatherService:
    """Get or create WeatherService singleton"""
    if not hasattr(get_weather_service, "_instance"):
        get_weather_service._instance = WeatherService()
    return get_weather_service._instance
