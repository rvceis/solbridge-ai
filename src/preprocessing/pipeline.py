"""Data preprocessing pipeline with validation, cleaning, and feature engineering"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import math

from src.utils.logger import get_logger
from src.utils.exceptions import (
    DataValidationError,
    DataProcessingError,
    FeatureEngineeringError,
    InsufficientDataError
)

logger = get_logger(__name__)


class DataValidator:
    """Validates incoming data"""
    
    # Validation rules by data type
    VALIDATION_RULES = {
        "solar_generation": {
            "power_kw": {"min": 0, "max": None},  # Max calculated from capacity
            "energy_kwh": {"min": 0},  # Monotonically increasing
            "voltage": {"min": 200, "max": 260},
            "current": {"min": 0, "max": 100},
            "frequency": {"min": 49.5, "max": 50.5},
            "power_factor": {"min": 0.8, "max": 1.0},
            "temperature": {"min": -20, "max": 60},
        },
        "consumption": {
            "power_kw": {"min": 0, "max": 50},  # Residential max
            "energy_kwh": {"min": 0},
            "voltage": {"min": 200, "max": 260},
            "current": {"min": 0, "max": 100},
            "frequency": {"min": 49.5, "max": 50.5},
            "power_factor": {"min": 0.8, "max": 1.0},
        },
        "weather": {
            "temperature": {"min": -20, "max": 60},
            "humidity": {"min": 0, "max": 100},
            "wind_speed": {"min": 0, "max": 50},
            "irradiance": {"min": 0, "max": 1400},
            "cloud_cover": {"min": 0, "max": 100},
        }
    }
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against schema
        
        Args:
            data: Data to validate
            schema: Expected schema
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        for field, field_type in schema.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], field_type):
                errors.append(f"Field '{field}' expected {field_type}, got {type(data[field])}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_ranges(
        data: pd.DataFrame,
        data_type: str,
        system_capacity_kw: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate value ranges
        
        Args:
            data: DataFrame to validate
            data_type: Type of data (solar_generation, consumption, weather)
            system_capacity_kw: System capacity for solar generation validation
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        if data_type not in DataValidator.VALIDATION_RULES:
            return True, []
        
        rules = DataValidator.VALIDATION_RULES[data_type]
        
        for field, limits in rules.items():
            if field not in data.columns:
                continue
            
            # Min check
            if "min" in limits and limits["min"] is not None:
                mask = data[field] < limits["min"]
                if mask.any():
                    errors.append(
                        f"Field '{field}': {mask.sum()} values below minimum {limits['min']}"
                    )
            
            # Max check
            if "max" in limits and limits["max"] is not None:
                max_limit = limits["max"]
                # Special case: solar generation max based on capacity
                if field == "power_kw" and data_type == "solar_generation" and system_capacity_kw:
                    max_limit = system_capacity_kw * 1.2  # Allow 20% overage
                
                mask = data[field] > max_limit
                if mask.any():
                    errors.append(
                        f"Field '{field}': {mask.sum()} values above maximum {max_limit}"
                    )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_temporal(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate temporal consistency"""
        errors = []
        
        if "timestamp" not in data.columns:
            return True, []
        
        # Check chronological order
        if not data["timestamp"].is_monotonic_increasing:
            errors.append("Data not in chronological order")
        
        # Check for duplicates
        dup_count = data["timestamp"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate timestamps")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_solar_generation(
        data: pd.DataFrame,
        system_capacity_kw: float,
        sunrise_hour: float = 6,
        sunset_hour: float = 18
    ) -> Tuple[bool, List[str]]:
        """
        Solar-specific validation
        
        Args:
            data: DataFrame with solar generation
            system_capacity_kw: Panel capacity
            sunrise_hour: Typical sunrise hour
            sunset_hour: Typical sunset hour
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        if "power_kw" not in data.columns:
            return True, []
        
        # Generation should be near zero at night
        data_copy = data.copy()
        if "timestamp" in data_copy.columns:
            data_copy["hour"] = pd.to_datetime(data_copy["timestamp"]).dt.hour
            night_mask = (data_copy["hour"] < sunrise_hour - 1) | (data_copy["hour"] > sunset_hour + 1)
            night_data = data_copy[night_mask]
            
            if (night_data["power_kw"] > 0.1).any():
                suspicious = (night_data["power_kw"] > 0.1).sum()
                errors.append(f"Night generation detected: {suspicious} suspicious readings")
        
        return len(errors) == 0, errors


class DataCleaner:
    """Cleans and preprocesses data"""
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        method: str = "forward_fill",
        max_gap_hours: int = 2
    ) -> pd.DataFrame:
        """
        Handle missing values using appropriate method
        
        Args:
            data: DataFrame with possible missing values
            method: Method to use (forward_fill, linear, seasonal, model_based)
            max_gap_hours: Maximum gap to fill
        
        Returns:
            DataFrame with filled values
        """
        original_missing = data.isnull().sum().sum()
        logger.info(f"Missing values before cleaning: {original_missing}")
        
        if original_missing == 0:
            return data
        
        data = data.copy()
        
        try:
            if method == "forward_fill":
                # For gaps < 1 hour, use forward fill
                data = data.fillna(method="ffill", limit=max_gap_hours)
                
            elif method == "linear":
                # Linear interpolation for short gaps
                data = data.interpolate(method="linear", limit_direction="both")
                
            elif method == "seasonal":
                # Seasonal average for longer gaps
                if "timestamp" in data.columns:
                    data["timestamp"] = pd.to_datetime(data["timestamp"])
                    data["hour"] = data["timestamp"].dt.hour
                    data["day_of_week"] = data["timestamp"].dt.dayofweek
                    
                    # Fill with hour + day_of_week average
                    for col in data.columns:
                        if col not in ["timestamp", "hour", "day_of_week"]:
                            mask = data[col].isnull()
                            if mask.any():
                                seasonal_mean = data.groupby(["hour", "day_of_week"])[col].transform("mean")
                                data.loc[mask, col] = seasonal_mean[mask]
                    
                    data = data.drop(columns=["hour", "day_of_week"])
            
            elif method == "knn":
                # KNN imputation
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            
            # Fill remaining with mean
            data = data.fillna(data.mean())
            
            filled = original_missing - data.isnull().sum().sum()
            logger.info(f"Missing values filled: {filled}/{original_missing}")
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to handle missing values: {str(e)}",
                stage="missing_value_imputation"
            )
        
        return data
    
    @staticmethod
    def detect_and_remove_outliers(
        data: pd.DataFrame,
        method: str = "iqr",
        contamination: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detect and remove outliers
        
        Args:
            data: DataFrame
            method: Method (iqr, zscore, isolation_forest)
            contamination: Expected outlier fraction
        
        Returns:
            (cleaned_df, outlier_info)
        """
        data = data.copy()
        outlier_info = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        try:
            if method == "iqr":
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_info[col] = mask.sum()
                    
                    data.loc[mask, col] = data.loc[~mask, col].mean()
            
            elif method == "zscore":
                from scipy import stats
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(data[col]))
                    mask = z_scores > 3
                    outlier_info[col] = mask.sum()
                    data.loc[mask, col] = data.loc[~mask, col].mean()
            
            elif method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_mask = iso_forest.fit_predict(data[numeric_cols]) == -1
                outlier_info["total_outliers"] = outlier_mask.sum()
                data = data[~outlier_mask]
            
            logger.info(f"Outliers removed: {outlier_info}")
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to remove outliers: {str(e)}",
                stage="outlier_detection"
            )
        
        return data, outlier_info
    
    @staticmethod
    def smooth_noisy_data(
        data: pd.DataFrame,
        window_size: int = 3,
        method: str = "rolling_mean"
    ) -> pd.DataFrame:
        """
        Smooth noisy data
        
        Args:
            data: DataFrame
            window_size: Window size for smoothing
            method: Method (rolling_mean, ewma, savgol)
        
        Returns:
            Smoothed DataFrame
        """
        data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        try:
            if method == "rolling_mean":
                for col in numeric_cols:
                    data[col] = data[col].rolling(
                        window=window_size,
                        center=True,
                        min_periods=1
                    ).mean()
            
            elif method == "ewma":
                for col in numeric_cols:
                    data[col] = data[col].ewm(span=window_size, adjust=False).mean()
            
            elif method == "savgol":
                from scipy.signal import savgol_filter
                for col in numeric_cols:
                    # Only apply if enough points
                    if len(data) > window_size:
                        data[col] = savgol_filter(data[col], window_length=window_size, polyorder=2)
            
            logger.info(f"Data smoothed using {method}")
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to smooth data: {str(e)}",
                stage="data_smoothing"
            )
        
        return data


class FeatureEngineer:
    """Creates features for ML models"""
    
    @staticmethod
    def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time features"""
        if "timestamp" not in data.columns:
            logger.warning("No timestamp column found, skipping time features")
            return data
        
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        try:
            # Hour of day (cyclical)
            hour = data["timestamp"].dt.hour
            data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (cyclical)
            day_of_week = data["timestamp"].dt.dayofweek
            data["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
            data["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Month (cyclical)
            month = data["timestamp"].dt.month
            data["month_sin"] = np.sin(2 * np.pi * month / 12)
            data["month_cos"] = np.cos(2 * np.pi * month / 12)
            
            # Day of year
            day_of_year = data["timestamp"].dt.dayofyear
            data["day_of_year"] = (day_of_year - 1) / 365
            
            # Boolean features
            data["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)
            data["is_daytime"] = ((hour >= 6) & (hour <= 18)).astype(int)
            data["is_peak_solar"] = ((hour >= 10) & (hour <= 15)).astype(int)
            data["is_peak_consumption"] = ((hour >= 18) & (hour <= 22)).astype(int)
            
            logger.info("Time features created successfully")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create time features: {str(e)}")
        
        return data
    
    @staticmethod
    def create_lag_features(
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 24, 168]
    ) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            data: DataFrame
            columns: Columns to create lags for
            lags: Lag hours
        
        Returns:
            DataFrame with lag features
        """
        data = data.copy()
        
        try:
            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} not found, skipping lag features")
                    continue
                
                for lag in lags:
                    data[f"{col}_lag_{lag}h"] = data[col].shift(lag)
            
            logger.info(f"Lag features created for {len(columns)} columns")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create lag features: {str(e)}")
        
        return data
    
    @staticmethod
    def create_rolling_features(
        data: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [6, 24, 168]
    ) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            data: DataFrame
            columns: Columns to create rolling features for
            windows: Window sizes (hours)
        
        Returns:
            DataFrame with rolling features
        """
        data = data.copy()
        
        try:
            for col in columns:
                if col not in data.columns:
                    continue
                
                for window in windows:
                    data[f"{col}_rolling_mean_{window}h"] = data[col].rolling(window, min_periods=1).mean()
                    data[f"{col}_rolling_std_{window}h"] = data[col].rolling(window, min_periods=1).std()
                    data[f"{col}_rolling_max_{window}h"] = data[col].rolling(window, min_periods=1).max()
                    data[f"{col}_rolling_min_{window}h"] = data[col].rolling(window, min_periods=1).min()
            
            logger.info(f"Rolling features created for {len(columns)} columns")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create rolling features: {str(e)}")
        
        return data
    
    @staticmethod
    def create_weather_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create derived weather features"""
        data = data.copy()
        
        try:
            # Clear sky index
            if "cloud_cover" in data.columns:
                data["clear_sky_index"] = 1.0 - (data["cloud_cover"] / 100) * 0.75
            
            # Estimated irradiance (simple proxy)
            if "cloud_cover" in data.columns:
                # Assume maximum irradiance of 1000 W/m² on clear day
                data["estimated_irradiance"] = 1000 * data["clear_sky_index"]
            
            # Heat index (simplified)
            if "temperature" in data.columns and "humidity" in data.columns:
                T = data["temperature"]
                RH = data["humidity"]
                # Simplified heat index formula
                data["heat_index"] = T + (5/9) * (RH/100 * (6.112 * np.exp((17.67*T)/(T+243.5)))) - 10
            
            # Wind chill (simplified, only valid for T < 10°C)
            if "temperature" in data.columns and "wind_speed" in data.columns:
                T = data["temperature"]
                WS = data["wind_speed"] * 3.6  # Convert m/s to km/h
                data["wind_chill"] = 13.12 + 0.6215*T - 11.37*(WS**0.16) + 0.3965*T*(WS**0.16)
            
            # Weather condition encoding
            if "description" in data.columns:
                data["is_clear"] = data["description"].str.lower().str.contains("clear|sunny").astype(int)
                data["is_cloudy"] = data["description"].str.lower().str.contains("cloud").astype(int)
                data["is_rainy"] = data["description"].str.lower().str.contains("rain|drizzle").astype(int)
            
            logger.info("Weather features created successfully")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create weather features: {str(e)}")
        
        return data
    
    @staticmethod
    def create_system_features(
        data: pd.DataFrame,
        system_specs: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create system-specific features
        
        Args:
            data: DataFrame
            system_specs: System specifications
        
        Returns:
            DataFrame with system features
        """
        data = data.copy()
        
        try:
            # Panel degradation
            if "panel_age_years" in system_specs:
                age = system_specs["panel_age_years"]
                data["panel_degradation_factor"] = 1.0 - (age * 0.007)
            
            # Theoretical maximum power
            if "panel_capacity_kw" in system_specs and "panel_degradation_factor" in data.columns:
                capacity = system_specs["panel_capacity_kw"]
                data["theoretical_max_power"] = capacity * data["panel_degradation_factor"]
            
            # Add static system features
            for key, value in system_specs.items():
                if key not in data.columns and isinstance(value, (int, float, str)):
                    data[f"system_{key}"] = value
            
            logger.info("System features created successfully")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create system features: {str(e)}")
        
        return data
    
    @staticmethod
    def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        data = data.copy()
        
        try:
            # Temperature x Hour interaction
            if "temperature" in data.columns and "hour_sin" in data.columns:
                data["temp_hour_interaction"] = data["temperature"] * data["hour_sin"]
            
            # Irradiance x Efficiency
            if "irradiance" in data.columns and "system_panel_efficiency" in data.columns:
                data["expected_generation_interaction"] = (
                    data["irradiance"] * data["system_panel_efficiency"]
                )
            
            # Temperature squared (non-linear)
            if "temperature" in data.columns:
                data["temperature_squared"] = data["temperature"] ** 2
            
            logger.info("Interaction features created successfully")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create interaction features: {str(e)}")
        
        return data


class FeatureScaler:
    """Handles feature scaling"""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize scaler
        
        Args:
            method: Scaling method (standard, minmax, robust)
        """
        self.method = method
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> 'FeatureScaler':
        """
        Fit scaler to data
        
        Args:
            data: Training data
            columns: Columns to scale (default: all numeric)
        
        Returns:
            Self
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        try:
            for col in columns:
                if self.method == "standard":
                    scaler = StandardScaler()
                elif self.method == "minmax":
                    scaler = MinMaxScaler()
                elif self.method == "robust":
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.method}")
                
                scaler.fit(data[[col]])
                self.scalers[col] = scaler
            
            logger.info(f"Scaler fitted for {len(columns)} features using {self.method} method")
            
        except Exception as e:
            raise DataProcessingError(f"Failed to fit scaler: {str(e)}")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data
        
        Args:
            data: Data to transform
        
        Returns:
            Transformed data
        """
        data = data.copy()
        
        try:
            for col, scaler in self.scalers.items():
                if col in data.columns:
                    data[col] = scaler.transform(data[[col]])
            
        except Exception as e:
            raise DataProcessingError(f"Failed to transform data: {str(e)}")
        
        return data
    
    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform"""
        return self.fit(data, columns).transform(data)


class DataPreprocessingPipeline:
    """Complete preprocessing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize pipeline"""
        self.config = config or {}
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.scaler = None
        logger.info("DataPreprocessingPipeline initialized")
    
    def preprocess_solar_data(
        self,
        data: pd.DataFrame,
        system_capacity_kw: float,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing for solar generation data
        
        Args:
            data: Raw solar data
            system_capacity_kw: System capacity
            validate: Whether to validate data
        
        Returns:
            Preprocessed data ready for model
        """
        logger.info("Starting solar data preprocessing")
        
        try:
            # 1. Validation
            if validate:
                is_valid, errors = self.validator.validate_ranges(
                    data,
                    data_type="solar_generation",
                    system_capacity_kw=system_capacity_kw
                )
                if not is_valid:
                    logger.warning(f"Validation warnings: {errors}")
            
            # 2. Missing value handling
            data = self.cleaner.handle_missing_values(data, method="linear", max_gap_hours=2)
            
            # 3. Outlier removal
            data, _ = self.cleaner.detect_and_remove_outliers(data, method="iqr")
            
            # 4. Smoothing
            data = self.cleaner.smooth_noisy_data(data, window_size=3, method="rolling_mean")
            
            # 5. Feature engineering
            data = self.engineer.create_time_features(data)
            data = self.engineer.create_lag_features(
                data,
                columns=["power_kw", "temperature"],
                lags=[1, 24, 168]
            )
            data = self.engineer.create_rolling_features(
                data,
                columns=["power_kw"],
                windows=[6, 24, 168]
            )
            data = self.engineer.create_weather_features(data)
            data = self.engineer.create_system_features(
                data,
                {"panel_capacity_kw": system_capacity_kw}
            )
            data = self.engineer.create_interaction_features(data)
            
            # 6. Remove NaN values (created by lag features)
            initial_rows = len(data)
            data = data.dropna()
            logger.info(f"Rows after NaN removal: {len(data)}/{initial_rows}")
            
            # 7. Scaling
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if self.scaler is None:
                self.scaler = FeatureScaler(method="standard")
                data = self.scaler.fit_transform(data)
            else:
                data = self.scaler.transform(data)
            
            logger.info("Solar data preprocessing completed")
            return data
            
        except Exception as e:
            logger.error(f"Solar preprocessing failed: {str(e)}")
            raise
    
    def preprocess_consumption_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess consumption data"""
        logger.info("Starting consumption data preprocessing")
        
        try:
            # 1. Missing value handling
            data = self.cleaner.handle_missing_values(data, method="seasonal")
            
            # 2. Outlier removal
            data, _ = self.cleaner.detect_and_remove_outliers(data)
            
            # 3. Feature engineering
            data = self.engineer.create_time_features(data)
            data = self.engineer.create_lag_features(data, columns=["power_kw"], lags=[1, 24, 168, 720])
            data = self.engineer.create_rolling_features(data, columns=["power_kw"])
            data = self.engineer.create_weather_features(data)
            data = self.engineer.create_interaction_features(data)
            
            # 4. Remove NaN
            data = data.dropna()
            
            # 5. Scaling
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if self.scaler is None:
                self.scaler = FeatureScaler(method="standard")
                data = self.scaler.fit_transform(data)
            else:
                data = self.scaler.transform(data)
            
            logger.info("Consumption data preprocessing completed")
            return data
            
        except Exception as e:
            logger.error(f"Consumption preprocessing failed: {str(e)}")
            raise
