"""
Model Initializer - Creates basic trained models on first startup if missing
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_training_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data for initial model training"""
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Solar generation data
    solar_data = []
    for ts in timestamps:
        hour = ts.hour
        # Simulate solar generation pattern (peak at noon)
        base_generation = max(0, np.sin((hour - 6) * np.pi / 12)) * 5
        noise = np.random.normal(0, 0.5)
        power = max(0, base_generation + noise)
        
        solar_data.append({
            'timestamp': ts,
            'power_kw': power,
            'temperature': 25 + np.random.normal(0, 5),
            'humidity': 60 + np.random.normal(0, 10),
            'cloud_cover': np.random.uniform(0, 100),
            'hour': hour,
            'day_of_week': ts.weekday(),
            'month': ts.month
        })
    
    # Demand data
    demand_data = []
    for ts in timestamps:
        hour = ts.hour
        # Simulate demand pattern (peak in evening)
        if 6 <= hour <= 10 or 17 <= hour <= 22:
            base_demand = 3.0 + np.random.normal(0, 0.5)
        elif 11 <= hour <= 16:
            base_demand = 2.0 + np.random.normal(0, 0.3)
        else:
            base_demand = 1.0 + np.random.normal(0, 0.2)
        
        demand_data.append({
            'timestamp': ts,
            'power_kw': max(0.5, base_demand),
            'temperature': 25 + np.random.normal(0, 5),
            'humidity': 60 + np.random.normal(0, 10),
            'hour': hour,
            'day_of_week': ts.weekday(),
            'is_weekend': 1 if ts.weekday() >= 5 else 0,
            'month': ts.month
        })
    
    return pd.DataFrame(solar_data), pd.DataFrame(demand_data)


def train_initial_solar_model(models_dir: Path) -> bool:
    """Train a basic solar forecasting model"""
    try:
        logger.info("Training initial Solar XGBoost model...")
        
        from src.models.solar_forecast import SolarXGBoostModel
        
        # Generate synthetic training data
        solar_data, _ = generate_synthetic_training_data(2000)
        
        # Prepare features and targets
        feature_cols = ['temperature', 'humidity', 'cloud_cover', 'hour', 'day_of_week', 'month']
        X = solar_data[feature_cols].values
        y = solar_data['power_kw'].values
        
        # Train model
        model = SolarXGBoostModel(n_estimators=100)
        model.train(X, y)
        
        # Save model
        model_path = models_dir / "solar_xgboost_model.pkl"
        model.save(str(model_path))
        
        logger.info(f"✓ Solar XGBoost model trained and saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to train solar model: {e}", exc_info=True)
        return False


def train_initial_demand_model(models_dir: Path) -> bool:
    """Train a basic demand forecasting model"""
    try:
        logger.info("Training initial Demand XGBoost model...")
        
        from src.models.demand_forecast import DemandXGBoostModel
        
        # Generate synthetic training data
        _, demand_data = generate_synthetic_training_data(2000)
        
        # Prepare features and targets
        feature_cols = ['temperature', 'humidity', 'hour', 'day_of_week', 'is_weekend', 'month']
        X = demand_data[feature_cols].values
        y = demand_data['power_kw'].values
        
        # Train model
        model = DemandXGBoostModel(n_estimators=100)
        model.train(X, y)
        
        # Save model
        model_path = models_dir / "demand_xgboost_model.pkl"
        model.save(str(model_path))
        
        logger.info(f"✓ Demand XGBoost model trained and saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to train demand model: {e}", exc_info=True)
        return False


def initialize_models_if_missing(models_dir: Path) -> dict:
    """Initialize models if they don't exist"""
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'solar_xgboost': False,
        'demand_xgboost': False
    }
    
    # Check and train solar model if missing
    solar_model_path = models_dir / "solar_xgboost_model.pkl"
    if not solar_model_path.exists():
        logger.info("Solar XGBoost model not found. Training initial model...")
        results['solar_xgboost'] = train_initial_solar_model(models_dir)
    else:
        logger.info(f"Solar XGBoost model found at {solar_model_path}")
        results['solar_xgboost'] = True
    
    # Check and train demand model if missing
    demand_model_path = models_dir / "demand_xgboost_model.pkl"
    if not demand_model_path.exists():
        logger.info("Demand XGBoost model not found. Training initial model...")
        results['demand_xgboost'] = train_initial_demand_model(models_dir)
    else:
        logger.info(f"Demand XGBoost model found at {demand_model_path}")
        results['demand_xgboost'] = True
    
    return results
