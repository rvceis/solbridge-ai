#!/usr/bin/env python3
"""
Standalone model training script - trains all models with synthetic data and saves them
Run this locally before deploying to populate the models directory
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime

from src.models.solar_forecast import SolarXGBoostModel, SolarLSTMModel
from src.models.demand_forecast import DemandXGBoostModel, DemandLSTMModel
from src.models.advanced_models import (
    DynamicPricingModel,
    InvestorRiskScoringModel,
    AnomalyDetectionModel,
    EquipmentFailurePredictorModel
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_solar_xgboost():
    """Train Solar XGBoost model"""
    logger.info("Training Solar XGBoost model...")
    try:
        # Generate synthetic data
        num_samples = 2000
        X_data = {
            "temperature": np.random.uniform(15, 45, num_samples),
            "humidity": np.random.uniform(30, 80, num_samples),
            "cloud_cover": np.random.uniform(0, 100, num_samples),
            "hour": np.random.randint(0, 24, num_samples),
            "day_of_week": np.random.randint(0, 7, num_samples),
            "month": np.random.randint(1, 13, num_samples),
        }
        X = pd.DataFrame(X_data)
        
        # Create realistic target (solar generation pattern)
        y = pd.Series([
            max(0, 5.0 * max(0, np.sin((x - 6) * np.pi / 12)) * (1 - c / 200) + np.random.normal(0, 0.1))
            for x, c in zip(X["hour"], X["cloud_cover"])
        ])
        
        # Train
        model = SolarXGBoostModel(n_estimators=100)
        model.train(X, y)
        
        # Save
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model.save(str(models_dir / "solar_xgboost_model.pkl"))
        logger.info("✓ Solar XGBoost saved")
        return True
    except Exception as e:
        logger.error(f"Solar XGBoost training failed: {e}", exc_info=True)
        return False


def train_demand_xgboost():
    """Train Demand XGBoost model"""
    logger.info("Training Demand XGBoost model...")
    try:
        # Generate synthetic data
        num_samples = 2000
        X_data = {
            "temperature": np.random.uniform(15, 45, num_samples),
            "humidity": np.random.uniform(30, 80, num_samples),
            "hour": np.random.randint(0, 24, num_samples),
            "day_of_week": np.random.randint(0, 7, num_samples),
            "is_weekend": np.random.randint(0, 2, num_samples),
            "month": np.random.randint(1, 13, num_samples),
        }
        X = pd.DataFrame(X_data)
        
        # Create realistic target (demand pattern)
        y = pd.Series([
            max(0.5, 2.0 + 1.5 * max(0, np.sin((h - 7) * np.pi / 12)) + np.random.normal(0, 0.2))
            for h in X["hour"]
        ])
        
        # Train
        model = DemandXGBoostModel(n_estimators=100)
        model.train(X, y)
        
        # Save
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model.save(str(models_dir / "demand_xgboost_model.pkl"))
        logger.info("✓ Demand XGBoost saved")
        return True
    except Exception as e:
        logger.error(f"Demand XGBoost training failed: {e}", exc_info=True)
        return False


def train_pricing_model():
    """Train Dynamic Pricing model"""
    logger.info("Training Dynamic Pricing model...")
    try:
        num_samples = 500
        X = pd.DataFrame({
            "supply": np.random.uniform(10, 100, num_samples),
            "demand": np.random.uniform(20, 80, num_samples),
            "supply_demand_ratio": np.random.uniform(0.3, 3.0, num_samples),
            "hour": np.random.randint(0, 24, num_samples),
            "grid_tariff": np.random.uniform(4, 8, num_samples),
        })
        
        # Price increases with demand ratio
        y = pd.Series([6 + (sr * 2) + np.random.normal(0, 0.3) for sr in X["supply_demand_ratio"]])
        y = pd.Series([min(12, max(4, p)) for p in y])
        
        model = DynamicPricingModel()
        model.train(X, y)
        logger.info("✓ Pricing model trained")
        return True
    except Exception as e:
        logger.error(f"Pricing training failed: {e}", exc_info=True)
        return False


def train_risk_model():
    """Train Investor Risk Scoring model"""
    logger.info("Training Risk Scoring model...")
    try:
        num_samples = 500
        X = pd.DataFrame({
            "location_latitude": np.random.uniform(8, 35, num_samples),
            "location_longitude": np.random.uniform(68, 97, num_samples),
            "financial_history_score": np.random.uniform(400, 900, num_samples),
            "system_age_years": np.random.uniform(0.5, 20, num_samples),
            "system_capacity_kw": np.random.uniform(3, 20, num_samples),
            "installation_quotes_received": np.random.randint(1, 10, num_samples),
        })
        
        # Risk: higher score = higher risk
        score = 0.3 * (900 - X["financial_history_score"]) / 500 + 0.3 * (X["system_age_years"] / 20)
        y = pd.Series([int(np.clip(s * 3, 0, 2)) for s in score])
        
        model = InvestorRiskScoringModel()
        model.train(X, y)
        logger.info("✓ Risk model trained")
        return True
    except Exception as e:
        logger.error(f"Risk training failed: {e}", exc_info=True)
        return False


def train_anomaly_model():
    """Train Anomaly Detection model"""
    logger.info("Training Anomaly Detection model...")
    try:
        num_samples = 1000
        X = pd.DataFrame({
            "voltage": np.random.normal(230, 5, num_samples),
            "current": np.random.normal(10, 2, num_samples),
            "temperature": np.random.normal(30, 3, num_samples),
        })
        
        model = AnomalyDetectionModel()
        model.train(X, contamination=0.05)
        logger.info("✓ Anomaly model trained")
        return True
    except Exception as e:
        logger.error(f"Anomaly training failed: {e}", exc_info=True)
        return False


def train_failure_model():
    """Train Equipment Failure Predictor model"""
    logger.info("Training Failure Predictor model...")
    try:
        num_samples = 800
        X = pd.DataFrame({
            "vibration": np.random.normal(0.5, 0.2, num_samples),
            "temperature": np.random.normal(35, 4, num_samples),
            "age_months": np.random.uniform(1, 60, num_samples),
        })
        
        # Failure risk increases with vibration, temp, age
        risk_score = (0.4 * X["vibration"] + 
                     0.3 * (X["temperature"] - 30) / 15 + 
                     0.3 * (X["age_months"] / 60))
        y = pd.Series((risk_score > 0.7).astype(int))
        
        model = EquipmentFailurePredictorModel()
        model.train(X, y)
        logger.info("✓ Failure model trained")
        return True
    except Exception as e:
        logger.error(f"Failure training failed: {e}", exc_info=True)
        return False


def main():
    """Train all models"""
    logger.info("=" * 80)
    logger.info("STANDALONE MODEL TRAINING")
    logger.info("=" * 80)
    
    results = {
        "solar_xgboost": train_solar_xgboost(),
        "demand_xgboost": train_demand_xgboost(),
        "pricing": train_pricing_model(),
        "risk_scoring": train_risk_model(),
        "anomaly_detection": train_anomaly_model(),
        "failure_prediction": train_failure_model(),
    }
    
    logger.info("=" * 80)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 80)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{model_name}: {status}")
    
    trained_count = sum(1 for v in results.values() if v)
    logger.info(f"Total trained: {trained_count}/{len(results)}")
    logger.info("=" * 80)
    
    if trained_count == len(results):
        logger.info("All models trained successfully!")
        return 0
    else:
        logger.warning(f"Only {trained_count}/{len(results)} models trained")
        return 1


if __name__ == "__main__":
    sys.exit(main())
