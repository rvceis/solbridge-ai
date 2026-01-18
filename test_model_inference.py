"""
Test inference with the trained XGBoost model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import get_settings
from api.main import ModelManager
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def test_model_inference():
    """
    Test the trained XGBoost model with sample data
    """
    logger.info("=" * 60)
    logger.info("Testing Trained Model Inference")
    logger.info("=" * 60)
    
    # Initialize model manager (will load saved models)
    model_manager = ModelManager()
    
    # Load the trained XGBoost model
    model_path = Path(settings.MODEL_DIR) / "solar_xgboost_model.pkl"
    if model_path.exists():
        logger.info(f"Loading model from: {model_path}")
        model_manager.solar_xgboost.load(str(model_path))
        logger.info("Model loaded successfully ✓")
    else:
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Load preprocessed data to get realistic feature ranges
    csv_path = Path(settings.DATA_DIR) / "processed" / "solar_processed_20260117_131739.csv"
    df = pd.read_csv(csv_path)
    
    # Get sample data from validation set (last 20% of data)
    sample_df = df.iloc[-10:].copy()
    
    logger.info(f"\nSample Input Data (10 samples):")
    logger.info(f"Temperature: {sample_df['Temperature'].values}")
    logger.info(f"GHI: {sample_df['GHI'].values}")
    logger.info(f"DNI: {sample_df['DNI'].values}")
    
    # Prepare features
    X_test = sample_df[['Temperature', 'GHI', 'DNI', 'DHI', 'Cloud Type', 'Solar Zenith Angle']].values
    
    # Make predictions
    logger.info("\nMaking predictions...")
    predictions = model_manager.solar_xgboost.predict(X_test)
    
    logger.info("\nPredictions (power_kw):")
    for i, pred in enumerate(predictions):
        logger.info(f"  Sample {i+1}: {pred:.3f} kW")
    
    logger.info(f"\nPrediction Statistics:")
    logger.info(f"  Min: {predictions.min():.3f} kW")
    logger.info(f"  Max: {predictions.max():.3f} kW")
    logger.info(f"  Mean: {predictions.mean():.3f} kW")
    logger.info(f"  Std: {predictions.std():.3f} kW")
    
    # Test edge cases
    logger.info("\n" + "=" * 60)
    logger.info("Testing Edge Cases")
    logger.info("=" * 60)
    
    # Night time (zero irradiance)
    night_features = np.array([[15.0, 0.0, 0.0, 0.0, 0.0, 90.0]])  # temp, GHI, DNI, DHI, cloud, zenith
    night_pred = model_manager.solar_xgboost.predict(night_features)
    logger.info(f"\nNight (GHI=0): {night_pred[0]:.3f} kW")
    
    # Peak sunshine
    peak_features = np.array([[25.0, 1000.0, 900.0, 100.0, 0.0, 20.0]])
    peak_pred = model_manager.solar_xgboost.predict(peak_features)
    logger.info(f"Peak Sun (GHI=1000): {peak_pred[0]:.3f} kW")
    
    # Cloudy day
    cloudy_features = np.array([[20.0, 300.0, 100.0, 200.0, 8.0, 45.0]])
    cloudy_pred = model_manager.solar_xgboost.predict(cloudy_features)
    logger.info(f"Cloudy (GHI=300): {cloudy_pred[0]:.3f} kW")
    
    logger.info("\n" + "=" * 60)
    logger.info("Model Inference Test Completed Successfully ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_model_inference()
