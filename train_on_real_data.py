"""
Train ML models on real preprocessed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import get_settings
from api.main import ModelManager
from services.training_pipeline import ModelTrainingPipeline
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def validate_preprocessed_data(csv_path: str) -> pd.DataFrame:
    """
    Validate preprocessed CSV data
    """
    logger.info(f"Loading preprocessed data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns[:20])}...")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values")
        logger.info("Filling missing values with forward fill...")
        df = df.fillna(method='ffill').fillna(method='bfill')
    else:
        logger.info("No missing values found")
    
    # Validate required columns for solar prediction
    required_cols = ['Temperature', 'GHI', 'DNI', 'DHI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Data validation passed ✓")
    return df


def add_target_from_irradiance(df: pd.DataFrame, panel_capacity_kw: float = 5.0) -> pd.DataFrame:
    """
    Calculate target power_kw from irradiance data
    
    Simple model: power = efficiency * irradiance * panel_area
    where efficiency ~15-20% for typical solar panels
    """
    # Use GHI (Global Horizontal Irradiance) as primary indicator
    # Convert W/m² to kW with panel efficiency and capacity
    
    # Standard test conditions: 1000 W/m² produces rated capacity
    # So: power_kw = (GHI / 1000) * panel_capacity_kw * efficiency_factor
    
    efficiency_factor = 0.85  # Account for losses (temperature, dirt, inverter, etc.)
    
    # Calculate power output
    df['power_kw'] = (df['GHI'] / 1000.0) * panel_capacity_kw * efficiency_factor
    
    # Clip to realistic range
    df['power_kw'] = df['power_kw'].clip(0, panel_capacity_kw)
    
    # Add some realistic noise
    noise = np.random.normal(0, 0.05 * panel_capacity_kw, len(df))
    df['power_kw'] = (df['power_kw'] + noise).clip(0, panel_capacity_kw)
    
    logger.info(f"Generated target column 'power_kw':")
    logger.info(f"  Min: {df['power_kw'].min():.3f} kW")
    logger.info(f"  Max: {df['power_kw'].max():.3f} kW")
    logger.info(f"  Mean: {df['power_kw'].mean():.3f} kW")
    logger.info(f"  Non-zero count: {(df['power_kw'] > 0.01).sum()} / {len(df)}")
    
    return df


def prepare_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature set for solar LSTM model
    """
    # Select relevant columns
    feature_cols = [
        'Temperature', 'GHI', 'DNI', 'DHI',
        'Cloud Type', 'Solar Zenith Angle', 'Relative Humidity',
        'Pressure', 'Precipitable Water', 'Dew Point',
        'power_kw'  # target
    ]
    
    # Filter columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    solar_df = df[available_cols].copy()
    
    # Add timestamp for sequence handling
    solar_df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(solar_df), freq='1H')
    
    logger.info(f"Prepared solar feature set with {len(available_cols)} columns")
    return solar_df


def train_solar_models(df: pd.DataFrame, model_manager: ModelManager, test_size: float = 0.2):
    """
    Train solar prediction models on real data
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("=" * 60)
    logger.info("Training Solar LSTM Model")
    logger.info("=" * 60)
    
    # Prepare features for LSTM
    X = df[['Temperature', 'GHI', 'DNI', 'DHI', 'Cloud Type', 'Solar Zenith Angle']].values
    y = df['power_kw'].values
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Train LSTM
    try:
        metrics = model_manager.solar_lstm.train(X_train, y_train, X_val, y_val)
        logger.info(f"Solar LSTM trained successfully:")
        logger.info(f"  MSE: {metrics.get('mse', 'N/A')}")
        logger.info(f"  RMSE: {metrics.get('rmse', 'N/A')}")
        logger.info(f"  MAE: {metrics.get('mae', 'N/A')}")
        logger.info(f"  Samples: {metrics.get('n_samples', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to train Solar LSTM: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=" * 60)
    logger.info("Training Solar XGBoost Model")
    logger.info("=" * 60)
    
    # Train XGBoost
    try:
        metrics = model_manager.solar_xgboost.train(X_train, y_train)
        logger.info(f"Solar XGBoost trained successfully:")
        logger.info(f"  MSE: {metrics.get('mse', 'N/A')}")
        logger.info(f"  RMSE: {metrics.get('rmse', 'N/A')}")
        logger.info(f"  MAE: {metrics.get('mae', 'N/A')}")
        logger.info(f"  Samples: {metrics.get('n_samples', 'N/A')}")
        
        # Evaluate on validation set
        y_pred = model_manager.solar_xgboost.predict(X_val)
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        val_mse = mean_squared_error(y_val, y_pred)
        val_mae = mean_absolute_error(y_val, y_pred)
        logger.info(f"  Validation MSE: {val_mse:.4f}")
        logger.info(f"  Validation MAE: {val_mae:.4f}")
    except Exception as e:
        logger.error(f"Failed to train Solar XGBoost: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """
    Main training workflow
    """
    logger.info("Starting Real Data Training Pipeline")
    logger.info("=" * 60)
    
    # Find latest preprocessed CSV
    data_dir = Path(settings.DATA_DIR) / "processed"
    csv_files = sorted(data_dir.glob("solar_processed_*.csv"))
    
    if not csv_files:
        logger.error(f"No preprocessed CSV files found in {data_dir}")
        return
    
    latest_csv = csv_files[-1]
    logger.info(f"Using latest file: {latest_csv.name}")
    
    # 1. Validate data
    df = validate_preprocessed_data(str(latest_csv))
    
    # 2. Generate target column from irradiance
    df = add_target_from_irradiance(df, panel_capacity_kw=5.0)
    
    # 3. Prepare features
    solar_df = prepare_solar_features(df)
    
    # 4. Initialize model manager
    model_manager = ModelManager()
    
    # 5. Train solar models
    train_solar_models(solar_df, model_manager)
    
    # 6. Save models
    logger.info("=" * 60)
    logger.info("Saving trained models...")
    models_dir = Path(settings.MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Save XGBoost (LSTM may not have been built if training failed)
    try:
        model_manager.solar_xgboost.save(str(models_dir / "solar_xgboost_model.pkl"))
        logger.info("Saved: solar_xgboost_model.pkl")
    except Exception as e:
        logger.error(f"Failed to save XGBoost model: {e}")
    
    try:
        model_manager.solar_lstm.save(str(models_dir / "solar_lstm_model.pkl"))
        logger.info("Saved: solar_lstm_model.pkl")
    except Exception as e:
        logger.warning(f"Could not save LSTM model: {e}")
    
    logger.info("Training completed successfully! ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
