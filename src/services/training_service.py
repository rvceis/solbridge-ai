"""
Model training service.

Trains solar generation, demand, risk scoring, and anomaly detection models
with MLflow integration for metrics/artifact tracking.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, roc_auc_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.keras

# Setup path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.config.settings import settings
from src.preprocessing.pipeline import DataPreprocessingPipeline

logger = get_logger(__name__)


class ModelTrainer:
    """Base trainer for all models"""
    
    def __init__(self, model_name: str, mlflow_tracking_uri: str = None):
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.preprocessor = DataPreprocessingPipeline()
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for name, value in params.items():
            mlflow.log_param(name, value)
    
    def save_model(self, model, path: str):
        """Save model to disk"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {model_path}")
        # Implementation depends on model type


class SolarGenerationTrainer(ModelTrainer):
    """Train solar generation forecasting model (LSTM + XGBoost ensemble)"""
    
    def __init__(self):
        super().__init__("solar_generation_forecast")
    
    def build_lstm_model(
        self,
        lookback: int = 24,
        n_features: int = 10,
        lstm_units: int = 128,
        dropout_rate: float = 0.2
    ):
        """Build LSTM model for solar forecasting"""
        model = Sequential([
            LSTM(lstm_units, activation='relu', input_shape=(lookback, n_features), return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, activation='relu', return_sequences=False),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='relu')  # Solar output always positive
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(
        self,
        df: pd.DataFrame,
        lookback: int = 24,
        epochs: int = 50,
        batch_size: int = 32,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train LSTM model on solar data.
        
        Assumes df has columns: GHI, Temperature, Hour, DayOfYear, etc.
        """
        with mlflow.start_run(run_name=f"{self.model_name}_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training LSTM for {self.model_name}")
            
            # Feature engineering
            features_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Humidity', 
                           'WindSpeed', 'Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos']
            available_cols = [c for c in features_cols if c in df.columns]
            
            X = df[available_cols].values
            y = df['GHI'].values if 'GHI' in df.columns else df.iloc[:, 0].values
            
            # Create sequences for LSTM
            X_seq, y_seq = self._create_sequences(X, y, lookback)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=test_size, random_state=42
            )
            
            # Log parameters
            params = {
                'lookback': lookback,
                'epochs': epochs,
                'batch_size': batch_size,
                'n_features': len(available_cols),
                'lstm_units': 128,
                'dropout_rate': 0.2
            }
            self.log_params(params)
            
            # Build and train
            model = self.build_lstm_model(lookback, len(available_cols))
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate
            y_pred = model.predict(X_test, verbose=0)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'val_loss': float(history.history['val_loss'][-1])
            }
            self.log_metrics(metrics)
            
            # Save model
            mlflow.keras.log_model(model, "lstm_model")
            model_path = ROOT / f"models/solar_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            model.save(str(model_path))
            
            logger.info(f"LSTM trained: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'test_predictions': y_pred,
                'test_actual': y_test
            }
    
    def train_xgboost(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200
    ) -> Dict[str, Any]:
        """Train XGBoost model as fallback for solar forecasting"""
        with mlflow.start_run(run_name=f"{self.model_name}_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training XGBoost for {self.model_name}")
            
            features_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Humidity',
                           'WindSpeed', 'Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos']
            available_cols = [c for c in features_cols if c in df.columns]
            
            X = df[available_cols]
            y = df['GHI'].values if 'GHI' in df.columns else df.iloc[:, 0].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            params = {
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators
            }
            self.log_params(params)
            
            model = xgb.XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            self.log_metrics(metrics)
            
            mlflow.xgboost.log_model(model, "xgboost_model")
            model_path = ROOT / f"models/solar_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"XGBoost trained: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'test_predictions': y_pred,
                'test_actual': y_test
            }
    
    @staticmethod
    def _create_sequences(X, y, lookback):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        return np.array(X_seq), np.array(y_seq)


class DemandTrainer(ModelTrainer):
    """Train demand/consumption forecasting model"""
    
    def __init__(self):
        super().__init__("demand_forecast")
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train demand forecasting model (XGBoost ensemble)"""
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training {self.model_name}")
            
            features_cols = ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
                           'Month_sin', 'Month_cos', 'Temperature', 'Humidity']
            available_cols = [c for c in features_cols if c in df.columns]
            
            X = df[available_cols]
            y = df['active_power_kw'].values if 'active_power_kw' in df.columns else df.iloc[:, 0].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=200, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            self.log_metrics(metrics)
            
            mlflow.xgboost.log_model(model, "demand_model")
            logger.info(f"Demand model trained: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
            
            return {'model': model, 'metrics': metrics, 'test_predictions': y_pred, 'test_actual': y_test}


class RiskScoringTrainer(ModelTrainer):
    """Train risk scoring model"""
    
    def __init__(self):
        super().__init__("risk_scoring")
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train risk classification model"""
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training {self.model_name}")
            
            # Assume 'risk_label' column exists (0=low, 1=high)
            if 'risk_label' not in df.columns:
                logger.warning("risk_label column missing, skipping risk model training")
                return {}
            
            features_cols = ['volatility', 'price_ratio', 'frequency_anomalies', 'power_factor_deviation']
            available_cols = [c for c in features_cols if c in df.columns]
            
            if not available_cols:
                logger.warning("No risk features found, skipping")
                return {}
            
            X = df[available_cols]
            y = df['risk_label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': float((y_pred == y_test).mean()),
                'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
            }
            self.log_metrics(metrics)
            
            mlflow.sklearn.log_model(model, "risk_model")
            logger.info(f"Risk model trained: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc_roc']:.4f}")
            
            return {'model': model, 'metrics': metrics}


class AnomalyDetectorTrainer(ModelTrainer):
    """Train anomaly detection model"""
    
    def __init__(self):
        super().__init__("anomaly_detection")
    
    def train(self, df: pd.DataFrame, contamination: float = 0.05) -> Dict[str, Any]:
        """Train anomaly detector (Isolation Forest)"""
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training {self.model_name}")
            
            features_cols = ['power_kw', 'voltage_v', 'current_a', 'frequency_hz', 'power_factor']
            available_cols = [c for c in features_cols if c in df.columns]
            
            if not available_cols:
                logger.warning("No anomaly features found, skipping")
                return {}
            
            X = df[available_cols].fillna(df[available_cols].mean())
            
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)
            
            n_anomalies = (predictions == -1).sum()
            metrics = {'anomalies_detected': int(n_anomalies), 'contamination': contamination}
            self.log_metrics(metrics)
            
            mlflow.sklearn.log_model(model, "anomaly_model")
            logger.info(f"Anomaly detector trained: {n_anomalies} anomalies detected")
            
            return {'model': model, 'metrics': metrics, 'predictions': predictions}


def train_all_models(data_dir: str = "ml-service/data/processed") -> Dict[str, Any]:
    """Main training pipeline: load processed data and train all models"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return {}
    
    results = {}
    
    # Load processed files
    csv_files = list(data_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} processed CSV files")
    
    for csv_file in csv_files[:3]:  # Limit to 3 for now
        try:
            logger.info(f"Loading {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            if 'GHI' in df.columns or 'DNI' in df.columns:
                # Solar data
                trainer = SolarGenerationTrainer()
                lstm_result = trainer.train_lstm(df, epochs=20)
                xgb_result = trainer.train_xgboost(df)
                results['solar_lstm'] = lstm_result
                results['solar_xgboost'] = xgb_result
            
            if 'active_power_kw' in df.columns:
                # Demand data
                trainer = DemandTrainer()
                results['demand'] = trainer.train(df)
        
        except Exception as e:
            logger.error(f"Error training on {csv_file.name}: {e}")
    
    logger.info(f"Training complete. Results: {list(results.keys())}")
    return results


if __name__ == "__main__":
    train_all_models()
