"""LSTM Solar Generation Forecasting Model"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import joblib
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, PredictionError, DataProcessingError

logger = get_logger(__name__)

# Lazy import TensorFlow to avoid startup failure if not installed
def _import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        return tf, keras, layers
    except ImportError:
        logger.warning("TensorFlow not installed - LSTM models will not be available")
        return None, None, None


class SolarLSTMModel:
    """LSTM-based solar generation forecasting"""
    
    def __init__(
        self,
        input_size: int,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        forecast_horizon: int = 48
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            forecast_horizon: Number of hours to forecast
        """
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        
        logger.info(f"Initializing SolarLSTMModel: input_size={input_size}, lstm_units={lstm_units}")
    
    def build_model(self, lookback_hours: int = 168):
        """
        Build LSTM architecture
        
        Args:
            lookback_hours: Historical window size
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model with lookback={lookback_hours}")
        
        try:
            tf, keras, layers = _import_tensorflow()
            if keras is None:
                raise ModelLoadError("SolarLSTMModel", "TensorFlow not installed")
            
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=(lookback_hours, self.input_size)),
                
                # First LSTM layer with dropout
                layers.LSTM(self.lstm_units, return_sequences=True),
                layers.Dropout(self.dropout_rate),
                layers.BatchNormalization(),
                
                # Second LSTM layer
                layers.LSTM(self.lstm_units // 2, return_sequences=True),
                layers.Dropout(self.dropout_rate),
                layers.BatchNormalization(),
                
                # Third LSTM layer
                layers.LSTM(self.lstm_units // 4, return_sequences=False),
                layers.Dropout(self.dropout_rate),
                layers.BatchNormalization(),
                
                # Dense layers
                layers.Dense(64, activation="relu"),
                layers.Dropout(self.dropout_rate),
                
                layers.Dense(32, activation="relu"),
                layers.Dropout(self.dropout_rate),
                
                # Output layer (forecast horizon predictions)
                layers.Dense(self.forecast_horizon, activation="relu")
            ])
            
            # Compile
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae", "mape"]
            )
            
            self.model = model
            logger.info("LSTM model built successfully")
            logger.info(f"Model parameters: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to build LSTM model: {str(e)}")
            raise ModelLoadError("SolarLSTMModel", str(e))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            verbose: Logging verbosity
        
        Returns:
            Training history
        """
        logger.info(f"Starting LSTM training: epochs={epochs}, batch_size={batch_size}")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        try:
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1
            )
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=verbose
            )
            
            logger.info("Training completed successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Input data (shape: samples x lookback_hours x features)
        
        Returns:
            (predictions, confidence_intervals)
        """
        if self.model is None:
            raise ModelLoadError("SolarLSTMModel", "Model not built")
        
        try:
            predictions = self.model.predict(X)
            
            # Ensure positive values
            predictions = np.maximum(predictions, 0)
            
            # Calculate confidence intervals (90% confidence)
            confidence_intervals = self._calculate_confidence_intervals(predictions)
            
            logger.info(f"Predictions generated for {len(X)} samples")
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="SolarLSTMModel")
    
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence: float = 0.9
    ) -> np.ndarray:
        """
        Calculate confidence intervals
        
        Args:
            predictions: Model predictions
            confidence: Confidence level
        
        Returns:
            Confidence intervals (lower, upper bounds)
        """
        # Estimate uncertainty as ±20% of prediction
        uncertainty = predictions * 0.2
        lower = np.maximum(predictions - uncertainty * 1.645, 0)
        upper = predictions + uncertainty * 1.645
        
        return np.stack([lower, upper], axis=-1)
    
    def save(self, filepath: str):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Load model"""
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError("SolarLSTMModel", str(e))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"status": "not_built"}
        
        return {
            "architecture": "LSTM",
            "input_size": self.input_size,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "forecast_horizon": self.forecast_horizon,
            "parameters": self.model.count_params(),
            "layers": len(self.model.layers)
        }


class SolarXGBoostModel:
    """XGBoost fallback model for solar forecasting"""
    
    def __init__(self, n_estimators: int = 500):
        """Initialize XGBoost model"""
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.n_estimators = n_estimators
            self.model = None
            logger.info(f"Initializing SolarXGBoostModel with {n_estimators} estimators")
        except ImportError:
            logger.error("XGBoost not installed")
            raise
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            Training history
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples")
        
        try:
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            self.model = self.xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=1
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set if eval_set else None,
                verbose=False
            )
            
            logger.info("XGBoost training completed")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ModelLoadError("SolarXGBoostModel", "Model not trained")
        
        try:
            predictions = self.model.predict(X)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="SolarXGBoostModel")
    
    def save(self, filepath: str):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            self.model.save_model(filepath)
            logger.info(f"XGBoost model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Load model"""
        try:
            self.model = self.xgb.XGBRegressor()
            self.model.load_model(filepath)
            logger.info(f"XGBoost model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError("SolarXGBoostModel", str(e))


class SolarForecastingEnsemble:
    """Ensemble of LSTM and XGBoost for solar forecasting"""
    
    def __init__(
        self,
        lstm_weight: float = 0.6,
        xgboost_weight: float = 0.4
    ):
        """Initialize ensemble"""
        self.lstm_weight = lstm_weight
        self.xgboost_weight = xgboost_weight
        self.lstm_model = None
        self.xgboost_model = None
        
        logger.info(f"Initializing Solar Forecasting Ensemble: LSTM={lstm_weight}, XGBoost={xgboost_weight}")
    
    def predict(
        self,
        X_lstm: np.ndarray,
        X_xgboost: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction
        
        Args:
            X_lstm: Data for LSTM model
            X_xgboost: Data for XGBoost model
        
        Returns:
            (ensemble_predictions, confidence_intervals)
        """
        try:
            # Get predictions from both models
            lstm_pred, lstm_ci = self.lstm_model.predict(X_lstm) if self.lstm_model else (None, None)
            xgb_pred = self.xgboost_model.predict(X_xgboost) if self.xgboost_model else None
            
            # Weighted ensemble
            if lstm_pred is not None and xgb_pred is not None:
                # Ensure same shape
                if lstm_pred.ndim == 2:
                    ensemble_pred = (
                        self.lstm_weight * lstm_pred.mean(axis=1) +
                        self.xgboost_weight * xgb_pred
                    )
                else:
                    ensemble_pred = (
                        self.lstm_weight * lstm_pred +
                        self.xgboost_weight * xgb_pred
                    )
            elif lstm_pred is not None:
                ensemble_pred = lstm_pred.mean(axis=1) if lstm_pred.ndim == 2 else lstm_pred
            else:
                ensemble_pred = xgb_pred
            
            logger.info(f"Ensemble predictions generated for {len(ensemble_pred)} samples")
            return ensemble_pred, lstm_ci
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="SolarForecastingEnsemble")
