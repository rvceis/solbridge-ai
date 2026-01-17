"""Demand Forecasting Models - XGBoost + LSTM Ensemble"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, PredictionError

logger = get_logger(__name__)


class DemandLSTMModel:
    """LSTM for consumption demand forecasting"""
    
    def __init__(
        self,
        input_size: int,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        forecast_horizon: int = 48
    ):
        """Initialize demand LSTM"""
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.model = None
        
        logger.info(f"Initializing DemandLSTMModel: input_size={input_size}")
    
    def build_model(self, lookback_hours: int = 168) -> keras.Model:
        """Build architecture"""
        logger.info(f"Building demand LSTM with lookback={lookback_hours}")
        
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=(lookback_hours, self.input_size)),
                
                # LSTM layers
                keras.layers.LSTM(self.lstm_units, return_sequences=True),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.BatchNormalization(),
                
                keras.layers.LSTM(self.lstm_units // 2, return_sequences=False),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.BatchNormalization(),
                
                # Dense layers
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(self.forecast_horizon, activation="relu")
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae", "mape"]
            )
            
            self.model = model
            logger.info("Demand LSTM built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build demand LSTM: {str(e)}")
            raise ModelLoadError("DemandLSTMModel", str(e))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """Train model"""
        logger.info(f"Training demand LSTM: epochs={epochs}")
        
        try:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True
                )
            ]
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info("Demand LSTM training completed")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ModelLoadError("DemandLSTMModel", "Model not built")
        
        try:
            predictions = self.model.predict(X)
            return np.maximum(predictions, 0)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="DemandLSTMModel")


class DemandXGBoostModel:
    """XGBoost for consumption demand forecasting"""
    
    def __init__(self, n_estimators: int = 300):
        """Initialize XGBoost demand model"""
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.n_estimators = n_estimators
            self.model = None
            logger.info(f"Initializing DemandXGBoostModel with {n_estimators} estimators")
        except ImportError:
            logger.error("XGBoost not installed")
            raise
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict[str, Any]:
        """Train XGBoost demand model"""
        logger.info(f"Training XGBoost demand model on {len(X_train)} samples")
        
        try:
            self.model = self.xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            eval_set = [(X_val, y_val)] if X_val is not None else []
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=15 if eval_set else None,
                verbose=False
            )
            
            logger.info("XGBoost demand training completed")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ModelLoadError("DemandXGBoostModel", "Model not trained")
        
        try:
            predictions = self.model.predict(X)
            return np.maximum(predictions, 0)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="DemandXGBoostModel")


class DemandForecastingEnsemble:
    """Ensemble model for demand forecasting"""
    
    def __init__(
        self,
        xgboost_weight: float = 0.6,
        lstm_weight: float = 0.4
    ):
        """Initialize ensemble"""
        self.xgboost_weight = xgboost_weight
        self.lstm_weight = lstm_weight
        self.xgboost_model = None
        self.lstm_model = None
        
        logger.info(f"Initializing Demand Forecasting Ensemble: XGBoost={xgboost_weight}, LSTM={lstm_weight}")
    
    def predict(
        self,
        X_xgboost: pd.DataFrame,
        X_lstm: np.ndarray
    ) -> np.ndarray:
        """Ensemble prediction"""
        try:
            xgb_pred = self.xgboost_model.predict(X_xgboost) if self.xgboost_model else None
            lstm_pred = self.lstm_model.predict(X_lstm) if self.lstm_model else None
            
            if xgb_pred is not None and lstm_pred is not None:
                # Handle shape differences
                if lstm_pred.ndim == 2:
                    lstm_pred = lstm_pred.mean(axis=1)
                
                ensemble_pred = (
                    self.xgboost_weight * xgb_pred +
                    self.lstm_weight * lstm_pred
                )
            elif xgb_pred is not None:
                ensemble_pred = xgb_pred
            else:
                ensemble_pred = lstm_pred.mean(axis=1) if lstm_pred.ndim == 2 else lstm_pred
            
            logger.info(f"Ensemble demand predictions for {len(ensemble_pred)} samples")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="DemandForecastingEnsemble")


class RandomForestDemandModel:
    """Random Forest as secondary demand forecasting model"""
    
    def __init__(self, n_estimators: int = 200):
        """Initialize Random Forest"""
        self.n_estimators = n_estimators
        self.model = None
        logger.info(f"Initializing RandomForestDemandModel with {n_estimators} trees")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Train Random Forest"""
        logger.info(f"Training Random Forest on {len(X_train)} samples")
        
        try:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            logger.info("Random Forest training completed")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ModelLoadError("RandomForestDemandModel", "Model not trained")
        
        try:
            predictions = self.model.predict(X)
            return np.maximum(predictions, 0)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="RandomForestDemandModel")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            "feature": range(len(self.model.feature_importances_)),
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df.head(20)
