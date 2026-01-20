"""Dynamic Pricing, Risk Scoring, and Anomaly Detection Models"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, PredictionError

logger = get_logger(__name__)


class DynamicPricingModel:
    """Dynamic pricing using supply-demand optimization"""
    
    def __init__(self):
        """Initialize pricing model"""
        self.model = None
        self.supply_demand_model = None
        self.scaler = StandardScaler()
        logger.info("Initializing DynamicPricingModel")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train pricing model
        
        Args:
            X_train: Features (supply, demand, time, etc.)
            y_train: Target (price in ₹/kWh)
        
        Returns:
            Training info
        """
        logger.info(f"Training DynamicPricingModel on {len(X_train)} samples")
        
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
            
            self.model.fit(X_scaled, y_train)
            logger.info("Dynamic pricing model trained")
            
            # Calculate metrics
            train_r2 = self.model.score(X_scaled, y_train)
            
            return {
                "status": "success",
                "train_r2": train_r2,
                "model_type": "GradientBoostingRegressor"
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict prices
        
        Args:
            X: Features for prediction
        
        Returns:
            Predicted prices in ₹/kWh
        """
        if self.model is None:
            raise ModelLoadError("DynamicPricingModel", "Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            prices = self.model.predict(X_scaled)
            
            # Ensure reasonable bounds: 4-12 ₹/kWh
            prices = np.clip(prices, 4, 12)
            
            return prices
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="DynamicPricingModel")
    
    def calculate_optimal_trading_hours(
        self,
        demand_forecast: pd.Series,
        supply_forecast: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate optimal trading times based on supply-demand
        
        Args:
            demand_forecast: 24-hour demand forecast
            supply_forecast: 24-hour supply forecast
        
        Returns:
            DataFrame with optimal hours and prices
        """
        try:
            hours_df = pd.DataFrame({
                'hour': range(24),
                'demand': demand_forecast,
                'supply': supply_forecast
            })
            
            # Calculate supply-demand ratio
            hours_df['supply_demand_ratio'] = hours_df['supply'] / (hours_df['demand'] + 0.1)
            
            # High demand, low supply = premium hours
            hours_df['price_multiplier'] = np.where(
                hours_df['supply_demand_ratio'] < 0.5,
                1.5,  # +50% premium
                np.where(
                    hours_df['supply_demand_ratio'] > 2,
                    0.7,  # -30% discount
                    1.0   # neutral
                )
            )
            
            # Identify optimal trading hours (balanced supply-demand)
            hours_df['is_optimal'] = (
                (hours_df['supply_demand_ratio'] >= 0.8) &
                (hours_df['supply_demand_ratio'] <= 1.2)
            ).astype(int)
            
            return hours_df[['hour', 'supply_demand_ratio', 'price_multiplier', 'is_optimal']]
            
        except Exception as e:
            logger.error(f"Optimal trading hours calculation failed: {str(e)}")
            raise


class InvestorRiskScoringModel:
    """Risk scoring for solar panel investors"""
    
    def __init__(self):
        """Initialize risk scoring model"""
        self.model = None
        self.scaler = StandardScaler()
        logger.info("Initializing InvestorRiskScoringModel")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train risk scoring model
        
        Args:
            X_train: Features (location, financial history, system specs, etc.)
            y_train: Risk labels (0=low, 1=medium, 2=high)
        
        Returns:
            Training info
        """
        logger.info(f"Training InvestorRiskScoringModel on {len(X_train)} samples")
        
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_scaled, y_train)
            logger.info("Risk scoring model trained")
            
            return {
                "status": "success",
                "classes": list(self.model.classes_),
                "model_type": "RandomForestClassifier"
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict risk scores
        
        Args:
            X: Features for prediction
        
        Returns:
            (risk_class, risk_probability)
        """
        if self.model is None:
            raise ModelLoadError("InvestorRiskScoringModel", "Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            risk_class = self.model.predict(X_scaled)
            risk_probs = self.model.predict_proba(X_scaled).max(axis=1)
            
            return risk_class, risk_probs
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="InvestorRiskScoringModel")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            "feature_index": range(len(self.model.feature_importances_)),
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df.head(15)


class AnomalyDetectionModel:
    """Anomaly detection for equipment failures"""
    
    def __init__(self):
        """Initialize anomaly detection"""
        self.model = None
        self.scaler = StandardScaler()
        logger.info("Initializing AnomalyDetectionModel")
    
    def train(
        self,
        X_train: pd.DataFrame,
        contamination: float = 0.05
    ) -> Dict[str, Any]:
        """
        Train anomaly detection model
        
        Args:
            X_train: Normal system data
            contamination: Expected anomaly fraction
        
        Returns:
            Training info
        """
        logger.info(f"Training AnomalyDetectionModel on {len(X_train)} samples")
        
        try:
            from sklearn.ensemble import IsolationForest
            
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_scaled)
            logger.info("Anomaly detection model trained")
            
            return {
                "status": "success",
                "contamination": contamination,
                "model_type": "IsolationForest"
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies
        
        Args:
            X: System data to check
        
        Returns:
            (anomaly_flags, anomaly_scores)
        """
        if self.model is None:
            raise ModelLoadError("AnomalyDetectionModel", "Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # -1 = anomaly, 1 = normal
            anomaly_flags = self.model.predict(X_scaled)
            anomaly_flags = (anomaly_flags == -1).astype(int)
            
            # Anomaly scores
            anomaly_scores = -self.model.score_samples(X_scaled)
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)
            
            return anomaly_flags, anomaly_scores
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="AnomalyDetectionModel")
    
    def detect_degradation(
        self,
        efficiency_history: np.ndarray,
        threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Detect equipment degradation
        
        Args:
            efficiency_history: Historical efficiency values
            threshold: Degradation threshold (2% per year default)
        
        Returns:
            Degradation analysis
        """
        try:
            if len(efficiency_history) < 30:
                return {"status": "insufficient_data", "days": len(efficiency_history)}
            
            # Linear regression to find trend
            X = np.arange(len(efficiency_history)).reshape(-1, 1)
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, efficiency_history)
            
            slope = model.coef_[0]
            current_eff = efficiency_history[-1]
            
            # Annual degradation rate
            annual_degradation = slope * 365
            degradation_rate = annual_degradation / current_eff if current_eff > 0 else 0
            
            is_degrading = degradation_rate < -threshold
            
            return {
                "status": "success",
                "current_efficiency": current_eff,
                "daily_degradation": slope,
                "annual_degradation_rate": degradation_rate,
                "is_degrading": is_degrading,
                "days_analyzed": len(efficiency_history)
            }
            
        except Exception as e:
            logger.error(f"Degradation detection failed: {str(e)}")
            raise
    
    def detect_voltage_anomaly(
        self,
        voltage_readings: np.ndarray,
        min_v: float = 200,
        max_v: float = 260
    ) -> Tuple[int, str]:
        """
        Detect voltage anomalies
        
        Args:
            voltage_readings: Voltage readings
            min_v: Minimum acceptable voltage
            max_v: Maximum acceptable voltage
        
        Returns:
            (anomaly_count, status_message)
        """
        try:
            out_of_range = (
                (voltage_readings < min_v) | (voltage_readings > max_v)
            ).sum()
            
            percentage = (out_of_range / len(voltage_readings)) * 100
            
            if percentage > 10:
                status = "CRITICAL: Frequent voltage anomalies"
                severity = "high"
            elif percentage > 5:
                status = "WARNING: Some voltage anomalies detected"
                severity = "medium"
            else:
                status = "OK: Voltage within acceptable range"
                severity = "low"
            
            return out_of_range, f"{status} ({percentage:.1f}%)"
            
        except Exception as e:
            logger.error(f"Voltage anomaly detection failed: {str(e)}")
            raise


class EquipmentFailurePredictorModel:
    """Predict equipment failure risk"""
    
    def __init__(self):
        """Initialize failure predictor"""
        self.model = None
        self.scaler = StandardScaler()
        logger.info("Initializing EquipmentFailurePredictorModel")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Train failure prediction model"""
        logger.info(f"Training EquipmentFailurePredictorModel on {len(X_train)} samples")
        
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_scaled, y_train)
            logger.info("Failure prediction model trained")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure probability"""
        if self.model is None:
            raise ModelLoadError("EquipmentFailurePredictorModel", "Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            failure_probs = self.model.predict_proba(X_scaled)[:, 1]
            return failure_probs
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), model_name="EquipmentFailurePredictorModel")
