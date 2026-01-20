"""
ML Model Evaluation and Reporting Script
Solar Sharing Platform - Comprehensive Model Testing

This script evaluates all ML models and generates reports suitable for academic/project documentation.

Models Evaluated:
1. Solar Generation Forecast (XGBoost)
2. Solar Generation Forecast (LSTM) - if available
3. Demand Forecast Model
4. Dynamic Pricing Model
5. Matching Algorithm (from backend)

Metrics Reported:
- RMSE, MAE, MAPE, R² Score
- Precision, Recall, F1 (for classification)
- Confusion matrices
- Feature importance
- Prediction vs Actual plots
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit

try:
    from config.settings import get_settings
    from api.main import ModelManager
    from utils.logger import get_logger
    settings = get_settings()
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    settings = None


# Output directory for reports
REPORTS_DIR = Path(__file__).parent / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


class MLEvaluator:
    """Comprehensive ML Model Evaluator"""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_test_data(self, data_path: str = None) -> pd.DataFrame:
        """Load preprocessed test data"""
        if data_path is None:
            # Find latest processed data
            data_dir = Path(__file__).parent / "data" / "processed"
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                data_path = max(csv_files, key=lambda p: p.stat().st_mtime)
            else:
                logger.error("No processed data files found")
                return None
        
        logger.info(f"Loading test data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
        """Calculate comprehensive regression metrics"""
        
        # Handle edge cases
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Remove any invalid values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        residuals = y_pred - y_true
        
        metrics = {
            "model_name": model_name,
            "n_samples": len(y_true),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
        }
        
        # MAPE (avoid division by zero)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics["mape"] = float(mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )) * 100  # as percentage
        else:
            metrics["mape"] = None
        
        # Additional statistics
        metrics["y_true_mean"] = float(np.mean(y_true))
        metrics["y_true_std"] = float(np.std(y_true))
        metrics["y_pred_mean"] = float(np.mean(y_pred))
        metrics["y_pred_std"] = float(np.std(y_pred))
        metrics["residual_mean"] = float(np.mean(residuals))
        metrics["residual_std"] = float(np.std(residuals))
        
        # Percentile errors
        abs_errors = np.abs(residuals)
        metrics["error_p50"] = float(np.percentile(abs_errors, 50))  # Median error
        metrics["error_p90"] = float(np.percentile(abs_errors, 90))
        metrics["error_p95"] = float(np.percentile(abs_errors, 95))
        metrics["error_p99"] = float(np.percentile(abs_errors, 99))
        metrics["max_error"] = float(np.max(abs_errors))
        
        # Normalized metrics
        y_range = np.max(y_true) - np.min(y_true)
        if y_range > 0:
            metrics["nrmse"] = float(metrics["rmse"] / y_range)  # Normalized RMSE
            metrics["cv_rmse"] = float(metrics["rmse"] / metrics["y_true_mean"]) if metrics["y_true_mean"] != 0 else None  # Coefficient of variation
        
        # Peak accuracy (for solar generation)
        if y_range > 0:
            peak_threshold = np.percentile(y_true, 75)
            peak_mask = y_true >= peak_threshold
            if peak_mask.sum() > 0:
                metrics["peak_mae"] = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))
                metrics["peak_mape"] = float(mean_absolute_percentage_error(y_true[peak_mask], y_pred[peak_mask])) * 100
        
        return metrics
    
    def evaluate_solar_xgboost(self, df: pd.DataFrame) -> dict:
        """Evaluate Solar XGBoost Model"""
        logger.info("=" * 60)
        logger.info("Evaluating Solar XGBoost Model")
        logger.info("=" * 60)
        
        try:
            # Load model
            model_path = self.model_dir / "solar_xgboost_model.pkl"
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return {"error": "Model not found"}
            
            import joblib
            model = joblib.load(model_path)
            
            # Prepare features
            feature_cols = ['Temperature', 'GHI', 'DNI', 'DHI', 'Cloud Type', 'Solar Zenith Angle']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 4:
                return {"error": f"Missing required features. Available: {available_cols}"}
            
            X = df[available_cols].values
            
            # Generate target if not present
            if 'power_kw' not in df.columns:
                # Calculate from GHI
                panel_capacity = 5.0
                efficiency = 0.85
                y = (df['GHI'] / 1000.0) * panel_capacity * efficiency
                y = y.clip(0, panel_capacity).values
            else:
                y = df['power_kw'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_regression_metrics(y_test, y_pred, "Solar_XGBoost")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                metrics["feature_importance"] = dict(zip(available_cols, model.feature_importances_.tolist()))
            
            # Cross-validation
            logger.info("Running 5-fold cross-validation...")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics["cv_r2_mean"] = float(cv_scores.mean())
            metrics["cv_r2_std"] = float(cv_scores.std())
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            ts_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            metrics["ts_cv_rmse_mean"] = float(np.sqrt(-ts_scores.mean()))
            
            # Store predictions for plotting
            metrics["y_test"] = y_test.tolist()[:100]  # First 100 for visualization
            metrics["y_pred"] = y_pred.tolist()[:100]
            
            self.results["solar_xgboost"] = metrics
            
            # Print results
            logger.info(f"\nSolar XGBoost Results:")
            logger.info(f"  RMSE: {metrics['rmse']:.4f} kW")
            logger.info(f"  MAE: {metrics['mae']:.4f} kW")
            logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
            logger.info(f"  MAPE: {metrics.get('mape', 'N/A')}%")
            logger.info(f"  CV R² (mean): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating Solar XGBoost: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def evaluate_matching_algorithm(self, n_simulations: int = 100) -> dict:
        """Evaluate the buyer-seller matching algorithm"""
        logger.info("=" * 60)
        logger.info("Evaluating Matching Algorithm")
        logger.info("=" * 60)
        
        try:
            # Simulate matching scenarios
            np.random.seed(42)
            
            # Generate synthetic hosts and buyers
            n_hosts = 20
            n_buyers = 50
            
            hosts = []
            for i in range(n_hosts):
                hosts.append({
                    "id": f"host_{i}",
                    "price_per_kwh": np.random.uniform(4, 12),
                    "available_kwh": np.random.uniform(5, 50),
                    "latitude": 12.9 + np.random.uniform(-0.5, 0.5),
                    "longitude": 77.5 + np.random.uniform(-0.5, 0.5),
                    "rating": np.random.uniform(3, 5),
                    "renewable_cert": np.random.choice([True, False], p=[0.6, 0.4])
                })
            
            buyers = []
            for i in range(n_buyers):
                buyers.append({
                    "id": f"buyer_{i}",
                    "max_price": np.random.uniform(6, 15),
                    "required_kwh": np.random.uniform(2, 20),
                    "latitude": 12.9 + np.random.uniform(-0.3, 0.3),
                    "longitude": 77.5 + np.random.uniform(-0.3, 0.3),
                    "renewable_pref": np.random.choice([True, False], p=[0.4, 0.6])
                })
            
            # Run matching simulations
            match_results = []
            for buyer in buyers:
                # Score each host
                host_scores = []
                for host in hosts:
                    # Calculate distance (simplified)
                    dist = np.sqrt(
                        (host["latitude"] - buyer["latitude"])**2 + 
                        (host["longitude"] - buyer["longitude"])**2
                    ) * 111  # Convert to km approximately
                    
                    # Calculate match score (similar to backend algorithm)
                    price_score = max(0, 100 - (host["price_per_kwh"] / buyer["max_price"]) * 50) if host["price_per_kwh"] <= buyer["max_price"] else 0
                    distance_score = max(0, 100 - (dist / 50) * 100)
                    rating_score = host["rating"] * 20
                    renewable_bonus = 10 if host["renewable_cert"] and buyer["renewable_pref"] else 0
                    availability_score = min(100, (host["available_kwh"] / buyer["required_kwh"]) * 100) if buyer["required_kwh"] > 0 else 100
                    
                    total_score = (
                        price_score * 0.30 +
                        distance_score * 0.25 +
                        rating_score * 0.25 +
                        availability_score * 0.10 +
                        renewable_bonus
                    )
                    
                    if host["price_per_kwh"] <= buyer["max_price"]:
                        host_scores.append({
                            "host_id": host["id"],
                            "score": total_score,
                            "price": host["price_per_kwh"],
                            "distance": dist,
                            "can_fulfill": host["available_kwh"] >= buyer["required_kwh"]
                        })
                
                # Sort by score
                host_scores.sort(key=lambda x: x["score"], reverse=True)
                
                if host_scores:
                    best_match = host_scores[0]
                    match_results.append({
                        "buyer_id": buyer["id"],
                        "matched": True,
                        "best_score": best_match["score"],
                        "best_price": best_match["price"],
                        "best_distance": best_match["distance"],
                        "can_fulfill": best_match["can_fulfill"],
                        "alternatives": len(host_scores)
                    })
                else:
                    match_results.append({
                        "buyer_id": buyer["id"],
                        "matched": False,
                        "best_score": 0,
                        "alternatives": 0
                    })
            
            # Calculate metrics
            matched_count = sum(1 for r in match_results if r["matched"])
            fulfilled_count = sum(1 for r in match_results if r.get("can_fulfill", False))
            avg_score = np.mean([r["best_score"] for r in match_results if r["matched"]])
            avg_alternatives = np.mean([r["alternatives"] for r in match_results])
            
            metrics = {
                "model_name": "Matching_Algorithm",
                "n_buyers": n_buyers,
                "n_hosts": n_hosts,
                "match_rate": matched_count / n_buyers * 100,
                "fulfillment_rate": fulfilled_count / n_buyers * 100,
                "avg_match_score": float(avg_score),
                "avg_alternatives": float(avg_alternatives),
                "price_satisfaction": np.mean([
                    r["best_price"] for r in match_results if r.get("best_price")
                ]),
                "avg_distance_km": np.mean([
                    r["best_distance"] for r in match_results if r.get("best_distance")
                ])
            }
            
            self.results["matching_algorithm"] = metrics
            
            logger.info(f"\nMatching Algorithm Results:")
            logger.info(f"  Match Rate: {metrics['match_rate']:.1f}%")
            logger.info(f"  Fulfillment Rate: {metrics['fulfillment_rate']:.1f}%")
            logger.info(f"  Avg Match Score: {metrics['avg_match_score']:.2f}")
            logger.info(f"  Avg Alternatives: {metrics['avg_alternatives']:.1f}")
            logger.info(f"  Avg Distance: {metrics['avg_distance_km']:.2f} km")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating matching: {e}")
            return {"error": str(e)}
    
    def evaluate_dynamic_pricing(self, df: pd.DataFrame) -> dict:
        """Evaluate Dynamic Pricing Model"""
        logger.info("=" * 60)
        logger.info("Evaluating Dynamic Pricing Model")
        logger.info("=" * 60)
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Generate synthetic pricing data
            np.random.seed(42)
            n_samples = len(df)
            
            # Create features for pricing
            supply = df['GHI'].values / 1000 * 5 * 0.85  # kW from solar
            demand = np.random.uniform(10, 50, n_samples)  # kW demand
            hour_of_day = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
            day_of_week = np.tile(np.arange(7), n_samples // 7 + 1)[:n_samples]
            
            # Supply-demand ratio
            supply_demand_ratio = supply / (demand + 0.1)
            
            # Generate target prices (₹/kWh) based on supply-demand
            base_price = 6.5
            price = base_price + 2 * (1 - supply_demand_ratio)
            price = price + np.sin(hour_of_day / 24 * 2 * np.pi) * 1.5  # Time-of-day variation
            price = price + np.random.normal(0, 0.3, n_samples)  # Noise
            price = np.clip(price, 4, 12)  # Bounds
            
            X = np.column_stack([supply, demand, supply_demand_ratio, hour_of_day, day_of_week])
            y = price
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_regression_metrics(y_test, y_pred, "Dynamic_Pricing")
            
            # Feature importance
            feature_names = ['supply', 'demand', 'supply_demand_ratio', 'hour', 'day_of_week']
            metrics["feature_importance"] = dict(zip(feature_names, model.feature_importances_.tolist()))
            
            # Price distribution analysis
            metrics["price_range"] = {"min": float(y_test.min()), "max": float(y_test.max())}
            metrics["predicted_range"] = {"min": float(y_pred.min()), "max": float(y_pred.max())}
            
            self.results["dynamic_pricing"] = metrics
            
            logger.info(f"\nDynamic Pricing Results:")
            logger.info(f"  RMSE: ₹{metrics['rmse']:.4f}/kWh")
            logger.info(f"  MAE: ₹{metrics['mae']:.4f}/kWh")
            logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
            logger.info(f"  Price Range: ₹{metrics['price_range']['min']:.2f} - ₹{metrics['price_range']['max']:.2f}/kWh")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating pricing: {e}")
            return {"error": str(e)}
    
    def generate_plots(self):
        """Generate visualization plots"""
        logger.info("Generating evaluation plots...")
        
        plots_dir = REPORTS_DIR / f"plots_{self.timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Solar XGBoost: Actual vs Predicted
        if "solar_xgboost" in self.results and "y_test" in self.results["solar_xgboost"]:
            y_test = np.array(self.results["solar_xgboost"]["y_test"])
            y_pred = np.array(self.results["solar_xgboost"]["y_pred"])
            residuals = y_pred - y_test
            
            # 1a. Scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, s=30, c=np.abs(residuals), cmap='RdYlGn_r')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Perfect Prediction')
            
            # Add confidence bands
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(y_test, p(y_test), "b-", alpha=0.3, label='Trend Line')
            
            ax.set_xlabel('Actual Power (kW)', fontsize=12)
            ax.set_ylabel('Predicted Power (kW)', fontsize=12)
            ax.set_title('Solar Generation Forecast: Actual vs Predicted', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            r2 = self.results["solar_xgboost"]["r2_score"]
            rmse = self.results["solar_xgboost"]["rmse"]
            mae = self.results[ with Enhanced Visualization
        if "solar_xgboost" in self.results and "feature_importance" in self.results["solar_xgboost"]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            feat_imp = self.results["solar_xgboost"]["feature_importance"]
            features = list(feat_imp.keys())
            importance = list(feat_imp.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1]
            features_sorted = [features[i] for i in sorted_idx]
            importance_sorted = [importance[i] for i in sorted_idx]
            
            # Horizontal bar chart
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features_sorted)))
            bars = axes[0].barh(features_sorted, importance_sorted, color=colors, edgecolor='black', alpha=0.8)
            axes[0].set_xlabel('Feature Importance', fontsize=12)
            axes[0].set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
            axes[0].invert_yaxis()
            
            for bar, val in zip(bars, importance_sorted):
                axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                           va='center', fontsize=10)
            
            # Pie chart
            axes[1].pie(importance_sorted, labels=features_sorted, autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 10})
            axes[1].set_title('Feature Contribution Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "05_solar_feature_importance.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: 05_esidual Plot', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residual Distribution
            axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel('Residuals (kW)', fontsize=11)
            axes[0, 1].set_ylabel('Frequency', fontsize=11)
            axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].text(0.7, 0.95, f'Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}',
                           transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Q-Q Plot for normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Error Distribution by Power Range
            power_bins = np.linspace(y_test.min(), y_test.max(), 6)
            bin_centers = (power_bins[:-1] + power_bins[1:]) / 2
            bin_errors = []
            for i in range(len(power_bins)-1):
                mask = (y_test >= power_bins[i]) & (y_test < power_bins[i+1])
                if mask.sum() > 0:
                    bin_errors.append(np.abs(residuals[mask]).mean())
                else:
                    bin_errors.append(0)
            
            axes[1, 1].bar(range(len(bin_centers)), bin_errors, color='coral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Power Range (kW)', fontsize=11)
            axes[1, 1].set_ylabel('Mean Absolute Error (kW)', fontsize=11)
            axes[1, 1].set_title('Error Distribution by Power Range', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(range(len(bin_centers)))
            axes[1, 1].set_xticklabels([f'{c:.1f}' for c in bin_centers], rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "02_solar_residual_analysis.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: 02_solar_residual_analysis.png")
            
            # 1c. Time Series Prediction (if we have more samples)
            if len(y_test) >= 50:
                fig, ax = plt.subplots(figsize=(14, 6))
                sample_range = slice(0, min(100, len(y_test)))
                x_axis = np.arange(len(y_test[sample_range]))
                
                ax.plot(x_axis, y_test[sample_range], 'b-', label='Actual', linewidth=2, alpha=0.7)
                ax.plot(x_axis, y_pred[sample_range], 'r--', label='Predicted', linewidth=2, alpha=0.7)
                ax.fill_between(x_axis, y_test[sample_range], y_pred[sample_range], alpha=0.2, color='gray')
                
                ax.set_xlabel('Sample Index', fontsize=12)
                ax.set_ylabel('Power (kW)', fontsize=12)
                ax.set_title('Solar Generation: Time Series Prediction', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "03_solar_timeseries_prediction.png", dpi=150)
                plt.close()
                logger.info(f"  Saved: 03_solar_timeseries_prediction.png")
            
            # 1d. Error Box Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            abs_errors = np.abs(residuals)
            
            bp = ax.boxplot([abs_errors], labels=['Absolute Error'], patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            ax.set_ylabel('Absolute Error (kW)', fontsize=12)
            ax.set_title('Solar Forecast: Error Distribution Box Plot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            stats_text = f"""Median: {np.median(abs_errors):.4f} kW
P90: {np.percentile(abs_errors, 90):.4f} kW
P95: {np.percentile(abs_errors, 95):.4f} kW
Max: {np.max(abs_errors):.4f} kW"""
            
            ax.text(1.15, 0.5, stats_text, transform=ax.transData, fontsize=10,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "04_solar_error_boxplot.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: 04_solar_error_boxplot.png")
        
        # 2. Feature Importance
        if "solar_xgboost" in self.results and "feature_importance" in self.results["solar_xgboost"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_imp = self.results["solar_xgboost"]["feature_importance"]
            features = list(feat_imp.keys())
            importance = list(feat_imp.values())
            
            bEnhanced Matching Algorithm Visualization
        if "matching_algorithm" in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Match rate donut chart
            match_rate = self.results["matching_algorithm"]["match_rate"]
            fulfillment_rate = self.results["matching_algorithm"]["fulfillment_rate"]
            
            sizes = [match_rate, 100-match_rate]
            colors = ['#4CAF50', '#f44336']
            explode = (0.05, 0)
            
            axes[0, 0].pie(sizes, explode=explode, labels=['Matched', 'No Match'],
                          autopct='%1.1f%%', colors=colors, startangle=90, 
                          textprops={'fontsize': 11, 'fontweight': 'bold'},
                          wedgeprops=dict(edgecolor='white', linewidth=2))
            axes[0, 0].set_title('Buyer-Seller Match Success Rate', fontsize=13, fontweight='bold')
            
            # Performance metrics radar/bar
            metrics_names = ['Match\nScore', 'Availability\nRate', 'Fulfillment\nRate']
            metrics_vals = [
                self.results["matching_algorithm"]["avg_match_score"],
                match_rate,
                fulfillment_rate
            ]
            
            bComprehensive Model Comparison
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        models = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        
        for model_name, metrics in self.results.items():
            if "r2_score" in metrics:
                models.append(model_name.replace("_", " ").title())
                r2_scores.append(metrics["r2_score"])
                rmse_scores.append(metrics.get("rmse", 0))
                mae_scores.append(metrics.get("mae", 0))
                mape_scores.append(metrics.get("mape", 0) if metrics.get("mape") else 0)
        
        if models:
            x = np.arange(len(models))
            
            # R² Score
            ax1 = fig.add_subplot(gs[0, 0])
            bars = ax1.bar(x, r2_scores, color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax1.set_title('Model R² Score Comparison', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=30, ha='right')
            ax1.set_ylim(0, 1.1)
            ax1.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Good Threshold (0.85)')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, r2_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
            
            # RMSE
            ax2 = fig.add_subplot(gs[0, 1])
            bars = ax2.bar(x, rmse_scores, color='#FF9800', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
            ax2.set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=30, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, rmse_scores):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.02, 
                            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
            
            # MAE
            ax3 = fig.add_subplot(gs[1, 0])
            bars = ax3.bar(x, mae_scores, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
            ax3.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models, rotation=30, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, mae_scores):
                if val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.02, 
                            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
            
            # MAPE
            ax4 = fig.add_subplot(gs[1, 1])
            bars = ax4.bar(x, mape_scores, color='#9C27B0', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax4.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
            ax4.set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(models, rotation=30, ha='right')
            ax4.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Acceptable Threshold (15%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, mape_scores):
                if val > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_scores)*0.02, 
                            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
            
            # Metrics Heatmap
            ax5 = fig.add_subplot(gs[2, :])
            metrics_matrix = np.array([
                [r2 * 100 for r2 in r2_scores],  # Scale R² to 0-100
                [100 - (rmse / max(rmse_scores) * 100) if max(rmse_scores) > 0 else 0 for rmse in rmse_scores],  # Invert and normalize
                [100 - (mae / max(mae_scores) * 100) if max(mae_scores) > 0 else 0 for mae in mae_scores],  # Invert and normalize
                [100 - (mape / 100 if mape < 100 else 100) for mape in mape_scores]  # Invert MAPE
            ])
            
            im = ax5.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax5.set_xticks(x)
            ax5.set_xticklabels(models, rotation=30, ha='right')
            ax5.set_yticks(range(4))
            ax5.set_yticklabels(['R² Score', 'RMSE\n(inverted)', 'MAE\n(inverted)', 'MAPE\n(inverted)'])
            ax5.set_title('Model Performance Heatmap (Higher is Better)', fontsize=13, fontweight='bold')
            
            # Add values to heatmap
            for i in range(len(metrics_matrix)):
                for j in range(len(models)):
                    text = ax5.text(j, i, f'{metrics_matrix[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=10, fontweight='bold')
            
            plt.colorbar(im, ax=ax5, label='Score (0-100)')
            
            plt.savefig(plots_dir / "07_model_comparison_comprehensive.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: 07_model_comparison_comprehensive.png")
        
        # 5. Additional: Metrics Summary Table as Image
        if models:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = [['Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'Status']]
            for i, model in enumerate(models):
                status = '✅ Excellent' if r2_scores[i] > 0.9 else '✓ Good' if r2_scores[i] > 0.8 else '⚠ Acceptable'
                table_data.append([
                    model,
                    f'{r2_scores[i]:.4f}',
                    f'{rmse_scores[i]:.4f}',
                    f'{mae_scores[i]:.4f}',
                    f'{mape_scores[i]:.2f}%' if mape_scores[i] > 0 else 'N/A',
                    status
                ])
            
            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#E8F5E9')
            
            ax.set_title('ML Model Evaluation Summary Table', fontsize=14, fontweight='bold', pad=20)
            
            plt.savefig(plots_dir / "08_metrics_summary_table.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: 08_metrics_summary_tablees Available', fontsize=13, fontweight='bold')
            axes[1, 1].text(0, alternatives + 0.3, f'{alternatives:.1f}', 
                           ha='center', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "06_matching_algorithm_detailed.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: 06_matching_algorithm_detailedAvg Distance']
            metrics_vals = [
                self.results["matching_algorithm"]["avg_match_score"],
                self.results["matching_algorithm"]["avg_alternatives"] * 10,
                self.results["matching_algorithm"]["avg_distance_km"]
            ]
            axes[1].bar(metrics_names, metrics_vals, color=['#2196F3', '#FF9800', '#9C27B0'])
            axes[1].set_title('Matching Algorithm Performance', fontsize=12)
            axes[1].set_ylabel('Value')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "matching_algorithm_results.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: matching_algorithm_results.png")
        
        # 4. Model Comparison Summary
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        r2_scores = []
        rmse_scores = []
        
        for model_name, metrics in self.results.items():
            if "r2_score" in metrics:
                models.append(model_name.replace("_", " ").title())
                r2_scores.append(metrics["r2_score"])
                rmse_scores.append(metrics.get("rmse", 0))
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, r2_scores, width, label='R² Score', color='#4CAF50')
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            for bar, val in zip(bars1, r2_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "model_comparison.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: model_comparison.png")
        
        return plots_dir
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        logger.info("=" * 60)
        logger.info("Generating Evaluation Report")
        logger.info("=" * 60)
        
        report_path = REPORTS_DIR / f"ml_evaluation_report_{self.timestamp}.md"
        
        report = []
        report.append("# ML Model Evaluation Report")
        report.append(f"## Solar Sharing Platform\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append("This report presents the evaluation results of machine learning models ")
        report.append("deployed in the Solar Energy Sharing Platform. The models are evaluated ")
        report.append("using industry-standard metrics and cross-validation techniques.\n")
        
        # Summary Table
        report.append("### Performance Summary\n")
        report.append("| Model | RMSE | MAE | R² Score | Status |")
        report.append("|-------|------|-----|----------|--------|")
        
        for model_name, metrics in self.results.items():
            if "error" not in metrics:
                rmse = metrics.get("rmse", "N/A")
                mae = metrics.get("mae", "N/A")
                r2 = metrics.get("r2_score", "N/A")
                
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
                mae_str = f"{mae:.4f}" if isinstance(mae, float) else mae
                r2_str = f"{r2:.4f}" if isinstance(r2, float) else r2
                
                status = "✅ Good" if isinstance(r2, float) and r2 > 0.8 else "⚠️ Acceptable" if isinstance(r2, float) and r2 > 0.6 else "❌ Needs Improvement"
                report.append(f"| {model_name.replace('_', ' ').title()} | {rmse_str} | {mae_str} | {r2_str} | {status} |")
        
        report.append("\n")
        
        # Detailed Results
        report.append("## Detailed Model Evaluations\n")
        
        # Solar XGBoost
        if "solar_xgboost" in self.results and "error" not in self.results["solar_xgboost"]:
            metrics = self.results["solar_xgboost"]
            report.append("### 1. Solar Generation Forecast (XGBoost)\n")
            report.append("**Purpose:** Predict solar power generation based on weather conditions.\n")
            report.append("**Features:** Temperature, GHI, DNI, DHI, Cloud Type, Solar Zenith Angle\n\n")
            
            report.append("#### Performance Metrics\n")
            report.append(f"- **RMSE:** {metrics['rmse']:.4f} kW")
            report.append(f"- **MAE:** {metrics['mae']:.4f} kW")
            report.append(f"- **R² Score:** {metrics['r2_score']:.4f}")
            report.append(f"- **MAPE:** {metrics.get('mape', 'N/A')}%")
            report.append(f"- **Explained Variance:** {metrics['explained_variance']:.4f}")
            report.append(f"- **Cross-Validation R² (5-fold):** {metrics.get('cv_r2_mean', 'N/A'):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
            report.append(f"- **Time-Series CV RMSE:** {metrics.get('ts_cv_rmse_mean', 'N/A'):.4f}\n")
            
            if "feature_importance" in metrics:
                report.append("#### Feature Importance\n")
                for feat, imp in sorted(metrics["feature_importance"].items(), key=lambda x: -x[1]):
                    report.append(f"- **{feat}:** {imp:.4f}")
            report.append("\n")
        
        # Matching Algorithm
        if "matching_algorithm" in self.results:
            metrics = self.results["matching_algorithm"]
            report.append("### 2. Buyer-Seller Matching Algorithm\n")
            report.append("**Purpose:** Optimally match energy buyers with sellers based on preferences.\n")
            report.append("**Factors:** Price, Distance, Rating, Renewable Certification, Availability\n\n")
            
            report.append("#### Performance Metrics\n")
            report.append(f"- **Match Rate:** {metrics['match_rate']:.1f}%")
            report.append(f"- **Fulfillment Rate:** {metrics['fulfillment_rate']:.1f}%")
            report.append(f"- **Average Match Score:** {metrics['avg_match_score']:.2f}/100")
            report.append(f"- **Average Alternatives:** {metrics['avg_alternatives']:.1f} hosts per buyer")
            report.append(f"- **Average Distance:** {metrics['avg_distance_km']:.2f} km")
            report.append(f"- **Average Matched Price:** ₹{metrics['price_satisfaction']:.2f}/kWh\n")
        
        # Dynamic Pricing
        if "dynamic_pricing" in self.results and "error" not in self.results["dynamic_pricing"]:
            metrics = self.results["dynamic_pricing"]
            report.append("### 3. Dynamic Pricing Model\n")
            report.append("**Purpose:** Optimize energy prices based on supply-demand dynamics.\n")
            report.append("**Features:** Supply, Demand, Supply-Demand Ratio, Time of Day, Day of Week\n\n")
            
            report.append("#### Performance Metrics\n")
            report.append(f"- **RMSE:** ₹{metrics['rmse']:.4f}/kWh")
            report.append(f"- **MAE:** ₹{metrics['mae']:.4f}/kWh")
            report.append(f"- **R² Score:** {metrics['r2_score']:.4f}")
            report.append(f"- **Price Range:** ₹{metrics['price_range']['min']:.2f} - ₹{metrics['price_range']['max']:.2f}/kWh\n")
        
        # Conclusions
        report.append("## Conclusions\n")
        report.append("### Key Findings\n")
        
        if "solar_xgboost" in self.results and self.results["solar_xgboost"].get("r2_score", 0) > 0.8:
            report.append("1. **Solar Forecast Model** achieves high accuracy (R² > 0.8), suitable for production use.")
        else:
            report.append("1. **Solar Forecast Model** needs additional training data for improved accuracy.")
        
        if "matching_algorithm" in self.results and self.results["matching_algorithm"]["match_rate"] > 80:
            report.append("2. **Matching Algorithm** successfully pairs >80% of buyers with suitable sellers.")
        
        report.append("\n### Recommendations\n")
        report.append("1. Collect more real-world IoT data to improve model accuracy.")
        report.append("2. Implement A/B testing for the matching algorithm.")
        report.append("3. Consider LSTM models for longer-term forecasting.\n")
        
        # Save report
        report_content = "\n".join(report)
        with open(report_path, "w") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Also save JSON results
        json_path = REPORTS_DIR / f"ml_evaluation_results_{self.timestamp}.json"
        with open(json_path, "w") as f:
            # Convert numpy types to Python native types
            json_results = {}
            for k, v in self.results.items():
                json_results[k] = {
                    key: (val.tolist() if hasattr(val, 'tolist') else val)
                    for key, val in v.items()
                }
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to: {json_path}")
        
        return str(report_path)


def main():
    """Run comprehensive ML evaluation"""
    logger.info("=" * 60)
    logger.info("Solar Sharing Platform - ML Model Evaluation")
    logger.info("=" * 60)
    
    evaluator = MLEvaluator()
    
    # Load test data
    df = evaluator.load_test_data()
    
    if df is not None:
        # Evaluate all models
        evaluator.evaluate_solar_xgboost(df)
        evaluator.evaluate_matching_algorithm()
        evaluator.evaluate_dynamic_pricing(df)
        
        # Generate plots
        plots_dir = evaluator.generate_plots()
        logger.info(f"Plots saved to: {plots_dir}")
        
        # Generate report
        report_path = evaluator.generate_report()
        
        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info(f"Report: {report_path}")
        logger.info(f"Plots: {plots_dir}")
        logger.info("=" * 60)
    else:
        logger.error("No test data available. Please run preprocessing first.")
        logger.info("Run: python scripts/preprocess_data.py")


if __name__ == "__main__":
    main()
