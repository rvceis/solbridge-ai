"""
Model evaluation and comparison utilities.

Computes MAE/MAPE/RMSE, compares model variants, and generates reports.
"""
from pathlib import Path
from typing import Dict, Tuple, Any, List
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate and compare trained models"""
    
    @staticmethod
    def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Compute MAE, RMSE, MAPE for regression models"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Also compute R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2)
        }
    
    @staticmethod
    def compute_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Compute accuracy, precision, recall, F1 for classification"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, y_pred_proba)
                metrics['auc_roc'] = float(auc)
            except:
                pass
        
        return metrics
    
    @staticmethod
    def compare_models(
        model_results: Dict[str, Dict[str, Any]],
        metric_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            model_results: Dict mapping model_name → {metrics, predictions, etc}
            metric_names: List of metrics to compare (defaults to all)
        
        Returns:
            DataFrame with comparison
        """
        if not model_results:
            return pd.DataFrame()
        
        comparison_data = {}
        
        for model_name, result in model_results.items():
            if 'metrics' not in result:
                continue
            
            metrics = result['metrics']
            if metric_names:
                metrics = {k: v for k, v in metrics.items() if k in metric_names}
            
            comparison_data[model_name] = metrics
        
        df = pd.DataFrame(comparison_data).T
        return df.sort_values('RMSE' if 'RMSE' in df.columns else df.columns[0], ascending=True)
    
    @staticmethod
    def generate_report(
        model_results: Dict[str, Dict[str, Any]],
        output_path: str = None
    ) -> str:
        """
        Generate evaluation report.
        
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            f"MODEL EVALUATION REPORT - {datetime.now().isoformat()}",
            "=" * 80,
            ""
        ]
        
        comparison = ModelEvaluator.compare_models(model_results)
        
        if not comparison.empty:
            report_lines.append("MODEL COMPARISON:")
            report_lines.append(comparison.to_string())
            report_lines.append("")
        
        # Detailed metrics per model
        for model_name, result in model_results.items():
            if 'metrics' not in result:
                continue
            
            report_lines.append(f"\n{model_name}:")
            report_lines.append("-" * 40)
            
            metrics = result['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric_name}: {value:.6f}")
                else:
                    report_lines.append(f"  {metric_name}: {value}")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {path}")
        
        return report_text


class ModelSelector:
    """Select best model based on metrics"""
    
    @staticmethod
    def select_best_regression_model(
        model_results: Dict[str, Dict[str, Any]],
        metric: str = 'RMSE'
    ) -> Tuple[str, Dict]:
        """
        Select best regression model based on metric (default: RMSE).
        
        Returns:
            (model_name, result)
        """
        valid_results = {
            name: result for name, result in model_results.items()
            if 'metrics' in result and metric in result['metrics']
        }
        
        if not valid_results:
            logger.warning(f"No valid results for metric {metric}")
            return None, {}
        
        best_model = min(valid_results.items(), key=lambda x: x[1]['metrics'][metric])
        return best_model[0], best_model[1]
    
    @staticmethod
    def select_best_classification_model(
        model_results: Dict[str, Dict[str, Any]],
        metric: str = 'f1'
    ) -> Tuple[str, Dict]:
        """Select best classification model (higher is better)"""
        valid_results = {
            name: result for name, result in model_results.items()
            if 'metrics' in result and metric in result['metrics']
        }
        
        if not valid_results:
            logger.warning(f"No valid results for metric {metric}")
            return None, {}
        
        best_model = max(valid_results.items(), key=lambda x: x[1]['metrics'][metric])
        return best_model[0], best_model[1]


class MLflowHelper:
    """Helper utilities for MLflow integration"""
    
    @staticmethod
    def register_model_to_registry(
        run_id: str,
        model_uri: str,
        model_name: str,
        stage: str = "Staging"
    ):
        """
        Register model to MLflow registry.
        
        Args:
            run_id: MLflow run ID
            model_uri: Model URI (e.g., runs:/run_id/model)
            model_name: Model name in registry
            stage: Stage (Staging, Production, Archived)
        """
        try:
            import mlflow
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {result.name} v{result.version}")
            
            # Transition to stage
            if stage:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage=stage
                )
                logger.info(f"Transitioned {model_name} to {stage}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    @staticmethod
    def get_best_model_version(model_name: str):
        """Retrieve best model version from registry"""
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(model_name, stages=["Production"])
            
            if latest:
                return latest[0]
            
            logger.warning(f"No Production version for {model_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    sample_results = {
        'LSTM': {'metrics': {'MAE': 50, 'RMSE': 75, 'MAPE': 0.15}},
        'XGBoost': {'metrics': {'MAE': 45, 'RMSE': 70, 'MAPE': 0.12}},
        'RandomForest': {'metrics': {'MAE': 55, 'RMSE': 85, 'MAPE': 0.18}}
    }
    
    evaluator = ModelEvaluator()
    
    # Compare
    comparison = evaluator.compare_models(sample_results)
    print(comparison)
    
    # Generate report
    report = evaluator.generate_report(sample_results, "evaluation_report.txt")
    print(report)
    
    # Select best
    selector = ModelSelector()
    best_model, best_result = selector.select_best_regression_model(sample_results)
    print(f"\nBest model: {best_model}")
    print(f"Metrics: {best_result['metrics']}")
