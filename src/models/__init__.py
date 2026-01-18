# Models module initialization
# Note: LSTM models are deferred imports due to optional TensorFlow dependency

try:
    from src.models.solar_forecast import (
        SolarLSTMModel,
        SolarXGBoostModel,
        SolarForecastingEnsemble
    )
except ImportError:
    # TensorFlow not available
    SolarLSTMModel = None
    SolarXGBoostModel = None
    SolarForecastingEnsemble = None

try:
    from src.models.demand_forecast import (
        DemandLSTMModel,
        DemandXGBoostModel,
        DemandForecastingEnsemble,
        RandomForestDemandModel
    )
except ImportError:
    # TensorFlow not available
    DemandLSTMModel = None
    DemandXGBoostModel = None
    DemandForecastingEnsemble = None
    RandomForestDemandModel = None

from src.models.advanced_models import (
    DynamicPricingModel,
    InvestorRiskScoringModel,
    AnomalyDetectionModel,
    EquipmentFailurePredictorModel
)

__all__ = [
    "SolarLSTMModel",
    "SolarXGBoostModel",
    "SolarForecastingEnsemble",
    "DemandLSTMModel",
    "DemandXGBoostModel",
    "DemandForecastingEnsemble",
    "RandomForestDemandModel",
    "DynamicPricingModel",
    "InvestorRiskScoringModel",
    "AnomalyDetectionModel",
    "EquipmentFailurePredictorModel"
]
