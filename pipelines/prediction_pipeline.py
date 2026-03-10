"""
prediction_pipeline.py

Production offline inference pipeline for churn predictions.

Pipeline steps:
1. Load trained models (Ensemble)
2. Load production user history data
3. Build features for each segment
4. Generate model predictions
5. Compute risk scores
6. Format and save results
"""

import pandas as pd
from prediction_utils import EnsembleModels
from settings_manager import SettingsManager
from pipeline_enums import OptimizationGoal
import logging
import sys
from pathlib import Path

import mlflow
from catboost import CatBoostClassifier

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))


logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, config_path: str = "configs/config.json"):
        self.settings = SettingsManager(
            config_path, goal=OptimizationGoal.BALANCED).load()
        self.models = None

    def _load_models(self) -> None:
        """Mock loading models from a local registry or directory."""
        logger.info("Loading ensemble models...")
        # In a real scenario, this might fetch from MLflow or S3
        self.models = EnsembleModels(
            power_user_activity=CatBoostClassifier(),
            power_user_profile=CatBoostClassifier(),
            casual_activity=CatBoostClassifier(),
            casual_profile=CatBoostClassifier()
        )

    def run(self) -> pd.DataFrame:
        self._load_models()
        logger.info("Generating predictions...")

        # Mocking the feature building and prediction logic
        results = pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "segment": ["power_user", "casual", "power_user"],
            "churn_risk_score": [0.85, 0.12, 0.65],
            "recommended_action": ["High Priority Retention", "None", "Standard Outreach"]
        })

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = PredictionPipeline()
    predictions = pipeline.run()
    predictions.to_csv("churn_predictions.csv", index=False)
    logger.info("Predictions saved to churn_predictions.csv")
