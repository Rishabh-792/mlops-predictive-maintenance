"""
training_pipeline.py

Model training pipeline for the four-model ensemble.

This module trains the production ensemble of binary classifiers:
1. Power User Segment - Activity Model
2. Power User Segment - Profile Model
3. Casual Segment - Activity Model
4. Casual Segment - Profile Model

All models are CatBoost classifiers with segment-specific hyperparameters.
Results are logged to MLflow.
Configuration is fully externalized.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure utils can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from pipeline_enums import UserSegment, OptimizationGoal
from settings_manager import SettingsManager
from model_training_utils import train_segment_model
import pandas as pd # Mocking data load

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path: str = "configs/settings.json"):
        self.settings_manager = SettingsManager(config_path, goal=OptimizationGoal.BALANCED)
        self.settings = self.settings_manager.load()
        
        # Set up MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.settings.project_name)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mock data loading function."""
        # Replace with actual data loading logic
        df = pd.DataFrame({"target": [0, 1, 0, 1] * 25})
        features = pd.DataFrame({"feat1": range(100), "feat2": ["A", "B"] * 50})
        return df, features

    def run(self) -> Dict[str, str]:
        """Executes the pipeline and returns a dictionary of MLflow model URIs."""
        logger.info("Starting ensemble training pipeline...")
        
        raw_df, feature_df = self.load_data()
        model_uris = {}
        
        segments = ["power_user", "casual"]
        model_types = ["activity", "profile"]
        
        with mlflow.start_run(run_name="Ensemble_Master_Run"):
            for segment in segments:
                for m_type in model_types:
                    identifier = f"{segment}_{m_type}"
                    logger.info(f"--- Training {identifier.upper()} ---")
                    
                    try:
                        _, uri = train_segment_model(
                            raw_df=raw_df,
                            feature_df=feature_df,
                            model_type=m_type,
                            segment=segment,
                            target_col=self.settings.schema.target_variable
                        )
                        model_uris[identifier] = uri
                    except Exception as e:
                        logger.error(f"Training failed for {identifier}: {str(e)}")
                        
        logger.info("Pipeline execution complete.")
        return model_uris

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TrainingPipeline()
    uris = pipeline.run()
    print("Logged Models:", uris)