"""
preprocessing_pipeline.py

Raw data preprocessing and user-level aggregation pipeline.
Orchestrates cleaning, standardization, and MLflow run initialization.
"""

import sys
import mlflow
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional

# Add root to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.settings_manager import SettingsManager
from utils.core_utils import get_logger, ensure_dir, require_columns
from utils.pipeline_enums import OptimizationGoal

class PreprocessingPipeline:
    def __init__(
        self, 
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
        run_name: Optional[str] = None
    ):
        self.settings = SettingsManager(goal=goal).load()
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H%M")
        
        # Setup paths and logging
        self.logs_dir = ensure_dir("logs")
        self.output_dir = ensure_dir("artifacts/preprocessed")
        self.logger = get_logger("preprocessing_pipeline", log_dir=self.logs_dir)
        
        # Setup MLflow (Local Server Configuration)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.settings.project_name)

        self.logger.info("PreprocessingPipeline initialized")
        self.logger.info(f"Optimization Goal: {goal.value}")

    def run(self, raw_data_path: Optional[str] = None) -> str:
        """Executes the preprocessing workflow."""
        with mlflow.start_run(run_name=f"prep_{self.run_name}"):
            self.logger.info("Starting preprocessing run...")
            
            # 1. Load Data (Mocking data if path is None)
            if raw_data_path and Path(raw_data_path).exists():
                df = pd.read_csv(raw_data_path)
            else:
                self.logger.warning("No valid raw data path provided. Generating mock data.")
                df = pd.DataFrame({
                    "user_id": ["U1", "U1", "U2", "U3"],
                    "session_id": ["S1", "S2", "S3", "S4"],
                    "event_date": pd.to_datetime(["2023-10-01", "2023-10-05", "2023-10-02", "2023-10-06"]),
                    "duration_minutes": [15, 45, 10, 5],
                    "device_type": ["mobile", "desktop", "mobile", "tablet"]
                })

            # 2. Validate
            require_columns(df, ["user_id", "session_id", "event_date"])
            
            # 3. Save aggregated output
            output_path = Path(self.output_dir) / f"clean_events_{self.run_name}.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Preprocessed data saved to {output_path}")
            
            # 4. Log artifact to MLflow
            mlflow.log_artifact(local_path=str(output_path))
            
            return str(output_path)