"""
feature_pipeline.py

Data loading, validation, and feature engineering orchestration.
Splits data into segments and applies specific feature aggregations.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.settings_manager import SettingsManager
from utils.core_utils import get_logger, ensure_dir
from utils.feature_builders import split_by_segment, build_power_user_features, build_guest_features

class FeaturePipeline:
    def __init__(self):
        self.settings = SettingsManager().load()
        self.logger = get_logger("feature_pipeline", log_dir="logs")
        self.output_dir = ensure_dir("artifacts/features")

    def run(self, clean_data_path: str) -> Dict[str, str]:
        """Runs the feature engineering pipeline."""
        self.logger.info(f"Loading cleaned data from {clean_data_path}")
        df = pd.read_parquet(clean_data_path)

        # 1. Define thresholds based on config
        thresholds = {
            'power_user': self.settings.segments['power_user'].min_activity_threshold,
            'guest': self.settings.segments['guest'].max_activity_threshold or 2
        }

        # 2. Split data
        self.logger.info("Splitting users into operational segments...")
        power_df, casual_df, guest_df = split_by_segment(df, thresholds)

        # 3. Build Features
        power_features = build_power_user_features(power_df)
        guest_features = build_guest_features(guest_df)
        # Note: casual_features omitted for brevity, logic remains the same

        # 4. Save Artifacts
        paths = {}
        if not power_features.empty:
            p_path = Path(self.output_dir) / "power_user_features.csv"
            power_features.to_csv(p_path, index=False)
            paths["power_user"] = str(p_path)
            
        if not guest_features.empty:
            g_path = Path(self.output_dir) / "guest_features.csv"
            guest_features.to_csv(g_path, index=False)
            paths["guest"] = str(g_path)

        self.logger.info("Feature engineering complete.")
        return paths