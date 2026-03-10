"""
prediction_utils.py

Unified prediction engine for customer churn prediction.
Combines multiple prediction strategies into a single module.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

# =========================================================================
# MODEL CONTAINER
# =========================================================================

@dataclass
class EnsembleModels:
    """
    Holds the 4 binary CatBoost models used in production.
    """
    power_user_activity: CatBoostClassifier
    power_user_profile: CatBoostClassifier
    casual_activity: CatBoostClassifier
    casual_profile: CatBoostClassifier

    @classmethod
    def load_from_directory(cls, base_dir: str) -> 'EnsembleModels':
        """Helper to instantiate from a local directory structure."""
        import os
        
        models = {}
        expected_models = [
            "power_user_activity", "power_user_profile", 
            "casual_activity", "casual_profile"
        ]
        
        for name in expected_models:
            model_path = os.path.join(base_dir, name, "model.cb")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model artifact: {model_path}")
                
            cb_model = CatBoostClassifier()
            cb_model.load_model(model_path)
            models[name] = cb_model
            
        return cls(**models)