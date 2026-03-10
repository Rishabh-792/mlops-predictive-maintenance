"""
inference.py

SageMaker entry point for the custom CatBoost ensemble container.
Handles model loading and request inference.
"""

import json
import os
import pandas as pd
from catboost import CatBoostClassifier

# =========================================================================
# Model loader (runs once when container starts)
# =========================================================================
MODELS = {}

def model_fn(model_dir):
    """
    Loads all four models from the SageMaker model directory.
    model_dir is usually /opt/ml/model inside the container.
    """
    # Because tar usually contains a top-level "model/" folder
    base_dir = os.path.join(model_dir, "model")
    
    # Load Power User Models
    m = CatBoostClassifier()
    m.load_model(os.path.join(base_dir, "power_user_activity", "model.cb"))
    MODELS["power_user_activity"] = m
    
    m = CatBoostClassifier()
    m.load_model(os.path.join(base_dir, "power_user_profile", "model.cb"))
    MODELS["power_user_profile"] = m

    # Load Casual Models
    m = CatBoostClassifier()
    m.load_model(os.path.join(base_dir, "casual_activity", "model.cb"))
    MODELS["casual_activity"] = m
    
    m = CatBoostClassifier()
    m.load_model(os.path.join(base_dir, "casual_profile", "model.cb"))
    MODELS["casual_profile"] = m

    return MODELS

def predict_fn(input_data, models):
    """
    Executes prediction on incoming JSON request.
    """
    # Parse incoming JSON payload
    data = json.loads(input_data)
    df = pd.DataFrame(data['features'])
    segment = data.get('segment', 'casual')
    
    # Route to the correct model based on segment logic
    if segment == 'power_user':
        model = models["power_user_activity"] 
    else:
        model = models["casual_activity"]
        
    predictions = model.predict_proba(df)[:, 1]
    
    return json.dumps({"risk_scores": predictions.tolist()})