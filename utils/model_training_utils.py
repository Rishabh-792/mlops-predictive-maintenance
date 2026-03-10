"""
model_training_utils.py

Model training and evaluation utilities for CatBoost binary classifiers.
Handles training, metric calculation, and MLflow experiment tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

def train_and_log_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: List[str],
    params: Dict,
    run_name: str,
) -> Tuple[CatBoostClassifier, str]:
    """Core training function with MLflow logging."""
    
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        logger.info(f"Training CatBoost model: {run_name}")
        
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        model = CatBoostClassifier(**params, cat_features=cat_features, verbose=100)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        # Predictions and metrics
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "roc_auc": roc_auc_score(y_val, y_prob),
            # Add custom business metrics here
        }
        mlflow.log_metrics(metrics)
        
        # Infer signature and log model
        signature = infer_signature(X_val, y_pred)
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=run_name
        )
        
        logger.info(f"Model logged to MLflow URI: {model_info.model_uri}")
        return model, model_info.model_uri


def train_segment_model(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_type: str, 
    segment: str,
    target_col: str,
    # additional config params...
) -> Tuple[CatBoostClassifier, str]:
    """
    Segment-specific model trainer.
    Handles data prep, missing values, and delegates to the core MLflow trainer.
    """
    logger.info(f"Preparing data for Segment: {segment} | Type: {model_type}")
    
    # [Insert your data preparation, class imbalance, and split logic here]
    # MOCK DATA FOR DEMONSTRATION
    X_train, X_val = feature_df.copy(), feature_df.copy()
    y_train, y_val = raw_df[target_col], raw_df[target_col]
    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    
    params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "eval_metric": "AUC",
        "random_seed": 42
    }
    
    run_name = f"{segment}_{model_type}_churn_model"
    return train_and_log_catboost(X_train, y_train, X_val, y_val, cat_features, params, run_name)