"""
feature_builders.py

Data pipeline utilities for feature engineering and target construction.
Translates raw engagement events into predictive ML features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def split_by_segment(df: pd.DataFrame, thresholds: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits users into segments based on historical session counts."""
    counts = df.groupby('user_id').size()
    
    power_users = counts[counts >= thresholds['power_user']].index
    guests = counts[counts < thresholds['guest']].index
    casuals = counts[(counts >= thresholds['guest']) & (counts < thresholds['power_user'])].index

    return (
        df[df['user_id'].isin(power_users)].copy(),
        df[df['user_id'].isin(casuals)].copy(),
        df[df['user_id'].isin(guests)].copy()
    )

def build_power_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds rich feature set for highly active users (gap stats, trends)."""
    logger.info("Building features for Power Users...")
    features = df.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_session_duration=('duration_minutes', 'mean'),
        std_session_duration=('duration_minutes', 'std'),
        last_active_date=('event_date', 'max')
    ).reset_index()
    
    # Fill NaNs for standard deviation if only 1 session exists
    features['std_session_duration'] = features['std_session_duration'].fillna(0)
    return features

def build_guest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds conservative, cold-start features for new users."""
    logger.info("Building cold-start features for Guests...")
    features = df.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        first_platform=('device_type', 'first')
    ).reset_index()
    return features