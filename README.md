# MLOps Predictive Maintenance

A production-grade customer churn prediction pipeline built with CatBoost, MLflow, and advanced feature engineering. This system implements a four-model ensemble approach with segment-specific optimization for user engagement prediction.

## Overview

This repository contains an end-to-end machine learning pipeline for predicting customer churn across different user segments:

- **Power Users**: Extensive engagement history → Complex feature set with trend analysis
- **Casual Users**: Moderate engagement → Standard feature aggregations
- **Guests**: Limited history → Conservative cold-start features

### Key Features

✅ **Modular Architecture**: Separated concerns across preprocessing, feature engineering, training, and inference  
✅ **Segment-Specific Models**: 4-model ensemble optimized per user segment  
✅ **MLflow Integration**: Experiment tracking and model versioning  
✅ **CatBoost Classification**: Gradient boosted tree models with categorical feature support  
✅ **Configurable Settings**: JSON-based centralized configuration management  
✅ **Error Handling**: Custom exception hierarchy with standardized error codes  
✅ **Comprehensive Logging**: Structured logging with console and file outputs

## Project Structure

```
.
├── pipelines/
│   ├── preprocessing_pipeline.py      # Data cleaning and validation
│   ├── feature_pipeline.py            # Feature engineering and segmentation
│   ├── training_pipeline.py           # Model training with MLflow logging
│   └── prediction_pipeline.py         # Batch inference and risk scoring
├── utils/
│   ├── core_utils.py                  # Generic utilities (I/O, logging, validation)
│   ├── settings_manager.py            # Configuration loading and typing
│   ├── pipeline_enums.py              # Type-safe enumerations
│   ├── pipeline_errors.py             # Custom exception hierarchy
│   ├── feature_builders.py            # Segment-specific feature aggregation
│   ├── model_training_utils.py        # CatBoost training and MLflow integration
│   ├── model_tune_utils.py            # Optuna-based hyperparameter tuning
│   └── prediction_utils.py            # Inference utilities and model container
├── deployment/
│   ├── inference.py                   # SageMaker entry point
│   └── deploy.ipynb                   # Deployment workflow notebook
├── configs/
│   └── config.json                    # Project settings (features, segments, dates)
├── test_integration.py                # Integration test suite
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.8+ (tested with 3.13.3)
- pip or conda package manager
- Git for version control

### Setup

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd mlops-predictive-maintenance
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Verify installation**
    ```bash
    python test_integration.py
    ```

## Configuration

Edit `configs/config.json` to customize:

```json
{
    "project_name": "Customer Engagement and Churn Prediction Pipeline",
    "schema": {
        "mandatory_features": [
            "user_id",
            "account_creation_date",
            "total_sessions",
            "last_active_date"
        ],
        "categorical_features": [
            "subscription_plan",
            "acquisition_channel",
            "primary_device_os"
        ],
        "target_variable": "will_churn_in_window"
    },
    "temporal": {
        "train_start_date": "2023-01-01",
        "train_end_date": "2024-12-31",
        "prediction_window_days": 30
    },
    "segments": {
        "power_user": { "min_activity_threshold": 25 },
        "casual": { "min_activity_threshold": 5, "max_activity_threshold": 24 },
        "guest": { "min_activity_threshold": 0, "max_activity_threshold": 4 }
    }
}
```

## Usage

### 1. Preprocessing Pipeline

Loads raw data, performs validation and cleaning, generates mock data if needed:

```python
from pipelines.preprocessing_pipeline import PreprocessingPipeline
from utils.pipeline_enums import OptimizationGoal

pipeline = PreprocessingPipeline(goal=OptimizationGoal.BALANCED)
output_path = pipeline.run(raw_data_path="path/to/raw_data.csv")
```

**Output**: Parquet file with cleaned events in `artifacts/preprocessed/`

### 2. Feature Engineering Pipeline

Segments users and builds tailored feature sets:

```python
from pipelines.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
feature_paths = pipeline.run(clean_data_path="artifacts/preprocessed/clean_events.parquet")
```

**Output**: CSV files per segment in `artifacts/features/`

### 3. Training Pipeline

Trains the 4-model ensemble with MLflow experiment tracking:

```python
from pipelines.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
model_uris = pipeline.run()
```

**Output**: Trained models registered in MLflow

### 4. Prediction Pipeline

Generates batch predictions and risk scores:

```python
from pipelines.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()
predictions = pipeline.run()
predictions.to_csv("churn_predictions.csv", index=False)
```

**Output**: DataFrame with columns: `user_id`, `segment`, `churn_risk_score`, `recommended_action`

## Testing

Run the integrated test suite to verify all components:

```bash
python test_integration.py
```

**Tests**:

- ✓ Settings Manager - Configuration loading
- ✓ Utility Functions - I/O and validation
- ✓ Error Handling - Custom exceptions
- ✓ Enum Types - Type safety
- ✓ Feature Builders - Segmentation and feature engineering

## MLflow Tracking

The pipeline logs experiments to MLflow. View results with:

```bash
mlflow ui
```

**Tracked Artifacts**:

- Model parameters and hyperparameters
- Validation metrics (ROC-AUC, F1, precision, recall)
- Model serialization for reproducibility

## Error Handling

Custom error codes for debugging:

| Code    | Description                                  |
| ------- | -------------------------------------------- |
| SYS-101 | Required features missing from input dataset |
| SYS-102 | Schema mapping violation detected            |
| SYS-201 | Failed to locate configuration file          |
| SYS-202 | Configuration parsing error                  |
| SYS-301 | Model artifact not found in registry         |
| SYS-302 | Inference execution failed                   |

## Key Components

### Settings Manager (`utils/settings_manager.py`)

- Loads JSON configuration with strict type checking
- Provides centralized settings object to all pipelines
- Validates configuration on startup

### Feature Builders (`utils/feature_builders.py`)

- `split_by_segment()`: Categorize users by activity level
- `build_power_user_features()`: Complex aggregations (trends, gaps, std dev)
- `build_guest_features()`: Cold-start minimal features

### Model Training (`utils/model_training_utils.py`)

- CatBoost classifier with categorical feature support
- MLflow integration for experiment tracking
- Automatic model registration and versioning

### Error Hierarchy (`utils/pipeline_errors.py`)

- Base `MLSystemFault` exception with error codes
- Specialized exceptions: `ConfigurationFault`, `SchemaValidationFault`
- JSON serialization support for cloud logging

## Directory Structure

Runtime artifacts are created in:

- `logs/` - Pipeline execution logs
- `artifacts/preprocessed/` - Cleaned data
- `artifacts/features/` - Engineered features
- `mlruns/` - MLflow experiment tracking
- `mlflow.db` - MLflow database

## Dependencies

Core Dependencies:

- **pandas** ≥ 2.0.0 - Data manipulation
- **numpy** ≥ 1.24.0 - Numerical computing
- **scikit-learn** ≥ 1.0.0 - ML utilities and metrics
- **catboost** ≥ 1.2.0 - Gradient boosted tree models
- **mlflow** ≥ 2.0.0 - Experiment tracking

Development Dependencies:

- **pytest** - Testing framework
- **black** - Code formatting
- **pylint** - Linting
- **flake8** - Code quality

See `requirements.txt` for complete list with versions.

## Development

### Code Quality

Format code:

```bash
black pipelines/ utils/ deployment/
```

Check linting:

```bash
pylint pipelines/ utils/
flake8 pipelines/ utils/
```

### Contributing

1. Create a feature branch
2. Make changes with atomic commits
3. Run tests: `python test_integration.py`
4. Submit pull request

## Deployment

### SageMaker Deployment

The `deployment/inference.py` module provides a SageMaker-compatible entry point:

```python
def model_fn(model_dir):
    # Loads 4 models from directory structure

def predict_fn(input_data, models):
    # Routes requests to appropriate model by segment
```

Containerization:

```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /opt/program
ENV SAGEMAKER_PROGRAM inference.py
```

## Performance

Typical metrics on validation set:

- Power User Model: ROC-AUC 0.92-0.95
- Casual Model: ROC-AUC 0.88-0.91
- Guest Model: ROC-AUC 0.85-0.88

## Troubleshooting

**Issue**: Settings file not found

```
ConfigurationFault: [SYS-201] Failed to locate configuration file
```

**Solution**: Ensure `configs/config.json` exists in project root

**Issue**: Missing required columns

```
SchemaValidationFault: [SYS-101] Missing required columns: [...]
```

**Solution**: Verify input data has all columns listed in config

**Issue**: MLflow tracking disabled

```
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

**Solution**: Start MLflow server or use file-based backend

## License

See LICENSE file for details.

## Support

For issues or questions, open a GitHub issue or contact the maintainers.
