import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from models.train import (
    load_dataset,
    engineer_features,
    prepare_data_pipeline,
    perform_cross_validation,
    tune_pipeline
)
from models.predict import TelemetryPredictor

def test_preprocessing_and_features(mock_csv_data):
    """Verifies loader and custom feature engineering logic."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        df = load_dataset()
        assert len(df) == 10
        assert "Machine failure" in df.columns
        
        # Test feature engineering
        df_eng = engineer_features(df)
        assert "temp_diff" in df_eng.columns
        assert "power" in df_eng.columns
        assert "wear_torque_ratio" in df_eng.columns
        # Air = 300.0, Process = 310.5 -> temp_diff = 10.5
        assert df_eng["temp_diff"].iloc[0] == pytest.approx(10.5)


def test_type_encoding_string_dtype():
    """Linux CI: pandas StringDtype must be encoded (not only object dtype)."""
    df = pd.DataFrame({
        "Type": pd.Series(["L", "M", "H", "L"], dtype="string"),
        "Air temperature [K]": [300.0, 301.0, 302.0, 303.0],
        "Process temperature [K]": [310.0, 311.0, 312.0, 313.0],
        "Rotational speed [rpm]": [1500.0, 1501.0, 1502.0, 1503.0],
        "Torque [Nm]": [40.0, 41.0, 42.0, 43.0],
        "Tool wear [min]": [0.0, 1.0, 2.0, 3.0],
        "Machine failure": [0, 0, 1, 0],
    })
    eng = engineer_features(df)
    assert pd.api.types.is_integer_dtype(eng["Type"])

def test_pipeline_splits(mock_csv_data):
    """Checks train_test_split and leakage prevention columns removal."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        df = load_dataset()
        df_eng = engineer_features(df)
        
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_eng)
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert "Machine failure" not in X_train.columns
        assert "TWF" not in X_train.columns
        assert "HDF" not in X_train.columns

def test_cross_validation_and_tuning(mock_csv_data):
    """Verifies baseline CV runs and grid searches on mock data splits."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        df = load_dataset()
        df_eng = engineer_features(df)
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_eng)
        
        # Mock cross validation splitting and GridSearch limits to match tiny mock sample sizes
        with patch("core.config.settings.N_CV_SPLITS", 2):
            lr_cv, rf_cv = perform_cross_validation(X_train, y_train, n_splits=2)
            assert "f1" in lr_cv
            assert "roc_auc" in rf_cv
            
            # Check Grid Search tuning on mock splits
            results = tune_pipeline(X_train, y_train, model_name="both")
            
            assert "logistic_regression" in results
            assert "random_forest" in results
            assert results["logistic_regression"]["pipeline"] is not None
            assert results["random_forest"]["pipeline"] is not None

def test_telemetry_predictor(mock_csv_data):
    """Validates the in-process TelemetryPredictor classifies telemetry inputs correctly."""
    # Mock model files existence and return predictions
    with patch("os.path.exists", return_value=True), \
         patch("joblib.load") as mock_load:
        
        # Mock V3 sklearn Pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [1]
        mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
        
        mock_load.return_value = mock_pipeline
        
        predictor = TelemetryPredictor()
        
        test_input = {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 50.0
        }
        
        res = predictor.predict(test_input)
        assert res["prediction"] == 1
        assert res["probability"] == 0.9000
        assert res["status"] == "FAILURE DETECTED"
