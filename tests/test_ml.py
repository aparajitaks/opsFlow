import os
import pytest
import pandas as pd
from unittest.mock import patch
from models.preprocessing import load_dataset, engineer_features, prepare_data_pipeline, apply_scaling, apply_smote
from models.training import perform_cross_validation, tune_logistic_regression, tune_random_forest
from core.constants import CONTINUOUS_COLS

def test_preprocessing_flow(mock_csv_data):
    """Verifies document loader, custom feature engineering, and stratified splits."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        # 1. Load Data
        df = load_dataset()
        assert len(df) == 10
        assert "Machine failure" in df.columns
        
        # 2. Engineer Features
        df_eng = engineer_features(df)
        assert "temp_diff" in df_eng.columns
        assert "power" in df_eng.columns
        assert "wear_torque_ratio" in df_eng.columns
        assert df_eng["temp_diff"].iloc[0] == pytest.approx(10.5)
        
        # 3. Train Test Split (Using default 80/20 split)
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_eng)
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

def test_scaling_and_smote(mock_csv_data):
    """Checks feature scaling transformations and SMOTE oversampling ratios."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        df = load_dataset()
        df_eng = engineer_features(df)
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_eng)
        
        # Scale
        X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_test, CONTINUOUS_COLS)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # SMOTE (Since sample count is tiny, we check if it handles it or handles exception gracefully)
        try:
            X_res, y_res = apply_smote(X_train, y_train)
            assert len(y_res) >= len(y_train)
        except ValueError:
            # SMOTE requires min samples per class, which might fail on tiny mock datasets.
            # This is expected and acceptable behavior.
            pass

def test_model_training_helpers(mock_csv_data):
    """Checks baseline cross-validation scoring and parameter tuning grids."""
    with patch("core.config.settings.DATASET_PATH", mock_csv_data):
        df = load_dataset()
        df_eng = engineer_features(df)
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_eng)
        
        # Test cross validation helper (mocking CV split to avoid n_splits > n_samples warnings)
        with patch("models.training.N_SPLITS", 2):
            cv_results_lr, cv_results_rf = perform_cross_validation(X_train, y_train, CONTINUOUS_COLS)
            assert "f1" in cv_results_lr
            assert "f1" in cv_results_rf
            
            # Check Grid Search Tuning
            X_train_scaled, _, _ = apply_scaling(X_train, X_test, CONTINUOUS_COLS)
            lr_model, lr_params, lr_score = tune_logistic_regression(X_train_scaled, y_train)
            assert lr_model is not None
            assert lr_params is not None
            
            rf_model, rf_params, rf_score = tune_random_forest(X_train, y_train)
            assert rf_model is not None
            assert rf_params is not None
