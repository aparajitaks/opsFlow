import sys
import os
import numpy as np
import joblib
import json
import datetime
from core.config import settings

def run_ml_pipeline():
    """
    Safe Inference-only ML pipeline (Hardened for Deployment).
    Loads saved artifacts only and performs dummy inference to verify model sanity.
    It does NOT download datasets, retrain models, or preprocess raw training datasets.
    """
    print("=================================================================")
    print("      ML PIPELINE: SAFE INFERENCE-ONLY STUB / VERIFICATION       ")
    print("=================================================================")
    
    lr_model_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "logistic_regression.pkl")
    rf_model_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "random_forest.pkl")
    scaler_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "scaler.pkl")
    
    if not os.path.exists(lr_model_path) or not os.path.exists(rf_model_path):
        print("Pre-trained model artifacts not found. Verification bypassed.")
        return
        
    try:
        # Load saved artifacts
        print(f"Loading pre-trained Logistic Regression: {lr_model_path}")
        lr_model = joblib.load(lr_model_path)
        print(f"Loading pre-trained Random Forest: {rf_model_path}")
        rf_model = joblib.load(rf_model_path)
        
        if os.path.exists(scaler_path):
            print(f"Loading pre-trained StandardScaler: {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
            
        # Perform dummy inference to verify sanity
        # Features map: [Type, Air temp, Process temp, Rotational speed, Torque, Tool wear, temp_diff, power, wear_torque_ratio]
        # Type = 1 (L), Air temp = 298.1, Process temp = 308.6, Rotational speed = 1500, Torque = 40.0, Tool wear = 50
        dummy_input = np.array([[1, 298.1, 308.6, 1500, 40.0, 50, 10.5, 60000.0, 1.2195]])
        
        # Test RF inference
        rf_pred = rf_model.predict(dummy_input)
        rf_prob = rf_model.predict_proba(dummy_input)[:, 1]
        print(f"Random Forest Dummy Prediction: {rf_pred[0]} (Probability: {rf_prob[0]:.4f})")
        
        # Test LR inference (with optional scaling)
        if scaler:
            scaled_input = dummy_input.copy()
            scaled_features = scaler.transform(dummy_input[:, 1:9])
            scaled_input[:, 1:9] = scaled_features
            lr_pred = lr_model.predict(scaled_input)
            lr_prob = lr_model.predict_proba(scaled_input)[:, 1]
        else:
            lr_pred = lr_model.predict(dummy_input)
            lr_prob = lr_model.predict_proba(dummy_input)[:, 1]
        print(f"Logistic Regression Dummy Prediction: {lr_pred[0]} (Probability: {lr_prob[0]:.4f})")
        
        print("\n=================================================================")
        print("      ML PIPELINE VERIFICATION PASSED SUCCESSFULLY              ")
        print("=================================================================")
    except Exception as e:
        print(f"Error during model artifact validation: {e}")

if __name__ == "__main__":
    run_ml_pipeline()
