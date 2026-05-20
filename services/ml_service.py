import os
import json
import joblib
import pandas as pd
from core.config import settings
from core.constants import TYPE_MAP, FEATURES_ORDER, CONTINUOUS_COLS

class MLService:
    """
    Service layer orchestrating machine failure prediction and model info retrieval.
    Caches model files in memory for fast performance.
    """
    def __init__(self):
        self._rf_model = None
        self._lr_model = None
        self._scaler = None
        self._model_summary = None

    def load_models(self):
        """
        Loads models and scalers into memory if not already cached.
        """
        rf_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "random_forest.pkl")
        lr_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "logistic_regression.pkl")
        scaler_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "scaler.pkl")
        summary_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "model_summary.json")

        if self._rf_model is None and os.path.exists(rf_path):
            try:
                self._rf_model = joblib.load(rf_path)
                print("[MLService] Loaded Random Forest model.")
            except Exception as e:
                print(f"[MLService Error] Failed to load Random Forest: {e}")

        if self._lr_model is None and os.path.exists(lr_path):
            try:
                self._lr_model = joblib.load(lr_path)
                print("[MLService] Loaded Logistic Regression model.")
            except Exception as e:
                print(f"[MLService Error] Failed to load Logistic Regression: {e}")

        if self._scaler is None and os.path.exists(scaler_path):
            try:
                self._scaler = joblib.load(scaler_path)
                print("[MLService] Loaded Standard Scaler.")
            except Exception as e:
                print(f"[MLService Error] Failed to load scaler: {e}")

        if self._model_summary is None and os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    self._model_summary = json.load(sf)
                print("[MLService] Loaded model summary metadata.")
            except Exception as e:
                print(f"[MLService Error] Failed to load model summary: {e}")

    def predict_failure(self, telemetry_data: dict, model_type: str = "random_forest") -> dict:
        """
        Preprocesses equipment telemetry variables and runs failure prediction using the chosen model.
        Returns predicted class (0/1) and failure probability.
        """
        self.load_models()
        
        # Select active model
        model = self._rf_model if model_type == "random_forest" else self._lr_model
        if model is None:
            raise FileNotFoundError(f"Selected model '{model_type}' is not trained or loaded. Run ML training pipeline first.")

        data = telemetry_data.copy()
        
        # Map category Type (H=0, L=1, M=2)
        if "Type" in data:
            if isinstance(data["Type"], str):
                data["Type"] = TYPE_MAP.get(data["Type"], 1)

        # Add engineered features
        data["temp_diff"] = data["Process temperature [K]"] - data["Air temperature [K]"]
        data["power"] = data["Torque [Nm]"] * data["Rotational speed [rpm]"]
        data["wear_torque_ratio"] = data["Tool wear [min]"] / (data["Torque [Nm]"] + 1)
        
        # Enforce feature order
        df = pd.DataFrame([data])[FEATURES_ORDER]
        
        # Scale continuous features for Logistic Regression
        if model_type == "logistic_regression" and self._scaler is not None:
            df[CONTINUOUS_COLS] = self._scaler.transform(df[CONTINUOUS_COLS])
            
        pred_class = int(model.predict(df)[0])
        pred_prob = float(model.predict_proba(df)[0][1])
        
        return {
            "prediction": pred_class,
            "probability": pred_prob,
            "model_used": model_type,
            "engineered_features": {
                "temp_diff": data["temp_diff"],
                "power": data["power"],
                "wear_torque_ratio": data["wear_torque_ratio"]
            }
        }

    def get_model_status(self) -> dict:
        """
        Returns model parameters, F1 metrics, training timestamps, and details.
        """
        self.load_models()
        if self._model_summary:
            return self._model_summary
        return {
            "run_timestamp": None,
            "best_model": "None",
            "best_f1": 0.0,
            "best_roc_auc": 0.0,
            "best_params": {},
            "top_features": [],
            "failure_rate_in_dataset": 0.0,
            "status": "No model has been trained yet."
        }

# Global ML Service instance
ml_service = MLService()
