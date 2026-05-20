import os
import sys
import json
import argparse
import joblib
import pandas as pd

# Ensure base dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from core.constants import FEATURES_ORDER, CONTINUOUS_COLS, TYPE_MAP, REVERSE_TYPE_MAP

class TelemetryPredictor:
    """In-process classifier using trained ML models to predict equipment failures."""
    
    def __init__(self):
        self.artifacts_dir = settings.MODEL_ARTIFACTS_DIR
        self.summary_path = self.artifacts_dir / "model_summary.json"
        
        # Load best model config
        self.best_model_name = "Random Forest"
        if self.summary_path.exists():
            try:
                with open(self.summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    self.best_model_name = summary.get("best_model", "Random Forest")
            except Exception:
                pass
        
        # Load models and scaler
        self.scaler = joblib.load(self.artifacts_dir / "scaler.pkl")
        self.rf_model = joblib.load(self.artifacts_dir / "random_forest.pkl")
        self.lr_model = joblib.load(self.artifacts_dir / "logistic_regression.pkl")
        
        # Select active model
        if self.best_model_name == "Logistic Regression":
            self.model = self.lr_model
            self.requires_scaling = True
        else:
            self.model = self.rf_model
            self.requires_scaling = False
            
        print(f"[ML] Predictor loaded with active model: {self.best_model_name}")

    def preprocess_input(self, raw_input: dict) -> pd.DataFrame:
        """Applies type mapping and feature engineering on a single input row."""
        data = raw_input.copy()
        
        # Convert Type (e.g. 'L', 'M', 'H') to integer
        t_val = data.get("Type", "L")
        if isinstance(t_val, str):
            data["Type"] = TYPE_MAP.get(t_val, 1)
        
        # Standardize temperature values to floats
        air_temp = float(data.get("Air temperature [K]", 300.0))
        proc_temp = float(data.get("Process temperature [K]", 310.0))
        speed = float(data.get("Rotational speed [rpm]", 1500.0))
        torque = float(data.get("Torque [Nm]", 40.0))
        wear = float(data.get("Tool wear [min]", 0.0))
        
        # Build features
        data["Air temperature [K]"] = air_temp
        data["Process temperature [K]"] = proc_temp
        data["Rotational speed [rpm]"] = speed
        data["Torque [Nm]"] = torque
        data["Tool wear [min]"] = wear
        
        # Add engineered features
        data["temp_diff"] = proc_temp - air_temp
        data["power"] = torque * speed
        data["wear_torque_ratio"] = wear / (torque + 1.0)
        
        # Convert to DataFrame with strict feature ordering
        df = pd.DataFrame([data])[FEATURES_ORDER]
        
        # Apply scaling if required (e.g., for Logistic Regression)
        if self.requires_scaling:
            df_scaled = df.copy()
            df_scaled[CONTINUOUS_COLS] = self.scaler.transform(df[CONTINUOUS_COLS])
            return df_scaled
            
        return df

    def predict(self, raw_input: dict) -> dict:
        """Runs feature engineering and predicts probability and binary class."""
        try:
            df = self.preprocess_input(raw_input)
            
            # Predict
            pred_class = int(self.model.predict(df)[0])
            pred_prob = float(self.model.predict_proba(df)[0][1])
            
            # Formulate failure explanation based on components
            explanation = []
            torque = float(raw_input.get("Torque [Nm]", 40.0))
            speed = float(raw_input.get("Rotational speed [rpm]", 1500.0))
            wear = float(raw_input.get("Tool wear [min]", 0.0))
            temp_diff = float(raw_input.get("Process temperature [K]", 310.0)) - float(raw_input.get("Air temperature [K]", 300.0))
            
            if pred_class == 1:
                explanation.append("High probability of machine failure detected.")
                # Root cause rules
                if temp_diff > 8.5:
                    explanation.append("Root Cause: Heat Dissipation Failure (HDF) - temperature delta exceeds threshold.")
                if torque * speed > 9000:
                    explanation.append("Root Cause: Power Failure (PWF) - power stress exceeds machine limit.")
                if wear > 200:
                    explanation.append("Root Cause: Tool Wear Failure (TWF) - replacement limit reached.")
                if torque > 65 or torque < 10:
                    explanation.append("Root Cause: Overstrain Failure (OSF) - torque stress outside operating envelope.")
            else:
                explanation.append("Machine operating parameters are within normal boundaries.")
                
            return {
                "prediction": pred_class,
                "probability": round(pred_prob, 4),
                "model_used": self.best_model_name,
                "status": "FAILURE DETECTED" if pred_class == 1 else "OPERATIONAL",
                "explanation": " ".join(explanation)
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "ERROR"
            }

def main():
    parser = argparse.ArgumentParser(description="Predict equipment failure from raw telemetry input.")
    parser.add_argument("--json", type=str, help="Raw telemetry as a JSON string.")
    parser.add_argument("--type", type=str, default="L", choices=["H", "L", "M"], help="Machine Type (H, L, M).")
    parser.add_argument("--air-temp", type=float, default=300.0, help="Air Temperature [K].")
    parser.add_argument("--process-temp", type=float, default=310.0, help="Process Temperature [K].")
    parser.add_argument("--speed", type=float, default=1500.0, help="Rotational Speed [rpm].")
    parser.add_argument("--torque", type=float, default=40.0, help="Torque [Nm].")
    parser.add_argument("--wear", type=float, default=50.0, help="Tool wear [min].")

    args = parser.parse_args()
    
    # Parse input dict
    if args.json:
        try:
            raw_input = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)
    else:
        raw_input = {
            "Type": args.type,
            "Air temperature [K]": args.air_temp,
            "Process temperature [K]": args.process_temp,
            "Rotational speed [rpm]": args.speed,
            "Torque [Nm]": args.torque,
            "Tool wear [min]": args.wear
        }

    predictor = TelemetryPredictor()
    res = predictor.predict(raw_input)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
