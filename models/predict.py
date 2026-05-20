"""
models/predict.py — Task 3: In-process Telemetry Predictor
"""
import argparse
import json

import pandas as pd

from core.config import settings
from core.exceptions import ModelArtifactError
from core.logger import get_logger
from core.types import PredictionResult
from models.artifacts import ModelArtifactStore
from models.features import engineer_telemetry_row

log = get_logger("models.predict")


class TelemetryPredictor:
    """Real-time failure classifier using persisted sklearn Pipeline artifacts."""

    def __init__(self, store: ModelArtifactStore | None = None):
        self._store = store or ModelArtifactStore()
        self.best_model_name = self._store.resolve_best_model_name()
        self.pipeline = self._store.load_pipeline(self.best_model_name)
        log.info(f"TelemetryPredictor ready — active model: {self.best_model_name}")

    def predict(self, raw_input: dict) -> PredictionResult:
        try:
            df = engineer_telemetry_row(raw_input)
            pred_class = int(self.pipeline.predict(df)[0])
            pred_prob = float(self.pipeline.predict_proba(df)[0][1])
            explanation = self._build_explanation(raw_input, pred_class)

            result: PredictionResult = {
                "prediction": pred_class,
                "probability": round(pred_prob, 4),
                "model_used": self.best_model_name,
                "status": "FAILURE DETECTED" if pred_class == 1 else "OPERATIONAL",
                "explanation": explanation,
            }
            log.info(
                f"Prediction: {result['status']} | "
                f"Probability: {pred_prob:.4f} | Model: {self.best_model_name}"
            )
            return result
        except ModelArtifactError:
            raise
        except Exception as e:
            log.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e), "status": "ERROR"}

    def _build_explanation(self, raw: dict, pred_class: int) -> str:
        if pred_class == 0:
            return "Machine operating parameters are within normal boundaries."

        torque = float(raw.get("Torque [Nm]", 40.0))
        speed = float(raw.get("Rotational speed [rpm]", 1500.0))
        wear = float(raw.get("Tool wear [min]", 0.0))
        air_temp = float(raw.get("Air temperature [K]", 300.0))
        proc_temp = float(raw.get("Process temperature [K]", 310.0))
        temp_diff = proc_temp - air_temp

        reasons = ["High probability of machine failure detected."]
        if temp_diff > settings.HDF_TEMP_DELTA:
            reasons.append(
                f"Root Cause: Heat Dissipation Failure (HDF) — "
                f"ΔT={temp_diff:.1f}K exceeds threshold {settings.HDF_TEMP_DELTA}K."
            )
        if torque * speed > settings.PWF_POWER:
            reasons.append(
                f"Root Cause: Power Failure (PWF) — "
                f"power={torque * speed:.0f} Nm·rpm exceeds limit {settings.PWF_POWER:.0f}."
            )
        if wear > settings.TWF_WEAR:
            reasons.append(
                f"Root Cause: Tool Wear Failure (TWF) — "
                f"wear={wear:.0f} min exceeds limit {settings.TWF_WEAR:.0f} min."
            )
        if torque > settings.OSF_TORQUE_HIGH or torque < settings.OSF_TORQUE_LOW:
            reasons.append(
                f"Root Cause: Overstrain Failure (OSF) — "
                f"torque={torque:.1f} Nm outside [{settings.OSF_TORQUE_LOW}, {settings.OSF_TORQUE_HIGH}] Nm."
            )
        return " ".join(reasons)


def main():
    parser = argparse.ArgumentParser(description="Predict equipment failure from telemetry.")
    parser.add_argument("--json", type=str, help="Raw telemetry as a JSON string.")
    parser.add_argument("--type", type=str, default="L", choices=["H", "L", "M"])
    parser.add_argument("--air-temp", type=float, default=300.0)
    parser.add_argument("--process-temp", type=float, default=310.0)
    parser.add_argument("--speed", type=float, default=1500.0)
    parser.add_argument("--torque", type=float, default=40.0)
    parser.add_argument("--wear", type=float, default=50.0)
    args = parser.parse_args()

    if args.json:
        try:
            raw_input = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise SystemExit(1) from e
    else:
        raw_input = {
            "Type": args.type,
            "Air temperature [K]": args.air_temp,
            "Process temperature [K]": args.process_temp,
            "Rotational speed [rpm]": args.speed,
            "Torque [Nm]": args.torque,
            "Tool wear [min]": args.wear,
        }

    predictor = TelemetryPredictor()
    result = predictor.predict(raw_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
