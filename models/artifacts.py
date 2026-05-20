"""ML artifact persistence and reliable loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from core.config import settings
from core.exceptions import ModelArtifactError
from core.logger import get_logger

log = get_logger("models.artifacts")

MODEL_FILE_MAP = {
    "Random Forest": ("random_forest_pipeline.pkl", "random_forest.pkl"),
    "Logistic Regression": ("logistic_regression_pipeline.pkl", "logistic_regression.pkl"),
}


class ModelArtifactStore:
    """Centralized access to trained pipelines and training summaries."""

    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = artifacts_dir or settings.MODEL_ARTIFACTS_DIR
        self.summary_path = self.artifacts_dir / "model_summary.json"

    def load_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {"best_model": "Random Forest"}
        try:
            with open(self.summary_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not read model summary: {e}")
            return {"best_model": "Random Forest"}

    def resolve_best_model_name(self) -> str:
        return self.load_summary().get("best_model", "Random Forest")

    def load_pipeline(self, model_name: str | None = None):
        """Load V3 pipeline artifact with legacy fallback."""
        name = model_name or self.resolve_best_model_name()
        v3_name, legacy_name = MODEL_FILE_MAP.get(
            name, ("random_forest_pipeline.pkl", "random_forest.pkl")
        )
        v3_path = self.artifacts_dir / v3_name
        legacy_path = self.artifacts_dir / legacy_name

        if v3_path.exists():
            log.debug(f"Loading pipeline: {v3_path}")
            return joblib.load(v3_path)
        if legacy_path.exists():
            log.warning(f"Falling back to legacy artifact: {legacy_path}")
            return joblib.load(legacy_path)
        raise ModelArtifactError(
            f"No model artifact at {v3_path} or {legacy_path}. Run training first."
        )

    def save_pipeline(self, pipeline, model_key: str) -> Path:
        """Persist pipeline under standard naming convention."""
        filenames = {
            "random_forest": ("random_forest_pipeline.pkl", "random_forest.pkl"),
            "logistic_regression": ("logistic_regression_pipeline.pkl", "logistic_regression.pkl"),
        }
        v3_name, legacy_name = filenames[model_key]
        v3_path = self.artifacts_dir / v3_name
        joblib.dump(pipeline, v3_path)
        joblib.dump(pipeline, self.artifacts_dir / legacy_name)
        log.info(f"Pipeline saved → {v3_path}")
        return v3_path

    def artifacts_available(self) -> bool:
        return any(
            (self.artifacts_dir / name).exists()
            for pair in MODEL_FILE_MAP.values()
            for name in pair
        )
