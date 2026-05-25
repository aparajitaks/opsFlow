"""ML artifact persistence and reliable loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from core.config import settings
from core.exceptions import ModelArtifactError
from core.logger import get_logger

log = get_logger("ml.artifacts")

MODEL_FILE_MAP = {
    "Random Forest": "random_forest_pipeline.pkl",
    "Logistic Regression": "logistic_regression_pipeline.pkl",
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
        """Load trained pipeline artifact."""
        name = model_name or self.resolve_best_model_name()
        filename = MODEL_FILE_MAP.get(
            name, "random_forest_pipeline.pkl"
        )
        path = self.artifacts_dir / filename

        if path.exists():
            log.debug(f"Loading pipeline: {path}")
            return joblib.load(path)
        raise ModelArtifactError(
            f"No model artifact at {path}. Run training first."
        )

    def save_pipeline(self, pipeline, model_key: str) -> Path:
        """Persist pipeline under standard naming convention."""
        filenames = {
            "random_forest": "random_forest_pipeline.pkl",
            "logistic_regression": "logistic_regression_pipeline.pkl",
        }
        filename = filenames[model_key]
        path = self.artifacts_dir / filename
        joblib.dump(pipeline, path)
        log.info(f"Pipeline saved → {path}")
        return path

    def artifacts_available(self) -> bool:
        return any(
            (self.artifacts_dir / filename).exists()
            for filename in MODEL_FILE_MAP.values()
        )
