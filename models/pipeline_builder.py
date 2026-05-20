"""sklearn Pipeline construction for Task 3."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import settings


def build_pipeline(classifier) -> Pipeline:
    """Preprocessor (in-fold scaling) + classifier — leakage-safe by design."""
    continuous = [c for c in settings.CONTINUOUS_COLS if c != "Type"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), continuous)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
