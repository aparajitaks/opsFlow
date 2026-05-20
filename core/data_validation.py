"""
Dataset schema and integrity validation for Task 3 telemetry pipeline.
"""
from __future__ import annotations

import pandas as pd

from core.config import settings
from core.exceptions import DatasetValidationError  # noqa: F401 — re-exported

__all__ = ["validate_dataset", "DatasetValidationError"]
from core.logger import get_logger

log = get_logger("core.data_validation")

REQUIRED_RAW_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    settings.TARGET_COL,
]

NUMERIC_BOUNDS = {
    "Air temperature [K]": (200.0, 400.0),
    "Process temperature [K]": (200.0, 450.0),
    "Rotational speed [rpm]": (0.0, 10000.0),
    "Torque [Nm]": (0.0, 200.0),
    "Tool wear [min]": (0.0, 500.0),
}


def validate_dataset(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """
    Validate AI4I dataset shape, schema, nulls, and basic value ranges.
    Returns cleaned copy (drops fully empty rows).
    """
    if df is None or df.empty:
        raise DatasetValidationError("Dataset is empty.")

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")

    out = df.copy()
    before = len(out)
    out = out.dropna(how="all")
    if len(out) < before:
        log.warning(f"Dropped {before - len(out)} fully empty rows.")

    # Target must be binary 0/1
    target = out[settings.TARGET_COL]
    if not target.isin([0, 1]).all():
        bad = target[~target.isin([0, 1])].unique()[:5]
        raise DatasetValidationError(f"Target column has non-binary values: {bad}")

    if target.nunique() < 2:
        raise DatasetValidationError("Target column has only one class — cannot train classifier.")

    # Type values
    if out["Type"].dtype == object:
        invalid_types = set(out["Type"].unique()) - set(settings.TYPE_MAP.keys())
        if invalid_types and strict:
            raise DatasetValidationError(f"Unknown Type values: {invalid_types}")

    # Nulls in critical columns
    critical = [c for c in REQUIRED_RAW_COLUMNS if c in out.columns]
    null_counts = out[critical].isna().sum()
    bad_nulls = null_counts[null_counts > 0]
    if not bad_nulls.empty:
        if strict:
            raise DatasetValidationError(f"Null values in columns: {bad_nulls.to_dict()}")
        log.warning(f"Null values present (non-strict): {bad_nulls.to_dict()}")
        out = out.dropna(subset=critical)

    # Range checks (warn or fail on extreme outliers)
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        oob = series[(series < lo) | (series > hi)]
        if len(oob) > 0:
            pct = 100.0 * len(oob) / len(out)
            msg = f"{col}: {len(oob)} rows ({pct:.2f}%) outside [{lo}, {hi}]"
            if strict and pct > 5.0:
                raise DatasetValidationError(msg)
            log.warning(msg)

    if len(out) < 50:
        log.warning(f"Small dataset after validation: {len(out)} rows.")

    log.info(f"Dataset validation passed — {len(out)} rows, failure rate {out[settings.TARGET_COL].mean():.4f}")
    return out
