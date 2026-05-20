"""Tests for dataset integrity validation."""
import pandas as pd
import pytest

from core.data_validation import DatasetValidationError, validate_dataset
from core.config import settings


def _minimal_valid_df(n=20):
    rows = []
    for i in range(n):
        rows.append({
            "Type": "L" if i % 3 else "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": float(i),
            settings.TARGET_COL: i % 5 == 0,
        })
    return pd.DataFrame(rows)


def test_validate_dataset_passes():
    df = validate_dataset(_minimal_valid_df(30))
    assert len(df) == 30


def test_validate_dataset_rejects_missing_target():
    df = _minimal_valid_df(10)
    df = df.drop(columns=[settings.TARGET_COL])
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_validate_dataset_rejects_single_class():
    df = _minimal_valid_df(10)
    df[settings.TARGET_COL] = 0
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)
