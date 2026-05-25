"""Feature engineering shared by training and inference."""
import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import settings
from core.logger import get_logger

log = get_logger("ml.features")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and encode Type for batch training data."""
    out = df.copy()
    drop_ids = [c for c in settings.DROP_ID_COLS if c in out.columns]
    out = out.drop(columns=drop_ids)

    if "Type" in out.columns and not pd.api.types.is_numeric_dtype(out["Type"]):
        out["Type"] = out["Type"].astype(str).map(settings.TYPE_MAP).fillna(1).astype(int)

    out["temp_diff"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["power"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"]
    out["wear_torque_ratio"] = out["Tool wear [min]"] / (out["Torque [Nm]"] + 1)

    log.info("Feature engineering applied: temp_diff, power, wear_torque_ratio")
    return out


def engineer_telemetry_row(raw: dict) -> pd.DataFrame:
    """Single-row feature engineering for real-time inference (parity with training)."""
    data = raw.copy()
    t_val = data.get("Type", "L")
    if isinstance(t_val, str):
        data["Type"] = settings.TYPE_MAP.get(t_val, 1)

    air_temp = float(data.get("Air temperature [K]", 300.0))
    proc_temp = float(data.get("Process temperature [K]", 310.0))
    speed = float(data.get("Rotational speed [rpm]", 1500.0))
    torque = float(data.get("Torque [Nm]", 40.0))
    wear = float(data.get("Tool wear [min]", 0.0))

    data.update({
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "temp_diff": proc_temp - air_temp,
        "power": torque * speed,
        "wear_torque_ratio": wear / (torque + 1.0),
    })
    return pd.DataFrame([data])[settings.FEATURES_ORDER]


def prepare_data_pipeline(df: pd.DataFrame):
    """Stratified train/test split with leakage columns removed."""
    if settings.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{settings.TARGET_COL}' missing from dataset.")

    y = df[settings.TARGET_COL]
    drop_cols = [c for c in settings.LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)

    log.info(f"Leakage columns removed: {drop_cols}")
    log.info(f"Features: {list(X.columns)}  |  Class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )
    log.info(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test
