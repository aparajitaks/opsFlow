# Shared constants for ML models and telemetry processing

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# Categorical mapping for telemetry 'Type' feature
TYPE_MAP = {"H": 0, "L": 1, "M": 2}
REVERSE_TYPE_MAP = {0: "H", 1: "L", 2: "M"}

# Features order as required by the model inputs pipeline
FEATURES_ORDER = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
    "wear_torque_ratio"
]

# Continuous features requiring StandardScaler in Logistic Regression
CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
    "wear_torque_ratio"
]
