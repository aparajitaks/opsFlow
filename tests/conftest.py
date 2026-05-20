import os
import sys
import tempfile
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure workspace root is in path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from core.config import settings

@pytest.fixture
def api_client():
    """Returns a FastAPI TestClient instance."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_telemetry_data():
    """Returns sample telemetry data dictionary."""
    return {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0
    }

@pytest.fixture
def mock_csv_data(tmp_path):
    """Creates a temporary miniature predictive maintenance CSV dataset."""
    df_data = {
        "UDI": range(1, 11),
        "Product ID": [f"L4718{i}" for i in range(10)],
        "Type": ["L", "M", "H", "L", "L", "M", "L", "H", "L", "M"],
        "Air temperature [K]": [298.1, 298.2, 298.1, 298.2, 298.3, 298.5, 298.6, 298.8, 299.0, 299.1],
        "Process temperature [K]": [308.6, 308.7, 308.5, 308.6, 308.7, 308.9, 309.0, 309.1, 309.2, 309.3],
        "Rotational speed [rpm]": [1551, 1408, 1498, 1433, 1408, 1425, 1558, 1527, 1600, 1450],
        "Torque [Nm]": [42.8, 46.3, 49.4, 39.5, 40.0, 41.2, 38.0, 45.0, 35.0, 48.0],
        "Tool wear [min]": [0, 3, 5, 7, 9, 12, 15, 18, 20, 24],
        "Machine failure": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        "TWF": [0]*10, "HDF": [0]*10, "PWF": [0]*10, "OSF": [0]*10, "RNF": [0]*10
    }
    df = pd.DataFrame(df_data)
    temp_file = tmp_path / "test_ai4i2020.csv"
    df.to_csv(temp_file, index=False)
    return temp_file
