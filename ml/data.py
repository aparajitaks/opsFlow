"""Dataset loading for Task 3."""
import pandas as pd

from core.config import settings
from core.data_validation import validate_dataset
from core.exceptions import DatasetValidationError
from core.logger import get_logger

log = get_logger("ml.data")


def load_dataset() -> pd.DataFrame:
    """Load and validate the AI4I predictive maintenance CSV."""
    local_path = settings.DATASET_PATH
    if not local_path.exists():
        raise DatasetValidationError(
            f"Dataset not found at '{local_path}'. Ensure data/ai4i2020.csv exists."
        )
    log.info(f"Loading dataset from: {local_path}")
    df = pd.read_csv(local_path)
    df = validate_dataset(df)
    log.info(f"Dataset loaded — shape: {df.shape}")
    return df
