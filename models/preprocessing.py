import os
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from core.config import settings
from core.constants import TEST_SIZE, RANDOM_STATE, CONTINUOUS_COLS

def load_dataset() -> pd.DataFrame:
    """
    Loads the AI4I 2020 Predictive Maintenance dataset from local CSV
    or falls back to KaggleHub download and saves a local copy.
    """
    local_path = settings.DATASET_PATH
    
    if local_path.exists():
        print(f"Loading local dataset from: {local_path} ...")
        return pd.read_csv(local_path)
        
    print("Local CSV not found. Loading dataset via KaggleHub...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"Saved local copy to: {local_path}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning and feature engineering:
    1. Removes identifiers (UDI, Product ID).
    2. Encodes 'Type' using a mapping: H=0, L=1, M=2.
    3. Adds three engineered columns:
       - temp_diff: Difference between process and air temperature.
       - power: Mechanical power approximation (Torque * Rotational speed).
       - wear_torque_ratio: Tool wear adjusted by torque stress.
    """
    processed_df = df.copy()
    
    # Drop identifiers if present
    cols_to_drop = [c for c in ['UDI', 'Product ID'] if c in processed_df.columns]
    processed_df = processed_df.drop(columns=cols_to_drop)
    
    # Encode Type using Type Mapping (H=0, L=1, M=2)
    type_map = {"H": 0, "L": 1, "M": 2}
    if 'Type' in processed_df.columns:
        # Check if type is already numeric
        if processed_df['Type'].dtype == object:
            processed_df['Type'] = processed_df['Type'].map(type_map).fillna(1).astype(int)
            
    # Feature 1: Temperature Difference
    processed_df['temp_diff'] = (
        processed_df['Process temperature [K]'] - processed_df['Air temperature [K]']
    )
    
    # Feature 2: Mechanical Power
    processed_df['power'] = (
        processed_df['Torque [Nm]'] * processed_df['Rotational speed [rpm]']
    )
    
    # Feature 3: Tool Wear to Torque Ratio
    processed_df['wear_torque_ratio'] = (
        processed_df['Tool wear [min]'] / (processed_df['Torque [Nm]'] + 1)
    )
    
    return processed_df

def prepare_data_pipeline(df: pd.DataFrame):
    """
    Prepares training and testing splits. Drops sub-failure columns (leakage)
    and splits features/targets. Enforces stratified splitting.
    """
    if 'Machine failure' not in df.columns:
        raise ValueError("Machine failure target column missing from dataset.")
        
    y = df['Machine failure']
    
    # Twf, Hdf, Pwf, Osf, Rnf are subset failure flags that represent data leakage
    leakage_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    drop_cols = [c for c in leakage_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    # 80/20 Train-Test split stratified by target class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def apply_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame, columns_to_scale: list = CONTINUOUS_COLS):
    """
    Applies StandardScaler to specific continuous columns (used for Logistic Regression).
    Also returns the scaler so it can be saved/serialized!
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scaler = StandardScaler()
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Applies SMOTE to balance the training set classes.
    """
    print("\n--- Applying SMOTE on Training Set ---")
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
    return X_train_res, y_train_res
