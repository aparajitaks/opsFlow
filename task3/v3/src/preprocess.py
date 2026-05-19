import os
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import TEST_SIZE, RANDOM_STATE

def load_dataset() -> pd.DataFrame:
    """
    Loads the AI4I 2020 Predictive Maintenance dataset from local CSV
    or falls back to KaggleHub download and saves a local copy.
    """
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "ai4i2020.csv")
    
    if os.path.exists(local_path):
        print(f"Loading local dataset from: {local_path} ...")
        return pd.read_csv(local_path)
        
    print("Local CSV not found. Loading dataset via KaggleHub...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"Saved local copy to: {local_path}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning and richer feature engineering:
    1. Removes identifiers (UDI, Product ID).
    2. Encodes 'Type' using LabelEncoder.
    3. Adds three engineered columns:
       - temp_diff: Difference between process and air temperature.
       - power: Mechanical power approximation (Torque * Rotational speed).
       - wear_torque_ratio: Tool wear adjusted by torque stress.
    """
    processed_df = df.copy()
    processed_df = processed_df.drop(columns=['UDI', 'Product ID'])
    
    le = LabelEncoder()
    processed_df['Type'] = le.fit_transform(processed_df['Type'])
    
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
    and splits features/targets.
    """
    y = df['Machine failure']
    leakage_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=leakage_cols)
    
    # 80/20 Train-Test split stratified by target class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def apply_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame, columns_to_scale: list):
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
