import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import datetime
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Ensure base dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from core.constants import RANDOM_STATE, TEST_SIZE, N_SPLITS, CONTINUOUS_COLS, TYPE_MAP

def load_dataset() -> pd.DataFrame:
    """Loads the predictive maintenance dataset from a local CSV path."""
    local_path = settings.DATASET_PATH
    if local_path.exists():
        print(f"[ML] Loading local dataset from: {local_path} ...")
        return pd.read_csv(local_path)
    raise FileNotFoundError(
        f"Local CSV dataset not found at '{local_path}'. "
        "Please ensure data/ai4i2020.csv exists."
    )

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies data engineering: encodes Type, removes ids, adds thermal delta & mechanical power."""
    processed_df = df.copy()
    
    # Drop identifiers
    cols_to_drop = [c for c in ['UDI', 'Product ID'] if c in processed_df.columns]
    processed_df = processed_df.drop(columns=cols_to_drop)
    
    # Map Type (H=0, L=1, M=2)
    if 'Type' in processed_df.columns:
        if processed_df['Type'].dtype == object:
            processed_df['Type'] = processed_df['Type'].map(TYPE_MAP).fillna(1).astype(int)
            
    # temp_diff: Difference process & air temp
    processed_df['temp_diff'] = processed_df['Process temperature [K]'] - processed_df['Air temperature [K]']
    
    # power: Rotational speed * Torque
    processed_df['power'] = processed_df['Torque [Nm]'] * processed_df['Rotational speed [rpm]']
    
    # wear_torque_ratio: Tool wear adjusted by torque stress
    processed_df['wear_torque_ratio'] = processed_df['Tool wear [min]'] / (processed_df['Torque [Nm]'] + 1)
    
    return processed_df

def prepare_data_pipeline(df: pd.DataFrame):
    """Splits target class and drops data-leakage subset failure flags (TWF, HDF, PWF, OSF, RNF)."""
    if 'Machine failure' not in df.columns:
        raise ValueError("Machine failure target column is missing from dataset.")
    
    y = df['Machine failure']
    
    # Drop target and subset leakage columns
    leakage_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    drop_cols = [c for c in leakage_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

def perform_cross_validation(X: pd.DataFrame, y: pd.Series):
    """Executes stratified 5-fold cross validation on baselines with in-fold scaling to prevent leakage."""
    print("[ML] Performing 5-Fold Stratified Cross-Validation on baseline models...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = ['f1', 'roc_auc', 'precision', 'recall']
    
    lr_cv = {s: [] for s in scorers}
    rf_cv = {s: [] for s in scorers}
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale only continuous features
        scaler = StandardScaler()
        X_tr_scaled = X_tr.copy()
        X_val_scaled = X_val.copy()
        X_tr_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_tr[CONTINUOUS_COLS])
        X_val_scaled[CONTINUOUS_COLS] = scaler.transform(X_val[CONTINUOUS_COLS])
        
        # Train baseline LR
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_tr_scaled, y_tr)
        lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
        lr_preds = lr.predict(X_val_scaled)
        
        # Train baseline RF
        rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        rf.fit(X_tr, y_tr)
        rf_probs = rf.predict_proba(X_val)[:, 1]
        rf_preds = rf.predict(X_val)
        
        # Save metrics
        lr_cv['f1'].append(f1_score(y_val, lr_preds))
        lr_cv['roc_auc'].append(roc_auc_score(y_val, lr_probs))
        lr_cv['precision'].append(precision_score(y_val, lr_preds, zero_division=0))
        lr_cv['recall'].append(recall_score(y_val, lr_preds))
        
        rf_cv['f1'].append(f1_score(y_val, rf_preds))
        rf_cv['roc_auc'].append(roc_auc_score(y_val, rf_probs))
        rf_cv['precision'].append(precision_score(y_val, rf_preds, zero_division=0))
        rf_cv['recall'].append(recall_score(y_val, rf_preds))
        
    return lr_cv, rf_cv

def tune_models(X_train: pd.DataFrame, y_train: pd.Series, X_train_lr: pd.DataFrame):
    """Tunes hyper-parameters using GridSearchCV."""
    print("[ML] Tuning Logistic Regression C and solver...")
    lr_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lr_search = GridSearchCV(estimator=lr, param_grid=lr_grid, cv=skf, scoring='f1', n_jobs=-1)
    lr_search.fit(X_train_lr, y_train)
    print(f" -> Best LR Parameters: {lr_search.best_params_} (F1 Score: {lr_search.best_score_:.4f})")
    
    print("[ML] Tuning Random Forest depth and estimators...")
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 5]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    rf_search = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=skf, scoring='f1', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    print(f" -> Best RF Parameters: {rf_search.best_params_} (F1 Score: {rf_search.best_score_:.4f})")
    
    return lr_search.best_estimator_, lr_search.best_params_, lr_search.best_score_, \
           rf_search.best_estimator_, rf_search.best_params_, rf_search.best_score_

def run_train():
    """Main execution flow for training the predictive models."""
    print("==================================================")
    print("    TASK 3 — PREDICTIVE ML PIPELINE: TRAINING     ")
    print("==================================================")
    
    # 1. Load Data
    raw_df = load_dataset()
    print(f"[ML] Dataset loaded. Shape: {raw_df.shape}")
    
    # 2. Preprocess & Feature Engineer
    df = engineer_features(raw_df)
    X_train, X_test, y_train, y_test = prepare_data_pipeline(df)
    
    # 3. Scaling for Logistic Regression
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = StandardScaler()
    X_train_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test_scaled[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])
    
    # 4. Perform CV baselines
    lr_cv, rf_cv = perform_cross_validation(X_train, y_train)
    print(f" -> Baseline LR CV F1-Score: {np.mean(lr_cv['f1']):.4f} | ROC-AUC: {np.mean(lr_cv['roc_auc']):.4f}")
    print(f" -> Baseline RF CV F1-Score: {np.mean(rf_cv['f1']):.4f} | ROC-AUC: {np.mean(rf_cv['roc_auc']):.4f}")
    
    # 5. Model Tuning
    best_lr, best_lr_params, best_lr_score, best_rf, best_rf_params, best_rf_score = tune_models(
        X_train, y_train, X_train_scaled
    )
    
    # 6. Serializing models & artifacts
    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_lr, artifacts_dir / "logistic_regression.pkl")
    joblib.dump(best_rf, artifacts_dir / "random_forest.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    
    # Determine best features by RF relative importances
    feature_names = X_train.columns.tolist()
    importances = best_rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [feature_names[i] for i in top_indices]
    
    failure_rate = float(raw_df['Machine failure'].mean())
    
    summary = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "best_model": "Random Forest" if best_rf_score >= best_lr_score else "Logistic Regression",
        "best_f1": round(float(max(best_rf_score, best_lr_score)), 4),
        "best_roc_auc": round(float(np.mean(rf_cv['roc_auc'])), 4),
        "best_params": {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_rf_params.items()},
        "top_features": top_features,
        "failure_rate_in_dataset": round(failure_rate, 4)
    }
    
    # Save training summary
    summary_path = artifacts_dir / "model_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=2)
    print(f"[ML] Training metadata summary saved to: {summary_path}")
    
    # Sync with RAG documents
    docs_dir = settings.DOCS_DIR
    shutil.copy(summary_path, docs_dir / "model_summary.json")
    print(f"[ML] Synced model summary directly to RAG corpus: {docs_dir / 'model_summary.json'}")
    
    # Save splits to temporary artifacts directory for evaluate.py to use
    joblib.dump((X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled), artifacts_dir / "data_splits.pkl")
    print("[ML] Data splits cached for evaluation script.")
    print("[ML] Training step completed successfully.")

if __name__ == "__main__":
    run_train()
