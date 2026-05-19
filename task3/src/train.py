import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, N_ESTIMATORS, N_SPLITS

def train_base_models(X_train_lr, X_train_rf, y_train):
    """
    Trains base Logistic Regression (on scaled data) and Random Forest (on raw data)
    using class_weight='balanced'.
    """
    print("\n--- Training Models with class_weight='balanced' ---")
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_lr, y_train)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf.fit(X_train_rf, y_train)
    
    return lr, rf

def train_rf_with_smote(X_train_res, y_train_res):
    """
    Trains a Random Forest classifier on the SMOTE-resampled training data.
    """
    print("Training Random Forest Classifier on SMOTE training data...")
    rf_smote = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf_smote.fit(X_train_res, y_train_res)
    return rf_smote

def perform_cross_validation(X, y, columns_to_scale: list):
    """
    Performs robust 5-Fold Stratified Cross Validation on both models.
    Handles scaling inside the CV folds for Logistic Regression to prevent data leakage.
    """
    print("\n--- Performing 5-Fold Stratified Cross-Validation ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = ['f1', 'roc_auc', 'precision', 'recall']
    
    lr_cv_results = {s: [] for s in scorers}
    rf_cv_results = {s: [] for s in scorers}
    
    # Loop manually over folds to ensure scaling is done purely on fold-train sets
    # and applied to fold-test sets to prevent any preprocessing leakage!
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale for LR
        scaler = StandardScaler()
        X_tr_scaled = X_tr.copy()
        X_val_scaled = X_val.copy()
        
        X_tr_scaled[columns_to_scale] = scaler.fit_transform(X_tr[columns_to_scale])
        X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
        
        # 1. Logistic Regression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_tr_scaled, y_tr)
        
        # Evaluate LR
        lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
        lr_preds = lr.predict(X_val_scaled)
        
        # 2. Random Forest
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        rf.fit(X_tr, y_tr)
        
        # Evaluate RF
        rf_probs = rf.predict_proba(X_val)[:, 1]
        rf_preds = rf.predict(X_val)
        
        # Calculate and store fold scores
        from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
        
        lr_cv_results['f1'].append(f1_score(y_val, lr_preds))
        lr_cv_results['roc_auc'].append(roc_auc_score(y_val, lr_probs))
        lr_cv_results['precision'].append(precision_score(y_val, lr_preds))
        lr_cv_results['recall'].append(recall_score(y_val, lr_preds))
        
        rf_cv_results['f1'].append(f1_score(y_val, rf_preds))
        rf_cv_results['roc_auc'].append(roc_auc_score(y_val, rf_probs))
        rf_cv_results['precision'].append(precision_score(y_val, rf_preds))
        rf_cv_results['recall'].append(recall_score(y_val, rf_preds))
        
    return lr_cv_results, rf_cv_results
