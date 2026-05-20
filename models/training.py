import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from core.constants import RANDOM_STATE, N_SPLITS, CONTINUOUS_COLS

def perform_cross_validation(X: pd.DataFrame, y: pd.Series, columns_to_scale: list = CONTINUOUS_COLS):
    """
    Performs baseline Stratified 5-Fold Cross Validation.
    Scaling is executed within folds to avoid target leakage.
    Returns baseline metrics dictionary.
    """
    print("\n--- Performing 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = ['f1', 'roc_auc', 'precision', 'recall']
    
    lr_cv_results = {s: [] for s in scorers}
    rf_cv_results = {s: [] for s in scorers}
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale continuous features inside the fold
        scaler = StandardScaler()
        X_tr_scaled = X_tr.copy()
        X_val_scaled = X_val.copy()
        X_tr_scaled[columns_to_scale] = scaler.fit_transform(X_tr[columns_to_scale])
        X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
        
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
        
        # Accumulate metrics
        lr_cv_results['f1'].append(f1_score(y_val, lr_preds))
        lr_cv_results['roc_auc'].append(roc_auc_score(y_val, lr_probs))
        lr_cv_results['precision'].append(precision_score(y_val, lr_preds, zero_division=0))
        lr_cv_results['recall'].append(recall_score(y_val, lr_preds))
        
        rf_cv_results['f1'].append(f1_score(y_val, rf_preds))
        rf_cv_results['roc_auc'].append(roc_auc_score(y_val, rf_probs))
        rf_cv_results['precision'].append(precision_score(y_val, rf_preds, zero_division=0))
        rf_cv_results['recall'].append(recall_score(y_val, rf_preds))
        
    return lr_cv_results, rf_cv_results

def tune_logistic_regression(X_train_scaled: pd.DataFrame, y_train: pd.Series):
    """
    Performs grid search hyperparameter tuning for Logistic Regression.
    """
    print("\n--- Tuning Logistic Regression with GridSearchCV ---")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    print(f"Logistic Regression Best Parameters: {grid.best_params_}")
    print(f"Logistic Regression Best CV F1 Score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Performs grid search hyperparameter tuning for Random Forest Classifier.
    """
    print("\n--- Tuning Random Forest with GridSearchCV ---")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 5]
    }
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Random Forest Best Parameters: {grid.best_params_}")
    print(f"Random Forest Best CV F1 Score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_
