import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, PrecisionRecallDisplay, average_precision_score

def compare_recall_strategies(rf_balanced, rf_smote, X_test, y_test):
    """
    Compares Class 1 recall between class_weight='balanced' and SMOTE resampled Random Forest.
    """
    print("\n--- Comparing Recall on Class 1 (Failures) ---")
    preds_balanced = rf_balanced.predict(X_test)
    recall_balanced = recall_score(y_test, preds_balanced)
    
    preds_smote = rf_smote.predict(X_test)
    recall_smote = recall_score(y_test, preds_smote)
    
    print(f"Tuned Random Forest Recall:               {recall_balanced:.4f}")
    print(f"SMOTE-Resampled Random Forest Recall:     {recall_smote:.4f}")
    return recall_balanced, recall_smote

def plot_feature_importances(rf, lr, feature_names, output_dir="outputs/plots"):
    """
    Plots and saves feature importance for Random Forest and Logistic Regression coefficients.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Random Forest Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest (Tuned) - Feature Importance (v3)", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color="steelblue", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Importance Score", fontsize=11)
    
    rf_img_path = os.path.join(output_dir, "rf_feature_importances.png")
    plt.savefig(rf_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved tuned Random Forest feature importances to: {rf_img_path}")
    
    # 2. Logistic Regression Signifiance
    coefs = np.abs(lr.coef_[0])
    lr_indices = np.argsort(coefs)
    
    plt.figure(figsize=(10, 6))
    plt.title("Logistic Regression (Tuned) - Absolute Coefficients", fontsize=14)
    plt.barh(range(len(lr_indices)), coefs[lr_indices], color="salmon", align="center")
    plt.yticks(range(len(lr_indices)), [feature_names[i] for i in lr_indices], fontsize=10)
    plt.xlabel("Absolute Coefficient Value", fontsize=11)
    
    lr_img_path = os.path.join(output_dir, "lr_coefficients.png")
    plt.savefig(lr_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved tuned Logistic Regression coefficients to: {lr_img_path}")

def plot_precision_recall_curves(lr, rf, X_test_lr, X_test_rf, y_test, output_dir="outputs/plots"):
    """
    Plots PrecisionRecallDisplay for both tuned models on the test set and calculates AP scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lr_probs = lr.predict_proba(X_test_lr)[:, 1]
    rf_probs = rf.predict_proba(X_test_rf)[:, 1]
    
    lr_ap = average_precision_score(y_test, lr_probs)
    rf_ap = average_precision_score(y_test, rf_probs)
    
    print("\n--- Tuned Models Precision-Recall Evaluation ---")
    print(f"Tuned Logistic Regression AP: {lr_ap:.4f}")
    print(f"Tuned Random Forest AP:       {rf_ap:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    PrecisionRecallDisplay.from_predictions(
        y_test, lr_probs, ax=ax, name=f"Logistic Regression (AP = {lr_ap:.4f})", color="salmon"
    )
    PrecisionRecallDisplay.from_predictions(
        y_test, rf_probs, ax=ax, name=f"Random Forest (AP = {rf_ap:.4f})", color="steelblue"
    )
    
    plt.title("Precision-Recall Curves Comparison (Tuned v3 Models)", fontsize=14)
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    
    pr_img_path = os.path.join(output_dir, "precision_recall_comparison.png")
    plt.savefig(pr_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved tuned Precision-Recall curves comparison to: {pr_img_path}")
    
    return lr_ap, rf_ap

def load_and_predict(model_path: str, scaler_path: str, input_dict: dict):
    """
    Step 4: Loads saved model and scaler, preprocesses a raw input dictionary,
    and returns the predicted class and failure probability.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model path not found: {model_path}")
        
    model = joblib.load(model_path)
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    data = input_dict.copy()
    
    # 1. Map category 'Type' -> encoding: H=0, L=1, M=2
    type_map = {"H": 0, "L": 1, "M": 2}
    if "Type" in data:
        data["Type"] = type_map.get(data["Type"], 1)
        
    # 2. Add engineered features
    data["temp_diff"] = data["Process temperature [K]"] - data["Air temperature [K]"]
    data["power"] = data["Torque [Nm]"] * data["Rotational speed [rpm]"]
    data["wear_torque_ratio"] = data["Tool wear [min]"] / (data["Torque [Nm]"] + 1)
    
    # 3. Align features with exact training columns order
    feature_order = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "temp_diff", "power", "wear_torque_ratio"
    ]
    
    df = pd.DataFrame([data])[feature_order]
    
    # 4. Scale continuous features if scaler is provided
    if scaler is not None:
        continuous_cols = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "temp_diff", "power", "wear_torque_ratio"
        ]
        df[continuous_cols] = scaler.transform(df[continuous_cols])
        
    # 5. Get model prediction and probability
    pred_class = int(model.predict(df)[0])
    pred_prob = float(model.predict_proba(df)[0][1])
    
    return pred_class, pred_prob
