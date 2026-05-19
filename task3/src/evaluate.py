import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, PrecisionRecallDisplay, average_precision_score

def compare_recall_strategies(rf_balanced, rf_smote, X_test, y_test):
    """
    Compares Class 1 recall between class_weight='balanced' and SMOTE resampled Random Forest.
    """
    print("\n--- Comparing Recall on Class 1 (Failures) ---")
    
    # 1. Prediction with class_weight='balanced'
    preds_balanced = rf_balanced.predict(X_test)
    recall_balanced = recall_score(y_test, preds_balanced)
    
    # 2. Prediction with SMOTE resampled RF
    preds_smote = rf_smote.predict(X_test)
    recall_smote = recall_score(y_test, preds_smote)
    
    print(f"Random Forest (class_weight='balanced') Class 1 Recall: {recall_balanced:.4f}")
    print(f"Random Forest (SMOTE Resampled) Class 1 Recall:       {recall_smote:.4f}")
    
    if recall_smote > recall_balanced:
        print("Strategy Winner: SMOTE Resampling yielded better recall for failures.")
    elif recall_smote < recall_balanced:
        print("Strategy Winner: class_weight='balanced' yielded better recall for failures.")
    else:
        print("Strategy Winner: Both strategies yielded identical class 1 recall.")
        
    return recall_balanced, recall_smote

def plot_feature_importances(rf, lr, feature_names, output_dir="outputs"):
    """
    Plots and saves:
    1. Random Forest Feature Importances.
    2. Logistic Regression Coefficients (Absolute Values).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Plot 1: Random Forest Feature Importance
    # -------------------------------------------------------------
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest - Feature Importance (v2)", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color="steelblue", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Importance Score", fontsize=11)
    
    rf_img_path = os.path.join(output_dir, "rf_feature_importances.png")
    plt.savefig(rf_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Random Forest feature importances plot to: {rf_img_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Logistic Regression Absolute Coefficients
    # -------------------------------------------------------------
    coefs = np.abs(lr.coef_[0])
    lr_indices = np.argsort(coefs)
    
    plt.figure(figsize=(10, 6))
    plt.title("Logistic Regression - Feature Significance (Absolute Coefficients)", fontsize=14)
    plt.barh(range(len(lr_indices)), coefs[lr_indices], color="salmon", align="center")
    plt.yticks(range(len(lr_indices)), [feature_names[i] for i in lr_indices], fontsize=10)
    plt.xlabel("Absolute Coefficient Value", fontsize=11)
    
    lr_img_path = os.path.join(output_dir, "lr_coefficients.png")
    plt.savefig(lr_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Logistic Regression coefficients plot to: {lr_img_path}")

def plot_precision_recall_curves(lr, rf, X_test_lr, X_test_rf, y_test, output_dir="outputs"):
    """
    Plots PrecisionRecallDisplay for both models on held-out test data.
    Computes Average Precision (AP) scores and saves the composite plot as a PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict failure probabilities
    lr_probs = lr.predict_proba(X_test_lr)[:, 1]
    rf_probs = rf.predict_proba(X_test_rf)[:, 1]
    
    # Compute Average Precision (AP)
    lr_ap = average_precision_score(y_test, lr_probs)
    rf_ap = average_precision_score(y_test, rf_probs)
    
    print("\n--- Precision-Recall Evaluation ---")
    print(f"Logistic Regression Average Precision (AP): {lr_ap:.4f}")
    print(f"Random Forest Average Precision (AP):       {rf_ap:.4f}")
    
    # Plotting both curves on a single chart
    fig, ax = plt.subplots(figsize=(8, 7))
    
    PrecisionRecallDisplay.from_predictions(
        y_test, lr_probs, ax=ax, name=f"Logistic Regression (AP = {lr_ap:.4f})", color="salmon"
    )
    PrecisionRecallDisplay.from_predictions(
        y_test, rf_probs, ax=ax, name=f"Random Forest (AP = {rf_ap:.4f})", color="steelblue"
    )
    
    plt.title("Precision-Recall Curves Comparison (Held-out Test Set)", fontsize=14)
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    
    pr_img_path = os.path.join(output_dir, "precision_recall_comparison.png")
    plt.savefig(pr_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved composite Precision-Recall curve comparison to: {pr_img_path}")
