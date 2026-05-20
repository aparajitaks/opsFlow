import os
import sys
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Ensure base dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from core.constants import RANDOM_STATE, CONTINUOUS_COLS

try:
    import shap
except ImportError:
    shap = None

def run_evaluation():
    print("==================================================")
    print("    TASK 3 — PREDICTIVE ML PIPELINE: EVALUATION   ")
    print("==================================================")

    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    splits_path = artifacts_dir / "data_splits.pkl"

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Cached data splits not found at {splits_path}. "
            "Please run `python models/train.py` first to generate models and splits."
        )

    # 1. Load cached splits and models
    print("[ML] Loading cached dataset partitions and serialized models...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = joblib.load(splits_path)
    
    lr_model = joblib.load(artifacts_dir / "logistic_regression.pkl")
    rf_model = joblib.load(artifacts_dir / "random_forest.pkl")
    scaler = joblib.load(artifacts_dir / "scaler.pkl")

    # 2. Evaluate Tuned Logistic Regression
    print("[ML] Evaluating Tuned Logistic Regression...")
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        "recall": round(float(recall_score(y_test, lr_preds)), 4),
        "precision": round(float(precision_score(y_test, lr_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, lr_preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, lr_probs)), 4),
        "avg_precision": round(float(average_precision_score(y_test, lr_probs)), 4)
    }

    # 3. Evaluate Tuned Random Forest (Class Weight Balanced)
    print("[ML] Evaluating Tuned Random Forest (Balanced)...")
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        "recall": round(float(recall_score(y_test, rf_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, rf_preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, rf_probs)), 4),
        "avg_precision": round(float(average_precision_score(y_test, rf_probs)), 4)
    }

    # 4. Train and Evaluate SMOTE Random Forest for comparative analysis
    print("[ML] Training SMOTE-Oversampled Random Forest for comparison...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
    # Use best parameters found during Random Forest tuning if available
    try:
        with open(artifacts_dir / "model_summary.json", "r") as f:
            summary = json.load(f)
            best_params = summary.get("best_params", {})
            # Remove any class_weight key since SMOTE balances the data explicitly
            best_params.pop("class_weight", None)
            rf_smote.set_params(**best_params)
    except Exception:
        pass
    
    rf_smote.fit(X_train_smote, y_train_smote)
    rf_smote_preds = rf_smote.predict(X_test)
    rf_smote_probs = rf_smote.predict_proba(X_test)[:, 1]
    
    rf_smote_metrics = {
        "recall": round(float(recall_score(y_test, rf_smote_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_smote_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, rf_smote_preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, rf_smote_probs)), 4),
        "avg_precision": round(float(average_precision_score(y_test, rf_smote_probs)), 4)
    }

    # 5. Generate and save Plots
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ML] Generating performance plots in {plots_dir} ...")

    # A. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_test, lr_probs, name=f"Logistic Regression (AP = {lr_metrics['avg_precision']})", color="salmon")
    PrecisionRecallDisplay.from_predictions(y_test, rf_probs, name=f"Random Forest - Balanced (AP = {rf_metrics['avg_precision']})", color="steelblue", ax=plt.gca())
    PrecisionRecallDisplay.from_predictions(y_test, rf_smote_probs, name=f"Random Forest - SMOTE (AP = {rf_smote_metrics['avg_precision']})", color="forestgreen", ax=plt.gca())
    plt.title("Precision-Recall Curves Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plots_dir / "precision_recall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # B. ROC Curves
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, lr_probs, name=f"Logistic Regression (AUC = {lr_metrics['roc_auc']})", color="salmon")
    RocCurveDisplay.from_predictions(y_test, rf_probs, name=f"Random Forest - Balanced (AUC = {rf_metrics['roc_auc']})", color="steelblue", ax=plt.gca())
    RocCurveDisplay.from_predictions(y_test, rf_smote_probs, name=f"Random Forest - SMOTE (AUC = {rf_smote_metrics['roc_auc']})", color="forestgreen", ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("ROC Curves Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plots_dir / "roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # C. Absolute Confusion Matrices
    lr_cm = confusion_matrix(y_test, lr_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(lr_cm, display_labels=["No Failure", "Failure"]).plot(cmap="Reds", ax=axes[0])
    axes[0].set_title("Logistic Regression")
    ConfusionMatrixDisplay(rf_cm, display_labels=["No Failure", "Failure"]).plot(cmap="Blues", ax=axes[1])
    axes[1].set_title("Random Forest (Tuned)")
    plt.suptitle("Holdout Confusion Matrices Comparison", fontsize=14, y=1.02)
    plt.savefig(plots_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. SHAP Explainability
    if shap is not None:
        print("[ML] Generating SHAP explainability plots (limiting background to 50 samples for speed)...")
        try:
            # Limit sample size to 50 for rapid explainability computations
            X_test_sample = X_test.iloc[:50]
            y_test_sample = y_test.iloc[:50]
            
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test_sample)

            # Handle multi-class / list shap_values structure
            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values_class1 = shap_values[:, :, 1]
            else:
                shap_values_class1 = shap_values

            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
                base_value_class1 = base_value[1]
            else:
                base_value_class1 = base_value

            # Beeswarm Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_class1, X_test_sample, feature_names=X_test.columns.tolist(), show=False)
            plt.title("SHAP Beeswarm Plot (Failure Drivers)", fontsize=14, pad=15)
            plt.savefig(plots_dir / "shap_beeswarm.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f" -> Saved: {plots_dir / 'shap_beeswarm.png'}")

            # Force Plot for first correctly predicted failure instance
            rf_sample_preds = rf_model.predict(X_test_sample)
            correct_failures = np.where((y_test_sample.values == 1) & (rf_sample_preds == 1))[0]
            
            if len(correct_failures) > 0:
                idx = correct_failures[0]
                plt.figure(figsize=(12, 4))
                shap.force_plot(
                    base_value_class1,
                    shap_values_class1[idx],
                    X_test_sample.iloc[idx],
                    matplotlib=True,
                    show=False
                )
                plt.title(f"SHAP Force Plot - Prediction Drivers (Instance {idx})", fontsize=12, pad=20)
                plt.savefig(plots_dir / "shap_force_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f" -> Saved: {plots_dir / 'shap_force_plot.png'}")
            else:
                print("[ML] No correctly predicted failure instances in SHAP background slice; skipping Force Plot.")
        except Exception as e:
            print(f"[ML] Error generating SHAP plots: {e}")
    else:
        print("[ML] SHAP library not available; skipping SHAP beeswarm & force plots.")

    # 7. Compile and save evaluation summary
    eval_summary = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "logistic_regression": lr_metrics,
        "random_forest_balanced": rf_metrics,
        "random_forest_smote": rf_smote_metrics,
        "recall_comparison": {
            "balanced_strategy_recall": rf_metrics["recall"],
            "smote_strategy_recall": rf_smote_metrics["recall"],
            "delta_recall": round(rf_smote_metrics["recall"] - rf_metrics["recall"], 4)
        }
    }

    eval_summary_path = artifacts_dir / "evaluation_summary.json"
    with open(eval_summary_path, 'w', encoding='utf-8') as ef:
        json.dump(eval_summary, ef, indent=2)
    print(f"[ML] Evaluation report successfully compiled and saved to: {eval_summary_path}")

    # Display results beautifully
    print("\n==================================================")
    print("          EVALUATION RESULTS COMPARISON           ")
    print("==================================================")
    print(f"Metrics                    | Tuned LR | RF (Balanced) | RF (SMOTE)")
    print(f"---------------------------|----------|---------------|-----------")
    print(f"Recall                     | {lr_metrics['recall']:.4f}   | {rf_metrics['recall']:.4f}        | {rf_smote_metrics['recall']:.4f}")
    print(f"Precision                  | {lr_metrics['precision']:.4f}   | {rf_metrics['precision']:.4f}        | {rf_smote_metrics['precision']:.4f}")
    print(f"F1-Score                   | {lr_metrics['f1_score']:.4f}   | {rf_metrics['f1_score']:.4f}        | {rf_smote_metrics['f1_score']:.4f}")
    print(f"ROC-AUC                    | {lr_metrics['roc_auc']:.4f}   | {rf_metrics['roc_auc']:.4f}        | {rf_smote_metrics['roc_auc']:.4f}")
    print(f"Average Precision (AP)     | {lr_metrics['avg_precision']:.4f}   | {rf_metrics['avg_precision']:.4f}        | {rf_smote_metrics['avg_precision']:.4f}")
    print("==================================================")
    print("[ML] Evaluation run completed successfully.")

if __name__ == "__main__":
    run_evaluation()
