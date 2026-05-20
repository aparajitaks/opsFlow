import sys
import os
import numpy as np
import pandas as pd
import joblib
import json
import datetime
import mlflow
from core.config import settings
from core.constants import RANDOM_STATE, CONTINUOUS_COLS

from models.preprocessing import (
    load_dataset,
    engineer_features,
    prepare_data_pipeline,
    apply_scaling,
    apply_smote
)
from models.training import (
    perform_cross_validation,
    tune_logistic_regression,
    tune_random_forest
)
from models.evaluation import (
    compare_recall_strategies,
    plot_feature_importances,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    load_and_predict
)
from models.explainability import explain_model

def run_ml_pipeline():
    """
    Full end-to-end Machine Learning training, tuning, evaluation, 
    explainability plotting, and MLOps serialization pipeline.
    """
    print("=================================================================")
    print("      TASK 3 — EQUIPMENT FAILURE PREDICTION: FULL PIPELINE      ")
    print("=================================================================")
    
    # 1. Setup MLflow Tracking Urilizing settings
    print(f"[MLOps] Setting MLflow tracking URI to: {settings.MLFLOW_TRACKING_URI}")
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Predictive_Maintenance_MLOps")
    except Exception as e:
        print(f"[Warning] Failed to configure MLflow tracking: {e}. Continuing without MLflow.")

    # Create target directories
    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Start MLflow run wrapping the entire pipeline training
    run_context = None
    try:
        run_context = mlflow.start_run(run_name="Predictive_Maintenance_Orchestration")
        run_id = run_context.info.run_id
        print(f"MLflow Active Run ID: {run_id}")
    except Exception as e:
        run_id = None
        print(f"[Warning] Failed to start MLflow run: {e}")

    try:
        # 2. Data Ingestion & Engineering
        raw_df = load_dataset()
        print(f"Dataset loaded with {len(raw_df)} samples.")
        engineered_df = engineer_features(raw_df)
        print("Feature engineering completed successfully.")
        
        # 3. Data Splits
        X_train, X_test, y_train, y_test = prepare_data_pipeline(engineered_df)
        print(f"Data splits prepared. Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale for LR, preserve raw for RF
        X_train_lr, X_test_lr, lr_scaler = apply_scaling(X_train, X_test, CONTINUOUS_COLS)
        X_train_rf, X_test_rf = X_train, X_test
        
        # Save StandardScaler for LR
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        joblib.dump(lr_scaler, scaler_path)
        print(f"Saved StandardScaler to: {scaler_path}")
        
        # 4. Class Imbalance SMOTE
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        
        # 5. Baseline Stratified 5-Fold CV
        lr_cv, rf_cv = perform_cross_validation(X_train, y_train, CONTINUOUS_COLS)
        
        # 6. Hyperparameter Tuning with GridSearchCV
        best_lr_model, best_lr_params, best_lr_score = tune_logistic_regression(X_train_lr, y_train)
        best_rf_model, best_rf_params, best_rf_score = tune_random_forest(X_train_rf, y_train)
        
        # Train a SMOTE-resampled Random Forest model with best RF tuned parameters
        print("\nTraining SMOTE Random Forest with Best Hyperparameters...")
        best_rf_smote = best_rf_model.__class__(**best_rf_params, random_state=RANDOM_STATE)
        best_rf_smote.fit(X_train_res, y_train_res)
        
        # 7. Compare Class 1 Recall Strategies
        compare_recall_strategies(best_rf_model, best_rf_smote, X_test_rf, y_test)
        
        # 8. Feature Importances Plotting
        plot_feature_importances(
            best_rf_model, best_lr_model, 
            feature_names=X_train.columns.tolist(), 
            output_dir=plots_dir
        )
        
        # 9. Precision-Recall Curves Plotting and AP Scoring
        lr_ap, rf_ap = plot_precision_recall_curves(
            best_lr_model, best_rf_model, 
            X_test_lr, X_test_rf, 
            y_test, 
            output_dir=plots_dir
        )
        
        # 10. ROC Curves Plotting and AUC Scoring
        lr_auc, rf_auc = plot_roc_curves(
            best_lr_model, best_rf_model, 
            X_test_lr, X_test_rf, 
            y_test, 
            output_dir=plots_dir
        )
        
        # 11. Confusion Matrices Plotting
        plot_confusion_matrices(
            best_lr_model, best_rf_model, 
            X_test_lr, X_test_rf, 
            y_test, 
            output_dir=plots_dir
        )
        
        # 12. SHAP Explainability Summaries
        explain_model(
            best_rf_model, X_test_rf, 
            feature_names=X_train.columns.tolist(), 
            y_test=y_test,
            output_dir=plots_dir
        )
        
        # 13. Serialize Best Models
        lr_model_path = os.path.join(artifacts_dir, "logistic_regression.pkl")
        rf_model_path = os.path.join(artifacts_dir, "random_forest.pkl")
        
        joblib.dump(best_lr_model, lr_model_path)
        joblib.dump(best_rf_model, rf_model_path)
        print(f"\nSaved best tuned models to artifacts directory:")
        print(f" - {lr_model_path}")
        print(f" - {rf_model_path}")
        
        # Create model_summary.json detailing Task 3 training metrics
        feature_names = X_train.columns.tolist()
        importances = best_rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [feature_names[idx] for idx in indices[:3]]
        
        # Calculate machine failure rate in raw dataset
        failure_rate = float(raw_df['Machine failure'].mean())
        
        summary_data = {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "best_model": "Random Forest",
            "best_f1": round(float(best_rf_score), 4),
            "best_roc_auc": round(float(np.mean(rf_cv['roc_auc'])), 4),
            "best_params": {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_rf_params.items()},
            "top_features": top_features,
            "failure_rate_in_dataset": round(failure_rate, 4)
        }
        
        summary_path = os.path.join(artifacts_dir, "model_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump(summary_data, sf, indent=2)
        print(f"Saved model summary JSON to: {summary_path}")
        
        # 14. MLflow Logging if run active
        if run_id is not None:
            print("\nLogging parameters, metrics, and tags to MLflow...")
            try:
                mlflow.log_params({
                    "lr_model_name": "Logistic Regression",
                    "lr_class_weight": "balanced",
                    "lr_random_state": RANDOM_STATE,
                    "lr_max_iter": 1000,
                    "lr_best_C": best_lr_params.get("C"),
                    "lr_best_solver": best_lr_params.get("solver"),
                    
                    "rf_model_name": "Random Forest Classifier",
                    "rf_class_weight": "balanced",
                    "rf_random_state": RANDOM_STATE,
                    "rf_best_n_estimators": best_rf_params.get("n_estimators"),
                    "rf_best_max_depth": best_rf_params.get("max_depth"),
                    "rf_best_min_samples_leaf": best_rf_params.get("min_samples_leaf")
                })
                
                mlflow.log_metrics({
                    "lr_baseline_f1_mean": np.mean(lr_cv['f1']),
                    "lr_baseline_roc_auc_mean": np.mean(lr_cv['roc_auc']),
                    "lr_baseline_precision_mean": np.mean(lr_cv['precision']),
                    "lr_baseline_recall_mean": np.mean(lr_cv['recall']),
                    "lr_tuned_f1_score": best_lr_score,
                    "lr_test_ap_score": lr_ap,
                    "lr_test_roc_auc_score": lr_auc,
                    
                    "rf_baseline_f1_mean": np.mean(rf_cv['f1']),
                    "rf_baseline_roc_auc_mean": np.mean(rf_cv['roc_auc']),
                    "rf_baseline_precision_mean": np.mean(rf_cv['precision']),
                    "rf_baseline_recall_mean": np.mean(rf_cv['recall']),
                    "rf_tuned_f1_score": best_rf_score,
                    "rf_test_ap_score": rf_ap,
                    "rf_test_roc_auc_score": rf_auc
                })
                
                mlflow.set_tags({
                    "model_type": "LogisticRegression & RandomForest",
                    "dataset": "AI4I 2020 Predictive Maintenance",
                    "imbalance_strategy": "balanced_weights_and_smote"
                })
            except Exception as e:
                print(f"[Warning] Failed to log details to MLflow: {e}")
                
        print("\n=================================================================")
        print("      TASK 3 PIPELINE COMPLETED SUCCESSFULLY                    ")
        print("=================================================================")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] ML pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if run_context is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

if __name__ == "__main__":
    run_ml_pipeline()
