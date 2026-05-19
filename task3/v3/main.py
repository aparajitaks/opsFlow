import sys
import os
import numpy as np
import joblib
import mlflow

# Append the current directory to Python path so relative imports from src work smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import (
    load_dataset,
    engineer_features,
    prepare_data_pipeline,
    apply_scaling,
    apply_smote
)
from src.train import (
    perform_cross_validation,
    tune_logistic_regression,
    tune_random_forest
)
from src.evaluate import (
    compare_recall_strategies,
    plot_feature_importances,
    plot_precision_recall_curves,
    load_and_predict
)
from src.explainability import explain_model
from src.config import RANDOM_STATE

def run_v3_pipeline():
    print("=================================================================")
    print("      TASK 3 — EQUIPMENT FAILURE PREDICTION: V3 PIPELINE        ")
    print("=================================================================")
    
    # 1. Setup MLflow Tracking
    mlflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "mlflow"))
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("Predictive_Maintenance_v3")
    
    # Start MLflow run wrapping the entire pipeline training
    with mlflow.start_run(run_name="Predictive_Maintenance_v3_Orchestration") as run:
        run_id = run.info.run_id
        print(f"MLflow Active Run ID: {run_id}")
        
        # 2. Data Ingestion & Engineering
        raw_df = load_dataset()
        engineered_df = engineer_features(raw_df)
        
        # 3. Data Splits
        X_train, X_test, y_train, y_test = prepare_data_pipeline(engineered_df)
        
        continuous_cols = [
            'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'temp_diff', 'power', 'wear_torque_ratio'
        ]
        
        # Scale for LR, preserve raw for RF
        X_train_lr, X_test_lr, lr_scaler = apply_scaling(X_train, X_test, continuous_cols)
        X_train_rf, X_test_rf = X_train, X_test
        
        # Save StandardScaler for LR
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs", "models"), exist_ok=True)
        scaler_path = os.path.join(os.path.dirname(__file__), "outputs", "models", "scaler_v3.pkl")
        joblib.dump(lr_scaler, scaler_path)
        print(f"Saved StandardScaler to: {scaler_path}")
        
        # 4. Class Imbalance SMOTE (For SMOTE analysis vs class_weight)
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        
        # 5. Baseline Stratified 5-Fold CV
        lr_cv, rf_cv = perform_cross_validation(X_train, y_train, continuous_cols)
        
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
            output_dir=os.path.join(os.path.dirname(__file__), "outputs", "plots")
        )
        
        # 9. Precision-Recall Curves Plotting and AP Scoring
        lr_ap, rf_ap = plot_precision_recall_curves(
            best_lr_model, best_rf_model, 
            X_test_lr, X_test_rf, 
            y_test, 
            output_dir=os.path.join(os.path.dirname(__file__), "outputs", "plots")
        )
        
        # 10. SHAP Explainability Summaries
        explain_model(
            best_rf_model, X_test_rf, 
            feature_names=X_train.columns.tolist(), 
            y_test=y_test,
            output_dir=os.path.join(os.path.dirname(__file__), "outputs", "plots")
        )
        
        # 11. Serialize Best Models
        lr_model_path = os.path.join(os.path.dirname(__file__), "outputs", "models", "logistic_regression_v3.pkl")
        rf_model_path = os.path.join(os.path.dirname(__file__), "outputs", "models", "random_forest_v3.pkl")
        
        joblib.dump(best_lr_model, lr_model_path)
        joblib.dump(best_rf_model, rf_model_path)
        print(f"\nSaved best tuned models to outputs/models/:")
        print(f" - {lr_model_path}")
        print(f" - {rf_model_path}")
        
        # Create model_summary.json detailing Task 3 training metrics
        import json
        import datetime
        
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
        
        summary_path = os.path.join(os.path.dirname(__file__), "outputs", "model_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump(summary_data, sf, indent=2)
        print(f"Saved model summary JSON to: {summary_path}")
        
        # 12. MLflow Logs
        print("\nLogging parameters, metrics, and tags to MLflow...")
        
        # Baseline CV and Tuned metrics
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
            
            "rf_baseline_f1_mean": np.mean(rf_cv['f1']),
            "rf_baseline_roc_auc_mean": np.mean(rf_cv['roc_auc']),
            "rf_baseline_precision_mean": np.mean(rf_cv['precision']),
            "rf_baseline_recall_mean": np.mean(rf_cv['recall']),
            "rf_tuned_f1_score": best_rf_score,
            "rf_test_ap_score": rf_ap
        })
        
        mlflow.set_tags({
            "model_type": "LogisticRegression & RandomForest",
            "dataset": "AI4I 2020 Predictive Maintenance",
            "imbalance_strategy": "balanced_weights_and_smote"
        })
        
    print("\n=================================================================")
    print("      V3 PIPELINE TUNING & RUN COMPLETED SUCCESSFULLY            ")
    print("=================================================================")
    print(f"MLflow Run ID: {run_id}")
    print("To view local experiment runs in the UI, execute: mlflow ui")
    print("=================================================================")
    
    # 13. Step 4 Serialization Demo
    print("\n--- Step 4 Serialization Prediction Demo ---")
    sample_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50
    }
    print(f"Sample Input Dictionary: {sample_input}")
    
    # Test loading and predicting on the sample input
    pred_class, pred_prob = load_and_predict(rf_model_path, None, sample_input)
    print(f"\n[Random Forest Prediction Result]")
    print(f"Predicted Class:            {pred_class} ({'Failure' if pred_class == 1 else 'No Failure'})")
    print(f"Failure State Probability:  {pred_prob:.4f}")
    
    # Test scaled Logistic Regression prediction demo
    pred_class_lr, pred_prob_lr = load_and_predict(lr_model_path, scaler_path, sample_input)
    print(f"\n[Logistic Regression Prediction Result]")
    print(f"Predicted Class:            {pred_class_lr} ({'Failure' if pred_class_lr == 1 else 'No Failure'})")
    print(f"Failure State Probability:  {pred_prob_lr:.4f}")

if __name__ == "__main__":
    run_v3_pipeline()
