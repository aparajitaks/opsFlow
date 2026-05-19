import sys
import os

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
    train_base_models,
    train_rf_with_smote,
    perform_cross_validation
)
from src.evaluate import (
    compare_recall_strategies,
    plot_feature_importances,
    plot_precision_recall_curves
)

def run_v2_pipeline():
    print("=================================================================")
    print("      TASK 3 — EQUIPMENT FAILURE PREDICTION: V2 PIPELINE        ")
    print("=================================================================")
    
    # 1. Load the dataset
    raw_df = load_dataset()
    
    # 2. Step 1: Feature Engineering
    print("\n--- Step 1: Richer Feature Engineering ---")
    engineered_df = engineer_features(raw_df)
    
    # 3. Split the dataset (preventing target leakage)
    X_train, X_test, y_train, y_test = prepare_data_pipeline(engineered_df)
    
    # Identify continuous columns for scaling (all except Type category)
    continuous_cols = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'temp_diff', 'power', 'wear_torque_ratio'
    ]
    
    # Scale continuous features for Logistic Regression ONLY
    X_train_lr, X_test_lr = apply_scaling(X_train, X_test, continuous_cols)
    
    # Random Forest uses raw continuous features
    X_train_rf, X_test_rf = X_train, X_test
    
    # 4. Step 2: Class Imbalance Handling (SMOTE)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 5. Step 3: Stratified 5-Fold Cross Validation
    lr_cv, rf_cv = perform_cross_validation(X_train, y_train, continuous_cols)
    
    print("\nStratified 5-Fold CV Mean Results (± Std Deviation):")
    for model_name, cv_res in [("Logistic Regression", lr_cv), ("Random Forest", rf_cv)]:
        print(f"\n{model_name}:")
        for scorer in ['roc_auc', 'f1', 'precision', 'recall']:
            mean_score = np.mean(cv_res[scorer])
            std_score = np.std(cv_res[scorer])
            print(f"  {scorer.upper():<10} = {mean_score:.4f} ± {std_score:.4f}")
            
    # 6. Train final estimators
    lr_model, rf_model = train_base_models(X_train_lr, X_train_rf, y_train)
    rf_smote_model = train_rf_with_smote(X_train_res, y_train_res)
    
    # Compare Recall on Class 1
    compare_recall_strategies(rf_model, rf_smote_model, X_test_rf, y_test)
    
    # 7. Step 4: Plot Feature Importances
    plot_feature_importances(
        rf_model, lr_model, 
        feature_names=X_train.columns.tolist(), 
        output_dir="outputs"
    )
    
    # 8. Step 5: Precision-Recall Curves
    plot_precision_recall_curves(
        lr_model, rf_model, 
        X_test_lr, X_test_rf, 
        y_test, 
        output_dir="outputs"
    )
    
    print("\n=================================================================")
    print("      V2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY               ")
    print("=================================================================")

if __name__ == "__main__":
    import numpy as np
    run_v2_pipeline()
