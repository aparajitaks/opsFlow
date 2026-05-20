import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import numpy as np
import os
import pandas as pd
from core.config import settings

def explain_model(model, X_test: pd.DataFrame, feature_names: list, y_test: pd.Series, output_dir=None):
    """
    Produces SHAP beeswarm and single instance force plot for the Random Forest model.
    Saves plots as PNGs under models/artifacts/plots/.
    """
    if output_dir is None:
        output_dir = os.path.join(settings.MODEL_ARTIFACTS_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Generating SHAP Explainability Plots ---")
    
    try:
        # Initialize TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Align output formatting for list or single array variations
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            if len(shap_values.shape) == 3:
                shap_values_class1 = shap_values[:, :, 1]
            else:
                shap_values_class1 = shap_values
                
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value_class1 = base_value[1]
        else:
            base_value_class1 = base_value
            
        # Plot 1: Summary / Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Beeswarm Plot (Model failure drivers)", fontsize=14, pad=15)
        
        beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP Beeswarm plot to: {beeswarm_path}")
        
        # Plot 2: Force Plot for a Correctly Predicted Failure Instance
        preds = model.predict(X_test)
        correct_failure_indices = np.where((y_test.values == 1) & (preds == 1))[0]
        
        if len(correct_failure_indices) > 0:
            instance_idx = correct_failure_indices[0]
            print(f"Selected failure instance at test set index: {instance_idx} for force plot.")
            
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                base_value_class1, 
                shap_values_class1[instance_idx], 
                X_test.iloc[instance_idx], 
                matplotlib=True, 
                show=False
            )
            plt.title(f"SHAP Force Plot - Prediction Failure Drivers (Instance {instance_idx})", fontsize=12, pad=20)
            
            force_path = os.path.join(output_dir, "shap_force_plot.png")
            plt.savefig(force_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP Force plot to: {force_path}")
        else:
            print("Warning: No correctly predicted failure instances were found in the test set. Skipping force plot.")
            
    except Exception as e:
        print(f"Error generating SHAP explainability plots: {e}")
