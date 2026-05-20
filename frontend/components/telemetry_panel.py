import os
import streamlit as st
from frontend.state import predict_ml, get_model_status
from core.config import settings

def render_telemetry_panel():
    """
    Renders standard telemetry inputs, predictions, model status,
    and all offline/online evaluation visualizations (ROC-AUC, Confusion Matrices, SHAP).
    """
    st.write("Adjust telemetry sliders below to simulate live equipment sensors and trigger predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eq_type = st.selectbox("Equipment Type", options=["L", "M", "H"], index=0, help="L = Low quality (60%), M = Medium (30%), H = High quality (10%)")
        air_temp = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        proc_temp = st.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
        
    with col2:
        speed = st.slider("Rotational Speed [rpm]", min_value=1000.0, max_value=3000.0, value=1500.0, step=10.0)
        torque = st.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
        wear = st.slider("Tool Wear [min]", min_value=0.0, max_value=260.0, value=100.0, step=1.0)

    model_type = st.selectbox(
        "Predictive Classification Model",
        options=[("random_forest", "Random Forest Classifier (Tuned)"), 
                 ("logistic_regression", "Logistic Regression (Tuned)")],
        format_func=lambda x: x[1]
    )[0]

    st.markdown("---")
    
    # Analyze trigger button
    if st.button("🔌 Analyze Machine State", use_container_width=True):
        payload = {
            "Type": eq_type,
            "Air_temperature": air_temp,
            "Process_temperature": proc_temp,
            "Rotational_speed": speed,
            "Torque": torque,
            "Tool_wear": wear
        }
        
        with st.spinner("Executing model prediction..."):
            res = predict_ml(payload, model_type)
            if res:
                pred = res["prediction"]
                prob = res["probability"]
                eng = res["engineered_features"]
                
                # Render results columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("#### Diagnosis Status")
                    if pred == 1:
                        st.error("🚨 EQUIPMENT FAILURE DETECTED")
                    else:
                        st.success("🟢 STABLE OPERATIONS VERIFIED")
                        
                with res_col2:
                    st.markdown("#### Probability Analysis")
                    st.metric("Failure Likelihood", f"{prob * 100:.1f}%")
                    st.progress(prob)
                    
                # Render calculated engineered features
                st.markdown("#### Engineered Attributes (Engineered on the fly)")
                e_col1, e_col2, e_col3 = st.columns(3)
                e_col1.metric("Temp Difference (ΔT)", f"{eng['temp_diff']:.2f} K")
                e_col2.metric("Calculated Power", f"{eng['power']:.1f} W")
                e_col3.metric("Wear-Torque Ratio", f"{eng['wear_torque_ratio']:.4f}")

    # Display best model summary parameters
    st.markdown("---")
    st.markdown("### 🏆 Trained ML Model Parameters & CV Performance")
    status = get_model_status()
    if status and status.get("run_timestamp"):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Best Model F1-Score", f"{status.get('best_f1'):.4f}")
        col_m2.metric("Mean CV ROC-AUC", f"{status.get('best_roc_auc'):.4f}")
        col_m3.metric("Failure Rate (Baseline)", f"{status.get('failure_rate_in_dataset') * 100:.2f}%")
        st.markdown(f"**Primary Failure Indicators:** `{', '.join(status.get('top_features', []))}`")
        st.markdown(f"**Best Parameters:** `{status.get('best_params')}`")
    else:
        st.info("No active model training metadata found.")

    # Ingest and display Matplotlib model visualizations (ROC, Confusion Matrix, SHAP)
    st.markdown("---")
    st.markdown("### 📈 Evaluation Plots & Feature Importance")
    
    plots_tab1, plots_tab2, plots_tab3 = st.tabs(["Evaluation Curves", "SHAP Explainability", "Confusion Matrices"])
    plots_dir = os.path.join(settings.MODEL_ARTIFACTS_DIR, "plots")
    
    with plots_tab1:
        col_pr, col_roc = st.columns(2)
        pr_path = os.path.join(plots_dir, "precision_recall_comparison.png")
        if os.path.exists(pr_path):
            col_pr.image(pr_path, caption="Precision-Recall Curves (Tuned Models)", use_container_width=True)
        else:
            col_pr.info("Precision-Recall chart not generated yet.")
            
        roc_path = os.path.join(plots_dir, "roc_comparison.png")
        if os.path.exists(roc_path):
            col_roc.image(roc_path, caption="ROC Curves (Tuned Models)", use_container_width=True)
        else:
            col_roc.info("ROC comparison chart not generated yet.")
            
    with plots_tab2:
        col_sh1, col_sh2 = st.columns(2)
        beeswarm_path = os.path.join(plots_dir, "shap_beeswarm.png")
        if os.path.exists(beeswarm_path):
            col_sh1.image(beeswarm_path, caption="SHAP Beeswarm Plot (Global Feature Drivers)", use_container_width=True)
        else:
            col_sh1.info("SHAP beeswarm chart not generated yet.")
            
        force_path = os.path.join(plots_dir, "shap_force_plot.png")
        if os.path.exists(force_path):
            col_sh2.image(force_path, caption="SHAP Force Plot (Single Failure Instance Analysis)", use_container_width=True)
        else:
            col_sh2.info("SHAP force chart not generated yet.")
            
    with plots_tab3:
        col_cm1, col_cm2 = st.columns(2)
        rf_cm_path = os.path.join(plots_dir, "rf_confusion_matrix.png")
        if os.path.exists(rf_cm_path):
            col_cm1.image(rf_cm_path, caption="Random Forest Confusion Matrix", use_container_width=True)
        else:
            col_cm1.info("Random Forest confusion matrix not generated yet.")
            
        lr_cm_path = os.path.join(plots_dir, "lr_confusion_matrix.png")
        if os.path.exists(lr_cm_path):
            col_cm2.image(lr_cm_path, caption="Logistic Regression Confusion Matrix", use_container_width=True)
        else:
            col_cm2.info("Logistic Regression confusion matrix not generated yet.")
