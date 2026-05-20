import os
import streamlit as st
from frontend.state import get_model_status, trigger_retraining, force_reindex, fetch_logs
from core.config import settings

def render_metrics_panel():
    """
    Renders ML performance dashboard:
    1. Active Model status & F1 metrics.
    2. Evaluation curves (ROC-AUC, PR comparison, Confusion Matrices).
    3. Explainability plots (SHAP).
    4. Reindexing controls & background retraining triggers.
    5. Retrieval log audits view.
    """
    st.subheader("📊 MLOps Dashboard & System Administration")
    
    # 1. Model Status Section
    status = get_model_status()
    if status and status.get("run_timestamp"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model F1-Score", f"{status.get('best_f1'):.4f}")
        col2.metric("Mean CV ROC-AUC", f"{status.get('best_roc_auc'):.4f}")
        col3.metric("Failure Rate (Baseline)", f"{status.get('failure_rate_in_dataset') * 100:.2f}%")
        
        st.markdown(f"**Best Parameters**: `{status.get('best_params')}`")
        st.markdown(f"**Primary failure indicators**: `{', '.join(status.get('top_features', []))}`")
        st.caption(f"Last training execution: {status.get('run_timestamp')}")
    else:
        st.info("No active training summary found. Please click the Retrain button below to initialize models.")

    # 2. Controls Section
    st.markdown("---")
    st.markdown("### 🛠️ Control Center")
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        st.markdown("#### Retrain Model")
        st.write("Starts the cross-validation hyperparameter sweep and logs metrics to MLflow.")
        if st.button("🚀 Trigger Background Retraining", use_container_width=True):
            res = trigger_retraining()
            if res:
                st.success("Retraining initiated! Check back in 1-2 minutes for updated parameters.")
                
    with control_col2:
        st.markdown("#### Re-Index Documents")
        st.write("Forces document ingestion, chunking, and database build.")
        use_semantic = st.checkbox("Enable Semantic Sentence Chunking (Slower but higher grounding accuracy)")
        if st.button("📂 Force Knowledge Base Re-Index", use_container_width=True):
            with st.spinner("Reindexing documents..."):
                ok = force_reindex(use_semantic)
                if ok:
                    st.success("Document reindexing completed successfully.")
                    
    # 3. Model Plots Section
    st.markdown("---")
    st.markdown("### 📈 Evaluation Plots")
    
    plots_tab1, plots_tab2, plots_tab3 = st.tabs(["Evaluation Curves", "SHAP Explainability", "Confusion Matrices"])
    
    plots_dir = os.path.join(settings.MODEL_ARTIFACTS_DIR, "plots")
    
    with plots_tab1:
        st.markdown("#### ROC and Precision-Recall Curve Comparisons")
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
        st.markdown("#### Random Forest Model Explainability (SHAP)")
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
        st.markdown("#### Model Confusion Matrices")
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

    # 4. Retrieval System Audit Logs
    st.markdown("---")
    st.markdown("### 📋 Retrieval Auditing System Log")
    log_lines = st.slider("Number of log lines to inspect", min_value=10, max_value=200, value=50, step=10)
    
    if st.button("🔍 Refresh Audit Logs", use_container_width=True):
        log_res = fetch_logs(log_lines)
        if log_res and log_res.get("exists"):
            st.text_area("retrieved_chunks.log (Tail)", value=log_res.get("content"), height=300)
        else:
            st.info("Log file is empty or not initialized yet.")
