import streamlit as st
from frontend.state import predict_ml

def render_telemetry_panel():
    """
    Renders telemetry sliders, model selection dropdown, failure prediction triggers,
    engineered features calculation, and failure probability gauges.
    """
    st.subheader("⚙️ Real-time Equipment Failure Predictor")
    st.markdown("Adjust telemetry attributes below to simulate operational states and calculate failure probability.")
    
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
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown("#### Diagnosis Status")
                    if pred == 1:
                        st.error("🚨 EQUIPMENT FAILURE DETECTED")
                        st.markdown(
                            """
                            <div style="background-color:#7f1d1d; border-radius:8px; padding:15px; border-left: 6px solid #ef4444; color: #fecaca; margin-bottom: 20px;">
                                <strong>High Risk Warning:</strong> Telemetry values violate normal threshold boundaries. 
                                Immediate inspection or tooling replacement is recommended.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.success("🟢 STABLE OPERATIONS VERIFIED")
                        st.markdown(
                            """
                            <div style="background-color:#14532d; border-radius:8px; padding:15px; border-left: 6px solid #22c55e; color: #bbf7d0; margin-bottom: 20px;">
                                <strong>System Safe:</strong> Machine parameters reside within normal operating bands. 
                                Equipment is safe for continuous runtime.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
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
