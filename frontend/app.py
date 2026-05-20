import streamlit as st
import os
from frontend.components.chat_panel import render_chat_panel
from frontend.components.telemetry_panel import render_telemetry_panel

# 1. Page Configuration (MUST be called first, before other streamlit commands)
st.set_page_config(
    page_title="opsFlow RAG & Failure Prediction",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Step 3 - Secure Session State Initialization:
if "messages" not in st.session_state:
    st.session_state.messages = []

def run_app():
    # 2. Sidebar Simple Navigation
    st.sidebar.title("🔧 opsFlow Navigation")
    st.sidebar.markdown("Use the controls below to switch between predictive maintenance diagnostics and the RAG technical assistant.")
    
    navigation = st.sidebar.radio(
        "Select Section:",
        ["🔌 Equipment Failure Prediction", "💬 Technical RAG Assistant"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Setup & Models")
    custom_key = st.sidebar.text_input("Custom Groq API Key", type="password", help="Overrides default GROQ_API_KEY environment variable if provided.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Model Parameters")
    st.sidebar.write("**LLM Generator:** `llama-3.1-8b-instant`")
    st.sidebar.write("**Dense Embedder:** `all-MiniLM-L6-v2`")
    st.sidebar.write("**Cross-Encoder:** `ms-marco-MiniLM-L-6-v2`")
    st.sidebar.write("**Keyword Search:** BM25 (Okapi)")
    st.sidebar.write("**Vector Engine:** ChromaDB")

    # 3. Render Selected Section
    if navigation == "🔌 Equipment Failure Prediction":
        st.title("🔌 Equipment Failure Prediction (Task 3)")
        st.markdown("Adjust the operational sensors below to calculate failure probability and compare classification performance.")
        try:
            render_telemetry_panel()
        except Exception as e:
            st.error("Error rendering Equipment Failure Prediction panel")
            st.exception(e)
    else:
        st.title("💬 Technical RAG Assistant (Task 4)")
        st.markdown("Ask grounding-focused maintenance questions. All answers are strictly retrieved from our technical manuals database.")
        try:
            render_chat_panel(custom_key if custom_key else None)
        except Exception as e:
            st.error("Error rendering RAG Assistant panel")
            st.exception(e)

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error("Application crashed due to an unhandled runtime error")
        st.exception(e)
