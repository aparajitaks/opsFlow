import streamlit as st
import os
from frontend.state import clear_query_cache
from frontend.components.chat_panel import render_chat_panel
from frontend.components.telemetry_panel import render_telemetry_panel
from frontend.components.metrics_panel import render_metrics_panel

# 1. Page Configuration (MUST be called first, before other streamlit commands)
st.set_page_config(
    page_title="opsFlow Maintenance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Step 3 - Secure Session State Initialization:
# Initialize critical session state keys at the very beginning of app execution to prevent Attribute/Key errors.
if "messages" not in st.session_state:
    st.session_state.messages = []

def run_app():
    # 2. Premium Styling Rules (Backdrop filters, glassmorphic themes)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Title and description styles */
        .title-text {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }
        .subtitle-text {
            font-size: 1.15rem;
            color: #94A3B8;
            margin-bottom: 2rem;
        }
        
        /* Custom stats display in sidebar */
        .sidebar-header {
            font-weight: 700;
            font-size: 1.25rem;
            color: #38BDF8;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #38BDF8;
            padding-bottom: 5px;
        }
        
        /* Message bubble enhancements */
        .stChatMessage {
            border-radius: 12px;
            margin-bottom: 1rem;
            padding: 1.2rem;
            border: 1px solid #1E293B;
            background-color: #0F172A;
        }
        
        /* Sources Cards styling */
        .source-card {
            background-color: #0B0F19;
            border-radius: 8px;
            padding: 0.9rem;
            margin-top: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #0072FF;
            border-right: 1px solid #1E293B;
            border-top: 1px solid #1E293B;
            border-bottom: 1px solid #1E293B;
        }
        .source-preview {
            font-size: 0.88rem;
            color: #94A3B8;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 3. Main Header
    st.markdown('<div class="title-text">🔧 opsFlow Intelligent Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Production-Grade RAG Semantic Grounding & Telemetry Failure Classification Console</div>', unsafe_allow_html=True)
    
    # 5. Sidebar controls
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>🔐 Security & Setup</div>", unsafe_allow_html=True)
        custom_key = st.text_input("Custom Groq API Key", type="password", help="Overrides system GROQ_API_KEY environment variable if provided.")
        
        st.markdown("<div class='sidebar-header'>⚙️ Action Controls</div>", unsafe_allow_html=True)
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared.")
            st.rerun()
            
        if st.button("⚡ Flush Query Cache", use_container_width=True):
            if clear_query_cache():
                st.success("Query cache flushed.")
                
        st.markdown("<div class='sidebar-header'>🏷️ Pipeline Metadata</div>", unsafe_allow_html=True)
        st.write("**LLM Generator:** `llama-3.1-8b-instant`")
        st.write("**LLM Auditor:** `llama-3.1-8b-instant`")
        st.write("**Dense Embedder:** `all-MiniLM-L6-v2`")
        st.write("**Cross-Encoder:** `ms-marco-MiniLM-L-6-v2`")
        st.write("**Keyword Search:** BM25 (Okapi)")
        st.write("**Vector Engine:** ChromaDB")
    
    # 6. Tab Navigation Layout
    tab_chat, tab_ml, tab_admin = st.tabs([
        "💬 Conversational RAG Assistant", 
        "⚙️ Telemetry Failure Diagnosis", 
        "📊 MLOps & System Admin"
    ])
    
    with tab_chat:
        try:
            render_chat_panel(custom_key if custom_key else None)
        except Exception as e:
            st.error("Error rendering Conversational RAG Assistant panel")
            st.exception(e)
            
    with tab_ml:
        try:
            render_telemetry_panel()
        except Exception as e:
            st.error("Error rendering Telemetry Failure Diagnosis panel")
            st.exception(e)
            
    with tab_admin:
        try:
            render_metrics_panel()
        except Exception as e:
            st.error("Error rendering MLOps & System Admin panel")
            st.exception(e)

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error("Application crashed due to an unhandled runtime error")
        st.exception(e)
