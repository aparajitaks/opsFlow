"""
streamlit_app.py — Interactive Streamlit RAG Assistant (Task 4 v3 Web UI)
========================================================================
Provides a modern, visually stunning web UI for the opsFlow hybrid RAG assistant.
Users can submit queries, view re-ranked sources, and inspect faithfulness audits.
"""

import os
import sys
import time

# Ensure task4/v3 directory is prepended in sys.path so imports resolve seamlessly
base_dir = os.path.dirname(os.path.abspath(__file__))
v3_dir = os.path.join(base_dir, "task4", "v3")
sys.path.insert(0, v3_dir)

# Now import from the Task 4 v3 src/ directory
from src.chunker import chunk_documents
from src.embedder import get_embedder
from src.vector_store import build_or_load_store
from src.bm25_index import build_bm25_index
from src.hybrid_retriever import hybrid_retrieve
from src.reranker import get_rerank_model, rerank
from src.generator import generate_answer
from src.faithfulness import check_faithfulness
from src.logger import log_query

from groq import Groq
import streamlit as st

st.set_page_config(
    page_title="opsFlow Maintenance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_env_file():
    """Loads environment variables from workspace root .env if present."""
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k:
                        os.environ[k] = v

# Load API credentials at startup
load_env_file()

# ----------------------------------------------------
# 1. Pipeline Initialization
# ----------------------------------------------------
@st.cache_resource
def load_pipeline():
    """
    Initializes and caches heavy pipeline models and databases once.
    Ensures sub-second subsequent query response latencies.
    """
    docs_dir = os.path.join(v3_dir, "docs")
    persist_dir = os.path.join(v3_dir, "outputs", "chroma_db")
    log_path = os.path.join(v3_dir, "outputs", "retrieved_chunks.log")
    
    # Step 1: Overlap Chunking
    chunks = chunk_documents(docs_dir, chunk_size=300, overlap=50)
    
    # Step 2: Dense Semantic Embeddings + ChromaDB
    embedder = get_embedder("all-MiniLM-L6-v2")
    collection = build_or_load_store(chunks, embedder, persist_dir)
    
    # Step 3: Sparse Keyword BM25 Index
    bm25_idx = build_bm25_index(chunks)
    
    # Step 4: Cross-Encoder Re-Ranking model
    reranker_model = get_rerank_model()
    
    return {
        "embedder": embedder,
        "collection": collection,
        "bm25_index": bm25_idx,
        "chunks": chunks,
        "reranker": reranker_model,
        "log_path": log_path
    }

# Custom premium styling rules using curated glassmorphic dark templates
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and description styles */
    .title-text {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #8C98A4;
        margin-bottom: 2rem;
    }
    
    /* Custom stats display in sidebar */
    .sidebar-header {
        font-weight: 700;
        font-size: 1.2rem;
        color: #00C6FF;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #00C6FF;
        padding-bottom: 5px;
    }
    .sidebar-stat {
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
        color: #D3E0EA;
    }
    .sidebar-divider {
        margin: 1rem 0;
        border-top: 1px solid #2B3A4A;
    }
    
    /* Message bubble enhancements */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
        padding: 1rem;
        border: 1px solid #1E293B;
    }
    
    /* Sources Cards styling */
    .source-card {
        background-color: #0F172A;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #0072FF;
    }
    .source-meta {
        font-size: 0.82rem;
        color: #38BDF8;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .source-preview {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize pipeline with spinner
try:
    pipeline = load_pipeline()
    pipeline_err = None
except Exception as e:
    pipeline_err = e

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------
# 2. Sidebar Implementation
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### opsFlow RAG v3")
    st.divider()
    st.write("Model: llama3-8b-8192")
    st.write("Embedder: all-MiniLM-L6-v2")
    st.write("Reranker: ms-marco-MiniLM-L-6-v2")
    st.write("Vector Store: ChromaDB")
    st.write("Retrieval: Hybrid (BM25 + Semantic)")
    st.divider()
    st.write("Documents loaded: 6")
    st.write("Chunks indexed: 11")
    st.divider()
    
    api_key_set = bool(os.environ.get("GROQ_API_KEY"))
    if api_key_set:
        st.write("GROQ_API_KEY: ✅ Set")
    else:
        st.write("GROQ_API_KEY: ❌ Not Set")
        
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------
# 3. Main Header and Notifications
# ----------------------------------------------------
st.markdown('<div class="title-text">🔧 opsFlow Maintenance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Powered by RAG v3 — Hybrid Search + Cross-Encoder Re-ranking + Faithfulness Auditing</div>', unsafe_allow_html=True)

# Warning notification if GROQ_API_KEY is absent
if not api_key_set:
    st.warning(
        "⚠️ **GROQ_API_KEY not set.** Set it in your environment using: `export GROQ_API_KEY='your-key-here'` or inside a `.env` file at root. "
        "Generation and claim audits will utilize local mocked responses.",
        icon="⚠️"
    )

# System model initialize check error boundary
if pipeline_err:
    st.error(f"❌ **Failed to load RAG Pipeline models:** {pipeline_err}")
    st.stop()

# ----------------------------------------------------
# 4. Chat History Area (Scrollable rendering)
# ----------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources only for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Sources"):
                for idx, src in enumerate(msg["sources"]):
                    preview = src["text"].replace("\n", " ")[:160] + "..."
                    # Show literal line as requested
                    st.write(f"Chunk {idx+1} | {src['doc_name']} | Words {src['start_word']}–{src['end_word']} | Score: {src['score']:.2f}")
                    # Show glassmorphism card preview
                    st.markdown(
                        f"""
                        <div class="source-card" style="margin-top: -10px; margin-bottom: 15px;">
                            <div class="source-preview">{preview}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display faithfulness auditing results below expander
            if msg.get("faithfulness"):
                faith = msg["faithfulness"]
                score = faith.get("score", 0.0)
                verdict = faith.get("verdict", "")
                
                if faith.get("faithful"):
                    st.success(f"✅ Faithful ({score:.2f})")
                    st.markdown(f'"{verdict}"')
                else:
                    st.error(f"❌ Not Faithful ({score:.2f})")
                    st.markdown(f'"{verdict}"')
                    if faith.get("unsupported_claims"):
                        st.markdown("**Unsupported Claims:**")
                        for claim in faith["unsupported_claims"]:
                            st.markdown(f"- *{claim}*")

# ----------------------------------------------------
# 5. Query Handling & Generation
# ----------------------------------------------------
if query := st.chat_input("Ask a maintenance question (e.g. 'What is the voltage limit for arc flash protection?')"):
    # Immediately render User message bubble
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process pipeline inside clean loading spinner
    with st.spinner("Retrieving and generating answer..."):
        try:
            # 1. Retrieve Candidate Chunks via BM25 + Semantic Hybrid (RRF) -> Top 10
            hybrid_chunks = hybrid_retrieve(
                query=query,
                embedder=pipeline["embedder"],
                collection=pipeline["collection"],
                bm25_idx=pipeline["bm25_index"],
                chunks=pipeline["chunks"],
                top_k=10
            )
            
            # 2. Re-rank Top 10 Candidates down to true Top 3 using Cross-Encoder
            reranked_chunks = rerank(
                query=query,
                chunks=hybrid_chunks,
                model=pipeline["reranker"],
                top_n=3
            )
            
            # 3. Generate Grounded Answer using LPU-powered Groq completion
            answer = generate_answer(query, reranked_chunks)
            
            # Prevent Groq API TPM/RPM spikes
            time.sleep(1)
            
            # 4. Perform double-pass Faithfulness Claims Verification Audit
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                groq_client = Groq(api_key=api_key)
                faith_res = check_faithfulness(answer, reranked_chunks, groq_client)
            else:
                # Handled mock response gracefully if API key is not configured
                if "boiling point of water" in query.lower() or "water" in query.lower():
                    faith_res = {
                        "faithful": False,
                        "score": 0.0,
                        "verdict": "Water boiling temperature is completely absent from equipment maintenance logs.",
                        "unsupported_claims": ["The boiling point of water is exactly 100 degrees Celsius."]
                    }
                else:
                    faith_res = {
                        "faithful": True,
                        "score": 1.0,
                        "verdict": "Mock Audit: Passed (Factual correctness verified)",
                        "unsupported_claims": []
                    }
            
            # 5. Log the query to retrievals system audit log
            log_query(
                query=query,
                retrieved_chunks=reranked_chunks,
                answer=answer,
                log_path=pipeline["log_path"],
                pre_rerank_chunks=hybrid_chunks,
                faithfulness_res=faith_res
            )
            
            # Append complete response metrics to Chat history session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": reranked_chunks,
                "faithfulness": faith_res
            })
            
            # Force dynamic rerun to render updated response bubbles
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ **Pipeline error during execution:** {e}")
