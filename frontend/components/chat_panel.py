import streamlit as st
from frontend.state import query_rag

def render_chat_panel(custom_api_key: str = None):
    """
    Renders the RAG Chatbot interface with scrolling conversation history,
    source document citations, confidence indicators, and faithfulness audits.
    """
    # Initialize session state for conversation messages if not set
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history rendering
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Citation expandable cards (Assistant only)
            if msg["role"] == "assistant":
                # 1. Caching label
                if msg.get("cached"):
                    st.caption("⚡ Response served from Cache (0ms latency)")
                
                # 2. Confidence Indicator
                if "confidence_score" in msg:
                    conf = msg["confidence_score"]
                    color = "green" if conf >= 0.7 else "orange" if conf >= 0.4 else "red"
                    st.markdown(f"**Confidence Level**: <span style='color:{color}; font-weight:bold;'>{conf*100:.1f}%</span>", unsafe_allow_html=True)
                
                # 3. Sources expansion
                if msg.get("sources"):
                    with st.expander("📚 View Grounding Sources"):
                        for idx, src in enumerate(msg["sources"]):
                            preview = src["text"].replace("\n", " ")[:160] + "..."
                            st.write(f"Chunk {idx+1} | {src['doc_name']} | Words {src['start_word']}–{src['end_word']} | Score: {src.get('score', 0.0):.2f}")
                            st.markdown(
                                f"""
                                <div class="source-card" style="margin-top: -10px; margin-bottom: 15px;">
                                    <div class="source-preview">{preview}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # 4. Faithfulness audit results
                if msg.get("faithfulness"):
                    faith = msg["faithfulness"]
                    score = faith.get("score", 0.0)
                    verdict = faith.get("verdict", "")
                    
                    if faith.get("faithful"):
                        st.success(f"✅ Factual Auditing: Faithful ({score:.2f})")
                        st.markdown(f'*" {verdict} "*')
                    else:
                        st.error(f"❌ Factual Auditing: Hallucination Flagged ({score:.2f})")
                        st.markdown(f'*" {verdict} "*')
                        if faith.get("unsupported_claims"):
                            st.markdown("**Unsupported LLM Claims:**")
                            for claim in faith["unsupported_claims"]:
                                st.markdown(f"- *{claim}*")

    # Chat Input handling
    if query := st.chat_input("Ask a maintenance question (e.g. 'How to resolve the warning thermal overload?')"):
        # User message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Loader spinner during API call
        with st.spinner("Executing RAG query pipeline..."):
            res = query_rag(query, custom_api_key)
            if res:
                # Add answer payload to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["retrieved_chunks"],
                    "faithfulness": res["faithfulness"],
                    "confidence_score": res["confidence_score"],
                    "cached": res["cached"]
                })
                st.rerun()
