import streamlit as st
from frontend.state import query_rag

def render_chat_panel(custom_api_key: str = None):
    """
    Renders the simplified RAG interface with clean native Streamlit chat bubbles
    and standard document citation expanders.
    """
    # Initialize session state for conversation messages if not set
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history rendering
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Citation indicators (Assistant only)
            if msg["role"] == "assistant":
                # 1. Caching label
                if msg.get("cached"):
                    st.caption("⚡ Response served from Cache (0ms latency)")
                
                # 2. Confidence Indicator
                if "confidence_score" in msg:
                    conf = msg["confidence_score"]
                    st.markdown(f"**Retrieval Confidence**: `{conf*100:.1f}%`")
                
                # 3. Sources expansion
                if msg.get("sources"):
                    with st.expander("📚 View Grounding Sources"):
                        for idx, src in enumerate(msg["sources"]):
                            st.markdown(f"**Chunk {idx+1} | {src['doc_name']} (Relevance Score: {src.get('score', 0.0):.2f})**")
                            st.markdown(f"```text\n{src['text']}\n```")
                
                # 4. Faithfulness audit results
                if msg.get("faithfulness"):
                    faith = msg["faithfulness"]
                    score = faith.get("score", 0.0)
                    verdict = faith.get("verdict", "")
                    
                    if faith.get("faithful"):
                        st.success(f"✅ Faithfulness Check: Passed ({score:.2f})")
                        st.caption(verdict)
                    else:
                        st.warning(f"⚠️ Faithfulness Check: Potential Hallucination ({score:.2f})")
                        st.caption(verdict)
                        if faith.get("unsupported_claims"):
                            st.markdown("**Unsupported Claims:**")
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
