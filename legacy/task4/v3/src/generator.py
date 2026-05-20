import os
import time
from groq import Groq

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Step 3: Grounded generation using Groq.
    Enforces deterministic output (temperature=0.0) based solely on retrieved chunks.
    Implements a robust retry logic with exponential backoff on 429 rate limit errors.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "[MOCK LLM RESPONSE - NO GROQ API KEY SET]\n"
            "To resolve, set your Groq API Key: export GROQ_API_KEY='your-key'\n"
            "Determined Grounding Context:\n"
            + "\n".join([f" - [{c['doc_name']} | Chunk {c['chunk_index']}]" for c in retrieved_chunks])
        )
        
    client = Groq(api_key=api_key)
    context = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
    
    max_retries = 3
    retry_delays = [3, 6, 12]  # seconds to wait between retries
    
    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are an industrial maintenance assistant. Answer ONLY using the provided context. If the answer is not in the context, say exactly: 'I don't have enough information in my knowledge base to answer this question.' Do not use outside knowledge."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                # Handle Groq model decommissioning by falling back to modern llama-3.1-8b-instant
                if "decommissioned" in str(e) or "not found" in str(e) or "400" in str(e):
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are an industrial maintenance assistant. Answer ONLY using the provided context. If the answer is not in the context, say exactly: 'I don't have enough information in my knowledge base to answer this question.' Do not use outside knowledge."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                        ],
                        temperature=0.0
                    )
                    return response.choices[0].message.content
                else:
                    raise e
        except Exception as e:
            # Check if this is a 429 Rate Limit error
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            if is_429 and attempt < max_retries - 1:
                wait = retry_delays[attempt]
                print(f"[Rate limit hit] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                return f"Error during Groq generation: {e}"
                
    return "Rate limit exceeded after retries. Please wait a moment and try again."

def get_groq_explanation() -> str:
    return (
        "--- WHY GROQ IS UTILIZED FOR GENERATION ---\n"
        "1. Blazing Fast Inference: Groq utilizes custom LPU (Language Processing Unit) hardware, "
        "   delivering speeds exceeding 500+ tokens per second. This ensures real-time microsecond "
        "   latencies inside responsive industrial conversational systems.\n"
        "2. Cost-Effective Scaling: Groq offers a generous free tier for developers, eliminating barriers "
        "   to initial validation, testing, and sandbox scaling.\n"
        "3. Strong Factual Performance: The Llama-3-8B model performs exceptionally well at grounded "
        "   factual question-answering tasks when constrained under tight temperature conditions, minimizing "
        "   hallucinations and enforcing exact contextual compliance."
    )
