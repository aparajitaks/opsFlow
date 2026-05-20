import os
import time
import streamlit as st
from groq import Groq
from core.config import settings
from core.security import sanitize_input, check_prompt_injection
from retrieval.chunker import chunk_documents
from retrieval.embedder import get_embedder
from retrieval.vector_store import build_or_load_store
from retrieval.bm25 import build_bm25_index, bm25_search
from retrieval.hybrid import hybrid_retrieve
from retrieval.reranker import get_rerank_model, rerank
from retrieval.cache import query_cache
from evaluation.faithfulness import check_faithfulness
from utils.logger import log_query

@st.cache_resource
def get_groq_client(api_key: str):
    """
    Creates and caches the Groq client instance in memory.
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip() or api_key.strip().lower() in ("none", "undefined", "null", ""):
        return None
    try:
        return Groq(api_key=api_key.strip())
    except Exception as e:
        print(f"[RAGService] Failed to initialize Groq client: {e}")
        return None

def is_valid_groq_key(api_key: str) -> bool:
    """
    Validates if the given key is a non-empty, non-trivial string.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    k = api_key.strip()
    return len(k) > 0 and k.lower() not in ("none", "undefined", "null", "")

class RAGService:
    """
    Orchestration service for the Retrieval-Augmented Generation (RAG) assistant.
    Initializes heavy models once and caches them.
    Manages security firewalls, query caches, and factual faithfulness auditing.
    """
    def __init__(self):
        self._embedder = None
        self._collection = None
        self._bm25_index = None
        self._chunks = None
        self._reranker = None
        self._is_initialized = False

    def initialize_pipeline(self, force: bool = False, use_semantic_chunking: bool = False):
        """
        Loads embedders, builds indexes, and creates ChromaDB collection.
        Saves time by caching components in memory.
        """
        if self._is_initialized and not force:
            return
            
        print("[RAGService] Starting RAG pipeline initialization...")
        docs_dir = str(settings.DOCS_DIR)
        persist_dir = str(settings.DATABASE_DIR)
        
        # 1. Load dense embedder model
        if self._embedder is None:
            self._embedder = get_embedder("all-MiniLM-L6-v2")
            
        # 2. Extract and segment text documents
        self._chunks = chunk_documents(docs_dir, self._embedder, chunk_size=300, overlap=50, use_semantic=use_semantic_chunking)
        print(f"[RAGService] Chunks created: {len(self._chunks)}")
        
        # 3. Setup persistent ChromaDB vector store
        self._collection = build_or_load_store(self._chunks, self._embedder, persist_dir)
        
        # 4. Build BM25 index
        self._bm25_index = build_bm25_index(self._chunks)
        
        # 5. Load Cross-Encoder reranker
        if self._reranker is None:
            self._reranker = get_rerank_model()
            
        self._is_initialized = True
        print("[RAGService] RAG pipeline initialization completed successfully.")

    def run_query(self, raw_query: str, groq_api_key: str = None) -> dict:
        """
        Main query responder path:
        1. Sanitize input & scan for prompt injections.
        2. Query cache lookup (exact & semantic).
        3. Hybrid Retrieval (BM25 + ChromaDB Vector search).
        4. Cross-Encoder Reranking (top 3 chunks).
        5. Check retrieval confidence and execute fallback if below threshold.
        6. Generate response from Groq client (grounded context).
        7. Audit generated answer for factual faithfulness (double-pass LLM pass).
        8. Audit logs and save to cache.
        """
        self.initialize_pipeline()
        
        # --- Security Firewall ---
        sanitized_query = sanitize_input(raw_query)
        if not sanitized_query:
            return {
                "answer": "Error: Empty query received.",
                "retrieved_chunks": [],
                "faithfulness": {"faithful": False, "score": 0.0, "verdict": "Empty query bypassed.", "unsupported_claims": []},
                "confidence_score": 0.0,
                "cached": False,
                "blocked": True
            }
            
        if check_prompt_injection(sanitized_query):
            print(f"[Security Alert] Bypassed prompt injection payload: '{sanitized_query}'")
            return {
                "answer": "Security block: Your query contains elements flagged by the prompt injection firewall. Please rephrase your question using standard maintenance terms.",
                "retrieved_chunks": [],
                "faithfulness": {"faithful": True, "score": 1.0, "verdict": "Prompt injection blocked.", "unsupported_claims": []},
                "confidence_score": 0.0,
                "cached": False,
                "blocked": True
            }

        # --- Cache Lookup ---
        cached_result = query_cache.get(sanitized_query, self._embedder)
        if cached_result:
            return {**cached_result, "cached": True, "blocked": False}

        # --- Hybrid Retrieval ---
        hybrid_chunks = hybrid_retrieve(
            query=sanitized_query,
            embedder=self._embedder,
            collection=self._collection,
            bm25_index=self._bm25_index,
            chunks=self._chunks,
            top_k=10
        )

        # --- Cross-Encoder Reranking ---
        reranked_chunks = rerank(
            query=sanitized_query,
            chunks=hybrid_chunks,
            model=self._reranker,
            top_n=3
        )

        # --- Confidence Calculation & Fallback ---
        # Normalize cross-encoder scores (which range from negative values to positive values)
        # We define confidence based on the highest cross_score returned.
        max_score = reranked_chunks[0]["cross_score"] if reranked_chunks else -99.0
        
        # Normalization approximation: cross-encoders range from -10 to +10.
        # Let's map -6.0 or below -> 0.0 confidence, and +2.0 or above -> 1.0 confidence.
        confidence = min(1.0, max(0.0, (max_score + 6.0) / 8.0))
        
        # Trigger fallback if confidence is extremely low
        if confidence < 0.15 or not reranked_chunks:
            answer = "I don't have enough information in my knowledge base to answer this question."
            faith_res = {
                "faithful": True,
                "score": 1.0,
                "verdict": "Low retrieval confidence. Refusal statement is factually faithful.",
                "unsupported_claims": []
            }
            res = {
                "answer": answer,
                "retrieved_chunks": [],
                "faithfulness": faith_res,
                "confidence_score": confidence,
                "cached": False,
                "blocked": False
            }
            # Cache the fallback result
            query_cache.set(sanitized_query, res, self._embedder)
            return res

        # --- Grounded Generation ---
        active_api_key = groq_api_key or settings.GROQ_API_KEY
        if is_valid_groq_key(active_api_key):
            try:
                groq_client = get_groq_client(active_api_key)
            except Exception as e:
                groq_client = None
                print(f"[RAGService] Error fetching cached Groq client: {e}")
            
            if groq_client:
                answer = self._generate_llm_completion(sanitized_query, reranked_chunks, groq_client)
            else:
                answer = self._get_mock_answer(sanitized_query, reranked_chunks)
        else:
            groq_client = None
            answer = self._get_mock_answer(sanitized_query, reranked_chunks)

        # --- Faithfulness Claim Audit ---
        faith_res = check_faithfulness(answer, reranked_chunks, groq_client)

        # --- Query Logging ---
        log_query(
            query=sanitized_query,
            retrieved_chunks=reranked_chunks,
            answer=answer,
            pre_rerank_chunks=hybrid_chunks,
            faithfulness_res=faith_res
        )

        # --- Save to Cache & Return ---
        result = {
            "answer": answer,
            "retrieved_chunks": reranked_chunks,
            "faithfulness": faith_res,
            "confidence_score": confidence,
            "cached": False,
            "blocked": False
        }
        query_cache.set(sanitized_query, result, self._embedder)
        return result

    def _get_mock_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Fallback mock generated answers for sandboxed testing.
        """
        if "water" in query.lower() or "boiling" in query.lower():
            return "The boiling point of water is exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
        return (
            "[MOCK RESPONDER - NO GROQ API KEY]\n"
            "Grounded Reference Chunks:\n"
            + "\n".join([f" • [{c['doc_name']} Chunk {c['chunk_index']}] - {c['text'][:80]}..." for c in retrieved_chunks])
        )

    def _generate_llm_completion(self, query: str, retrieved_chunks: list[dict], client: Groq) -> str:
        """
        Executes grounded generation call to Groq client with exponential backoff retries.
        """
        context = "\n\n".join([
            f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
            for c in retrieved_chunks
        ])
        
        max_retries = 3
        retry_delays = [3, 6, 12]
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=settings.GROQ_MODEL_GENERATOR,
                    messages=[
                        {"role": "system", "content": "You are an industrial maintenance assistant. Answer ONLY using the provided context. If the answer is not in the context, say exactly: 'I don't have enough information in my knowledge base to answer this question.' Do not use outside knowledge or extrapolate."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                is_429 = "429" in str(e) or "rate limit" in str(e).lower()
                if is_429 and attempt < max_retries - 1:
                    wait = retry_delays[attempt]
                    print(f"[Rate limit hit in Generator] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait)
                else:
                    return f"Error during Groq generation: {e}"
                    
        return "Rate limit exceeded in Generator after multiple retries."

# Global RAG Service instance
rag_service = RAGService()
