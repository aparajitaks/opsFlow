import os
import time
from core.config import settings
from rag.chunking import chunk_documents
from rag.embeddings import get_embedder
from rag.retriever import (
    build_or_load_store,
    build_bm25_index,
    hybrid_retrieve,
    get_reranker,
    rerank,
    query_cache
)
from rag.generator import (
    is_valid_groq_key,
    get_groq_client,
    sanitize_input,
    check_prompt_injection,
    generate_llm_completion,
    get_mock_answer,
    check_faithfulness,
    log_query
)

class RAGPipeline:
    """
    Unified modular orchestration class for the Retrieval-Augmented Generation (RAG) pipeline.
    Initializes heavy neural models thread-safely via singletons, handles security firewalls,
    hybrid reciprocal-rank search, cross-encoder reranking, cached lookups, and double-pass faithfulness claim audits.
    """
    def __init__(self):
        self._embedder = None
        self._collection = None
        self._bm25_index = None
        self._chunks = None
        self._reranker = None
        self._is_initialized = False

    def initialize_pipeline(self, force: bool = False, use_semantic_chunking: bool = False):
        """Loads dense embedders, builds lexical indexes, and prepares persistent ChromaDB."""
        if self._is_initialized and not force:
            return
            
        print("[RAG] Starting consolidated RAG pipeline initialization...")
        docs_dir = str(settings.DOCS_DIR)
        persist_dir = str(settings.DATABASE_DIR)
        
        # 1. Load CPU-based sentence-transformer embedder
        if self._embedder is None:
            self._embedder = get_embedder("all-MiniLM-L6-v2")
            
        # 2. Extract and segment text documents (support semantic chunking or sliding window)
        self._chunks = chunk_documents(
            docs_dir, 
            self._embedder, 
            chunk_size=300, 
            overlap=50, 
            use_semantic=use_semantic_chunking
        )
        print(f"[RAG] Document segmenting complete. Generated chunks: {len(self._chunks)}")
        
        # 3. Setup persistent SQLite-backed ChromaDB vector store
        self._collection = build_or_load_store(self._chunks, self._embedder, persist_dir)
        
        # 4. Build lexical BM25 index
        self._bm25_index = build_bm25_index(self._chunks)
        
        # 5. Load Cross-Encoder reranking model
        if self._reranker is None:
            self._reranker = get_reranker()
            
        self._is_initialized = True
        print("[RAG] Consolidated RAG pipeline successfully initialized.")

    def run_query(self, raw_query: str, groq_api_key: str = None) -> dict:
        """
        Grounded hybrid-retrieval generation loop:
        1. Sanitize input & run prompt injection regex firewalls.
        2. Query cache lookup (exact matches and semantic cosine matching).
        3. Reciprocal Rank Fusion hybrid retrieval (lexical BM25 and vector ChromaDB).
        4. Cross-Encoder self-attention re-ranking (filtering top 3 chunks).
        5. Verify retrieval confidence, triggering graceful refusal if too low.
        6. Execute grounded completions with Groq generator.
        7. Audit generated text factual faithfulness via Groq claims checking.
        8. Log the interaction parameters to log file & update the cache.
        """
        self.initialize_pipeline()
        
        # --- 1. Security Firewalls ---
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
            print(f"[Security Alert] Flagged potential prompt injection payload: '{sanitized_query}'")
            return {
                "answer": "Security block: Your query contains elements flagged by the prompt injection firewall. Please rephrase your question using standard maintenance terms.",
                "retrieved_chunks": [],
                "faithfulness": {"faithful": True, "score": 1.0, "verdict": "Prompt injection blocked.", "unsupported_claims": []},
                "confidence_score": 0.0,
                "cached": False,
                "blocked": True
            }

        # --- 2. Cache Lookup ---
        cached_result = query_cache.get(sanitized_query, self._embedder)
        if cached_result:
            return {**cached_result, "cached": True, "blocked": False}

        # --- 3. Hybrid RRF Retrieval ---
        hybrid_chunks = hybrid_retrieve(
            query=sanitized_query,
            embedder=self._embedder,
            collection=self._collection,
            bm25_index=self._bm25_index,
            chunks=self._chunks,
            top_k=10
        )

        # --- 4. Cross-Encoder Re-Ranking ---
        reranked_chunks = rerank(
            query=sanitized_query,
            chunks=hybrid_chunks,
            model=self._reranker,
            top_n=3
        )

        # --- 5. Confidence Verification & Fallbacks ---
        max_score = reranked_chunks[0]["cross_score"] if reranked_chunks else -99.0
        
        # Map cross-encoder scores range approx [-6.0, 2.0] to normalized confidence [0.0, 1.0]
        confidence = min(1.0, max(0.0, (max_score + 6.0) / 8.0))
        
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
            query_cache.set(sanitized_query, res, self._embedder)
            return res

        # --- 6. Grounded Completion ---
        active_api_key = groq_api_key or settings.GROQ_API_KEY
        if is_valid_groq_key(active_api_key):
            groq_client = get_groq_client(active_api_key)
            if groq_client:
                answer = generate_llm_completion(sanitized_query, reranked_chunks, groq_client)
            else:
                answer = get_mock_answer(sanitized_query, reranked_chunks)
        else:
            groq_client = None
            answer = get_mock_answer(sanitized_query, reranked_chunks)

        # --- 7. Faithfulness Auditing ---
        faith_res = check_faithfulness(answer, reranked_chunks, groq_client)

        # --- 8. persistent Logging ---
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

# Expose global instance
rag_pipeline = RAGPipeline()
