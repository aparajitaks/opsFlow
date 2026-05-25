"""
rag/pipeline.py — RAG orchestration for Task 4.

Single-class orchestrator that:
  1. Initializes embedder, ChromaDB collection, BM25 index, and cross-encoder on first use.
  2. On each query: sanitize → cache check → hybrid retrieve → rerank → generate → audit.
"""
import math

from core.config import settings
from core.logger import get_logger
from core.security import sanitize_input, check_prompt_injection

log = get_logger("rag.pipeline")

from rag.chunking import chunk_documents
from rag.embeddings import get_embedder
from rag.retriever import (
    build_or_load_store,
    build_bm25_index,
    hybrid_retrieve,
    get_reranker,
    rerank,
    query_cache,
)
from rag.generator import (
    is_valid_groq_key,
    get_groq_client,
    generate_llm_completion,
    get_mock_answer,
    check_faithfulness,
    log_query,
)


class RAGPipeline:
    """
    Orchestrates the full RAG flow for industrial maintenance Q&A.

    Heavy models (embedder, cross-encoder) are loaded once and reused across queries.
    """

    def __init__(self):
        self._embedder = None
        self._collection = None
        self._bm25_index = None
        self._chunks = None
        self._reranker = None
        self._is_initialized = False

    def initialize_pipeline(self, force: bool = False, use_semantic_chunking: bool = False):
        """Load models and build indexes. Idempotent — safe to call multiple times."""
        if self._is_initialized and not force:
            return

        log.info("Initializing RAG pipeline...")
        docs_dir = str(settings.DOCS_DIR)
        persist_dir = str(settings.DATABASE_DIR)

        self._embedder = get_embedder(settings.EMBEDDING_MODEL)
        self._chunks = chunk_documents(
            docs_dir,
            self._embedder,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
            use_semantic=use_semantic_chunking or settings.USE_SEMANTIC,
        )
        log.info("Chunked %d segments from knowledge base.", len(self._chunks))

        self._collection = build_or_load_store(self._chunks, self._embedder, persist_dir)
        self._bm25_index = build_bm25_index(self._chunks)
        self._reranker = get_reranker()

        self._is_initialized = True
        log.info("RAG pipeline ready.")

    def run_query(
        self,
        raw_query: str,
        groq_api_key: str = None,
        conversation_history: list = None,
    ) -> dict:
        """
        Process one query through the full RAG pipeline.

        Steps:
          1. Sanitize + injection check
          2. Cache lookup (exact + semantic)
          3. Hybrid BM25 + dense retrieval with RRF
          4. Cross-encoder reranking
          5. Confidence check — refuse if below threshold
          6. Grounded generation (Groq or mock)
          7. Faithfulness audit
          8. Log + cache result
        """
        self.initialize_pipeline()

        # 1. Security
        sanitized = sanitize_input(raw_query)
        if not sanitized:
            return _blocked("Empty query.")

        if check_prompt_injection(sanitized):
            log.warning("Prompt injection blocked: '%s'", sanitized[:80])
            return _blocked(
                "Security block: query contains patterns flagged by the prompt injection check. "
                "Please rephrase using standard maintenance terminology."
            )

        # 2. Cache
        cached = query_cache.get(sanitized, self._embedder, similarity_threshold=settings.CACHE_SIM_THRESHOLD)
        if cached:
            return {**cached, "cached": True, "blocked": False}

        # 3. Hybrid retrieval
        hybrid_chunks = hybrid_retrieve(
            query=sanitized,
            embedder=self._embedder,
            collection=self._collection,
            bm25_index=self._bm25_index,
            chunks=self._chunks,
            top_k=settings.TOP_K,
            rrf_k=settings.RRF_K,
        )

        # 4. Reranking
        reranked = rerank(
            query=sanitized,
            chunks=hybrid_chunks,
            model=self._reranker,
            top_n=settings.TOP_N_RERANK,
        )

        # 5. Confidence check
        max_score = reranked[0]["cross_score"] if reranked else -99.0
        confidence = 1.0 / (1.0 + math.exp(-0.7 * (max_score + 2.0))) if reranked else 0.0

        if confidence < settings.CONFIDENCE_THRESH or not reranked:
            refusal = settings.KB_REFUSAL_MESSAGE
            faith_res = {
                "faithful": True,
                "score": 1.0,
                "verdict": "Low retrieval confidence — refusal is factually faithful.",
                "unsupported_claims": [],
            }
            result = {
                "answer": refusal,
                "retrieved_chunks": [],
                "faithfulness": faith_res,
                "confidence_score": confidence,
                "cached": False,
                "blocked": False,
            }
            query_cache.set(sanitized, result, self._embedder)
            log_query(query=sanitized, retrieved_chunks=[], answer=refusal, faithfulness_res=faith_res)
            return result

        # 6. Generation
        active_key = groq_api_key or settings.GROQ_API_KEY
        if is_valid_groq_key(active_key):
            groq_client = get_groq_client(active_key)
            answer = generate_llm_completion(sanitized, reranked, groq_client, conversation_history)
        else:
            groq_client = None
            answer = get_mock_answer(sanitized, reranked)

        # 7. Faithfulness audit
        faith_res = check_faithfulness(answer, reranked, groq_client)

        # 8. Log + cache
        log_query(
            query=sanitized,
            retrieved_chunks=reranked,
            answer=answer,
            pre_rerank_chunks=hybrid_chunks,
            faithfulness_res=faith_res,
        )

        result = {
            "answer": answer,
            "retrieved_chunks": reranked,
            "faithfulness": faith_res,
            "confidence_score": confidence,
            "cached": False,
            "blocked": False,
        }
        query_cache.set(sanitized, result, self._embedder)
        return result


def _blocked(reason: str) -> dict:
    return {
        "answer": reason,
        "retrieved_chunks": [],
        "faithfulness": {"faithful": True, "score": 1.0, "verdict": "Blocked.", "unsupported_claims": []},
        "confidence_score": 0.0,
        "cached": False,
        "blocked": True,
    }


# Global singleton — initialized lazily on first query
rag_pipeline = RAGPipeline()
