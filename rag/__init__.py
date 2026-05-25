"""rag/__init__.py — RAG package public API for Task 4."""
from rag.chunking import split_into_sentences, semantic_chunk_text, chunk_documents
from rag.embeddings import get_embedder, embed_texts
from rag.retriever import (
    get_reranker,
    build_or_load_store,
    semantic_retrieve,
    build_bm25_index,
    bm25_search,
    hybrid_retrieve,
    rerank,
    query_cache,
    QueryCache,
)
from rag.generator import (
    is_valid_groq_key,
    get_groq_client,
    generate_llm_completion,
    get_mock_answer,
    check_faithfulness,
    parse_faithfulness_json,
    log_query,
)
from rag.pipeline import RAGPipeline, rag_pipeline
