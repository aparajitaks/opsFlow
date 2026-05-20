import os
import threading
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.config import settings
from core.logger import get_logger
from rag.vector_store.chroma_store import build_or_load_store

log = get_logger("rag.retriever")

# Global cache for the CrossEncoder reranker model
_reranker_instance = None
_reranker_lock = threading.Lock()

def get_reranker(model_name: str = None) -> CrossEncoder:
    if model_name is None:
        model_name = settings.RERANKER_MODEL
    """
    Thread-safe global singleton caching of CrossEncoder model
    to prevent redundant initialization and save CPU cycles.
    """
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                log.info("Initializing Cross-Encoder '%s' on CPU.", model_name)
                _reranker_instance = CrossEncoder(model_name, device="cpu")
    return _reranker_instance


def semantic_retrieve(query: str, embedder, collection, k: int = 3, where_filter: dict = None) -> list[dict]:
    """
    Embeds the user query and searches ChromaDB for
    the top k semantically similar chunks.
    """
    query_vector = embedder.encode([query])[0].tolist()
    
    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": k
    }
    if where_filter:
        query_kwargs["where"] = where_filter
        
    results = collection.query(**query_kwargs)
    retrieved_chunks = []
    
    if results and "documents" in results and len(results["documents"]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[0.0] * len(documents)])[0]
        
        for idx in range(len(documents)):
            meta = metadatas[idx]
            distance = float(distances[idx])
            # Normalize L2 distance to score: 1.0 - distance
            score = 1.0 - distance
            
            retrieved_chunks.append({
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "start_word": meta["start_word"],
                "end_word": meta["end_word"],
                "word_count": meta["word_count"],
                "score": score,
                "text": documents[idx]
            })
            
    return retrieved_chunks

# BM25 Lexical Search Functions
def tokenize(text: str) -> list[str]:
    """Tokenizes text by lowercasing and splitting on whitespace."""
    return text.lower().split()

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Builds a BM25Okapi index from a list of chunks."""
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)

def bm25_search(query: str, bm25_index: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Searches the BM25 index for the top_k closest lexical matches."""
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    ranked_results = []
    for idx, score in enumerate(scores):
        ranked_results.append({
            "chunk": chunks[idx],
            "bm25_score": float(score)
        })
        
    ranked_results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return ranked_results[:top_k]

# Hybrid Reciprocal Rank Fusion Retrieval
def hybrid_retrieve(query: str, embedder, collection, bm25_index, chunks: list[dict], top_k: int = 10, rrf_k: int = 60) -> list[dict]:
    """
    Merges semantic search (ChromaDB) and BM25 search results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Run semantic search
    semantic_chunks = semantic_retrieve(query, embedder, collection, k=top_k)
    
    # 2. Run BM25 search
    bm25_results = bm25_search(query, bm25_index, chunks, top_k=top_k)
    
    bm25_chunks = []
    for item in bm25_results:
        c = item["chunk"].copy()
        c["bm25_score"] = item["bm25_score"]
        bm25_chunks.append(c)
        
    fused_map = {}
    semantic_rank_map = {c["chunk_index"]: idx + 1 for idx, c in enumerate(semantic_chunks)}
    bm25_rank_map = {c["chunk_index"]: idx + 1 for idx, c in enumerate(bm25_chunks)}
    
    for c in semantic_chunks:
        idx = c["chunk_index"]
        fused_map[idx] = {
            "doc_name": c["doc_name"],
            "chunk_index": idx,
            "start_word": c["start_word"],
            "end_word": c["end_word"],
            "word_count": c["word_count"],
            "text": c["text"],
            "semantic_score": c["score"],
            "bm25_score": 0.0,
            "sources": ["semantic"]
        }
        
    for c in bm25_chunks:
        idx = c["chunk_index"]
        if idx in fused_map:
            fused_map[idx]["bm25_score"] = c["bm25_score"]
            fused_map[idx]["sources"].append("bm25")
        else:
            fused_map[idx] = {
                "doc_name": c["doc_name"],
                "chunk_index": idx,
                "start_word": c["start_word"],
                "end_word": c["end_word"],
                "word_count": c["word_count"],
                "text": c["text"],
                "semantic_score": 0.0,
                "bm25_score": c["bm25_score"],
                "sources": ["bm25"]
            }
            
    # Calculate RRF scores
    for idx, c in fused_map.items():
        sem_rank = semantic_rank_map.get(idx, float('inf'))
        bm_rank = bm25_rank_map.get(idx, float('inf'))
        
        score_sem = 1.0 / (rrf_k + sem_rank) if sem_rank != float('inf') else 0.0
        score_bm = 1.0 / (rrf_k + bm_rank) if bm_rank != float('inf') else 0.0
        
        c["rrf_score"] = score_sem + score_bm
        c["score"] = c["rrf_score"]
        
    fused_chunks = list(fused_map.values())
    fused_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return fused_chunks[:top_k]

# CrossEncoder Re-Ranking
def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_n: int = 3) -> list[dict]:
    """
    Scores each candidate pair (query, chunk_text) using the CrossEncoder rerank model.
    """
    if not chunks:
        return []
        
    for idx, c in enumerate(chunks):
        c["pre_rerank_rank"] = idx + 1
        
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    
    for idx, score in enumerate(scores):
        chunks[idx]["cross_score"] = float(score)
        chunks[idx]["score"] = float(score)
        
    reranked = sorted(chunks, key=lambda x: x["cross_score"], reverse=True)
    
    for idx, c in enumerate(reranked):
        c["post_rerank_rank"] = idx + 1
        
    return reranked[:top_n]

# Thread-safe Query Cache
class QueryCache:
    """Thread-safe query cache supporting exact and semantic matching."""
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}
        self._embeddings = {}

    def get(self, query: str, embedder=None, similarity_threshold: float = 0.95) -> list[dict] | None:
        query_clean = query.strip().lower()
        
        with self._lock:
            # 1. Exact Match Check
            for cached_query in self._cache:
                if cached_query.strip().lower() == query_clean:
                    print(f"[RAG Cache] Exact cache hit for query: '{query}'")
                    return self._cache[cached_query]
            
            # 2. Semantic Match Check
            if embedder is not None and len(self._cache) > 0:
                try:
                    query_emb = embedder.encode([query], show_progress_bar=False)[0]
                    
                    for cached_query, cached_emb in self._embeddings.items():
                        norm1 = np.linalg.norm(query_emb)
                        norm2 = np.linalg.norm(cached_emb)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(query_emb, cached_emb) / (norm1 * norm2)
                        else:
                            sim = 0.0
                            
                        if sim >= similarity_threshold:
                            print(f"[RAG Cache] Semantic cache hit (similarity {sim:.3f}): '{cached_query}' for '{query}'")
                            return self._cache[cached_query]
                except Exception as e:
                    print(f"[RAG Cache Warning] Semantic cache check failure: {e}")
                    
        return None

    def set(self, query: str, results: list[dict], embedder=None) -> None:
        with self._lock:
            self._cache[query] = results
            if embedder is not None:
                try:
                    self._embeddings[query] = embedder.encode([query], show_progress_bar=False)[0]
                except Exception as e:
                    print(f"[RAG Cache Warning] Embedding computation failed for cached query: {e}")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            print("[RAG Cache] Semantic query cache cleared successfully.")

query_cache = QueryCache()

# Detailed explanations for the Technical Recruiter
def get_chromadb_vs_faiss_explanation() -> str:
    return (
        "=== CHROMADB VS FAISS COMPARISON ---\n"
        "1. Complete Database vs. Core Vector Library: FAISS is a lightweight search library focused strictly "
        "   on in-memory similarity operations. It does not manage raw text, IDs, or metadata internally; developers "
        "   must maintain separate lookup tables. ChromaDB is a feature-rich, serverless vector database that "
        "   handles vectors, text contents, and metadata in an integrated SQL-backed store.\n"
        "2. Persistent Lifecycle: In FAISS, persistence requires dumping and loading index files manually. ChromaDB "
        "   manages persistence out of the box using a standardized SQLite-backed engine via PersistentClient.\n"
        "3. Metadata Filtering & CRUD: FAISS does not support direct semantic metadata filtering during indexing. "
        "   ChromaDB natively supports metadata filtering (e.g., retrieving only safety manual chunks) using standard "
        "   where clauses at search time, alongside simple CRUD operations to update or delete vectors dynamically."
    )

def get_bm25_explanation() -> str:
    return (
        "--- WHY BM25 CATCHES QUERIES THAT SEMANTIC SEARCH MISSES ---\n"
        "1. Precise Keyword Matching: BM25 is a term-frequency/inverse document frequency (TF-IDF) derived\n"
        "   lexical search algorithm. It scores documents based on exact keyword occurrences.\n"
        "2. Cosine Distance Coordinate Smoothing: Dense transformer models map terms into low-dimensional semantic\n"
        "   embedding spaces. For highly specific industrial codes, technical identifiers, or exact numerical values\n"
        "   (such as error codes like 'ERR-101' or voltage numbers like '480V'), the dense vector representation\n"
        "   smooths these values with surrounding words, yielding a near-zero similarity score for exact numeric lookups.\n"
        "3. Concrete Example:\n"
        "   If a technician queries 'How to resolve ERR-101?', a semantic search model might retrieve chunks related\n"
        "   to general faults (like general warnings or standard errors) due to the similarity of surrounding words\n"
        "   like 'resolve' and 'error'. However, BM25 targets the high-IDF rare keyword 'ERR-101', instantly boosting\n"
        "   the exact troubleshooting manual chunk containing that exact code to the top of the list."
    )

def get_rrf_explanation() -> str:
    return (
        "--- RECIPROCAL RANK FUSION (RRF) EXPLANATION ---\n"
        "1. What is RRF: Reciprocal Rank Fusion is a robust algorithm that merges multiple ranked lists of\n"
        "   search results into a single unified list. The rank position of a document in each retrieval run\n"
        "   contributes reciprocally to its unified RRF score. Highly ranked documents in any list receive\n"
        "   exponentially higher score boosts.\n"
        "2. Why it is Superior to Score Averaging:\n"
        "   - Different Mathematical Coordinate Systems: Cosine similarity scores from dense embeddings represent\n"
        "     distance measurements within a high-dimensional vector space (typically ranging from -1 to 1 or 0 to 1).\n"
        "     BM25 scores represent unbounded logarithmic term-match values (ranging from 0 to 20+ depending on query size).\n"
        "     Averaging these raw scores is mathematically nonsensical (like adding Celsius to Fahrenheit directly).\n"
        "   - Calibration Independence: Document scoring scales vary dramatically per query based on document size,\n"
        "     frequency, and dense coordinate density. RRF operates purely on ordinal ranks, making it completely\n"
        "     calibrated, translation-invariant, and highly resilient to scoring noise in both dense and lexical channels."
    )

def get_rerank_explanation() -> str:
    return (
        "--- BI-ENCODER VS CROSS-ENCODER EXPLANATION ---\n"
        "1. Bi-Encoder Architecture (Embeddings):\n"
        "   - Processes query and documents completely independently of each other. The document text is mapped to\n"
        "     a dense vector representation at index time; the query is mapped to a vector at runtime.\n"
        "   - Similarity is computed via a simple, fast dot-product or cosine distance between these two vectors.\n"
        "   - Advantage: Extremely fast (sub-millisecond searching) across massive millions-scale document databases.\n"
        "   - Disadvantage: Cannot capture fine-grained token-to-token interactions because query and document tokens\n"
        "     never attend to each other during modeling.\n"
        "2. Cross-Encoder Architecture (Re-Ranking):\n"
        "   - Processes the query and candidate document joined together as a single input sequence (Query [SEP] Document).\n"
        "   - Applies full, deep self-attention across every query token and document token at every layer of the transformer.\n"
        "   - Advantage: Massive boost in lexical/semantic reasoning accuracy by modeling full token interactions.\n"
        "   - Disadvantage: Computationally expensive. Running deep self-attention over thousands of documents is too\n"
        "     slow for first-pass retrieval. It is therefore utilized as a powerful second-pass re-ranker over a smaller\n"
        "     set of candidate documents (e.g., top 10)."
    )
