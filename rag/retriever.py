"""
rag/retriever.py — hybrid retrieval for Task 4.

Components:
  - Dense semantic search via ChromaDB
  - Lexical BM25 search via rank-bm25
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - Cross-encoder reranking for final ordering
  - In-memory query cache with exact + semantic matching
"""
import threading
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.config import settings
from core.logger import get_logger
from rag.vector_store.chroma_store import build_or_load_store

log = get_logger("rag.retriever")

# Cross-encoder singleton — loaded once per process
_reranker_instance = None


def get_reranker(model_name: str = None) -> CrossEncoder:
    """Returns the cross-encoder model, loading it on first call."""
    global _reranker_instance
    if model_name is None:
        model_name = settings.RERANKER_MODEL
    if _reranker_instance is None:
        log.info("Loading cross-encoder '%s'.", model_name)
        _reranker_instance = CrossEncoder(model_name, device="cpu")
    return _reranker_instance


# ---------------------------------------------------------------------------
# Dense semantic search
# ---------------------------------------------------------------------------

def semantic_retrieve(query: str, embedder, collection, k: int = 3) -> list[dict]:
    """Embeds the query and searches ChromaDB for top-k similar chunks."""
    query_vector = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    chunks = []
    if not results or not results.get("documents") or not results["documents"][0]:
        return chunks

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[0.0] * len(docs)])[0]

    for i in range(len(docs)):
        meta = metas[i]
        chunks.append({
            "doc_name": meta["doc_name"],
            "chunk_index": meta["chunk_index"],
            "start_word": meta["start_word"],
            "end_word": meta["end_word"],
            "word_count": meta["word_count"],
            "score": 1.0 - float(dists[i]),
            "text": docs[i],
        })
    return chunks


# ---------------------------------------------------------------------------
# BM25 lexical search
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Builds a BM25Okapi index over the chunk corpus."""
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_search(query: str, bm25_index: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Returns top-k chunks ranked by BM25 score."""
    scores = bm25_index.get_scores(tokenize(query))
    ranked = [{"chunk": chunks[i], "bm25_score": float(scores[i])} for i in range(len(chunks))]
    ranked.sort(key=lambda x: x["bm25_score"], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def expand_acronyms(query: str) -> str:
    """Preprocesses query strings to expand common industrial maintenance acronyms for higher recall."""
    import re
    replacements = {
        r"\bloto\b": "Lockout-Tagout",
        r"\brtd\b": "Resistance Temperature Detector",
        r"\bhmi\b": "Human-Machine Interface",
        r"\bplc\b": "Programmable Logic Controller",
        r"\bvfd\b": "Variable Frequency Drive",
        r"\bemi\b": "Electromagnetic Interference",
    }
    expanded = query
    for pattern, replacement in replacements.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def hybrid_retrieve(
    query: str,
    embedder,
    collection,
    bm25_index,
    chunks: list[dict],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Merges dense and lexical results with Reciprocal Rank Fusion.

    RRF score: sum(1 / (k + rank)) across retrieval channels.
    This is rank-based (not score-based), making it robust to different score scales.
    """
    expanded_query = expand_acronyms(query)
    semantic_chunks = semantic_retrieve(expanded_query, embedder, collection, k=top_k)
    bm25_results = bm25_search(expanded_query, bm25_index, chunks, top_k=top_k)
    bm25_chunks = [{**item["chunk"], "bm25_score": item["bm25_score"]} for item in bm25_results]

    sem_rank = {c["chunk_index"]: i + 1 for i, c in enumerate(semantic_chunks)}
    bm_rank = {c["chunk_index"]: i + 1 for i, c in enumerate(bm25_chunks)}

    fused: dict[int, dict] = {}
    for c in semantic_chunks:
        idx = c["chunk_index"]
        fused[idx] = {**c, "bm25_score": 0.0, "sources": ["semantic"]}
    for c in bm25_chunks:
        idx = c["chunk_index"]
        if idx in fused:
            fused[idx]["bm25_score"] = c["bm25_score"]
            fused[idx]["sources"].append("bm25")
        else:
            fused[idx] = {**c, "semantic_score": 0.0, "sources": ["bm25"]}

    for idx, c in fused.items():
        sr = sem_rank.get(idx, float("inf"))
        br = bm_rank.get(idx, float("inf"))
        rrf = (1.0 / (rrf_k + sr) if sr != float("inf") else 0.0) + \
              (1.0 / (rrf_k + br) if br != float("inf") else 0.0)
        c["rrf_score"] = rrf
        c["score"] = rrf

    results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_n: int = 3) -> list[dict]:
    """
    Scores query-chunk pairs with a cross-encoder and returns top_n by score.
    Cross-encoders see both query and chunk together, giving better accuracy than bi-encoders alone.
    """
    if not chunks:
        return []

    for i, c in enumerate(chunks):
        c["pre_rerank_rank"] = i + 1

    expanded_query = expand_acronyms(query)
    scores = model.predict([(expanded_query, c["text"]) for c in chunks])
    for i, score in enumerate(scores):
        chunks[i]["cross_score"] = float(score)
        chunks[i]["score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["cross_score"], reverse=True)
    for i, c in enumerate(reranked):
        c["post_rerank_rank"] = i + 1

    return reranked[:top_n]


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class QueryCache:
    """
    In-memory cache with exact-match and semantic-similarity lookup.

    Semantic lookup embeds the incoming query and compares cosine similarity
    against cached query embeddings. Avoids re-running the full pipeline for
    near-duplicate queries within a session.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._embeddings: dict = {}

    def get(self, query: str, embedder=None, similarity_threshold: float = 0.95) -> dict | None:
        query_norm = query.strip().lower()
        with self._lock:
            # Exact match (case-insensitive, strip whitespace)
            for cq in self._cache:
                if cq.strip().lower() == query_norm:
                    log.debug("Cache exact hit: '%s'", query)
                    return self._cache[cq]

            # Semantic match
            if embedder and self._embeddings:
                try:
                    qv = embedder.encode([query], show_progress_bar=False)[0]
                    for cq, cv in self._embeddings.items():
                        n1, n2 = np.linalg.norm(qv), np.linalg.norm(cv)
                        sim = float(np.dot(qv, cv) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
                        if sim >= similarity_threshold:
                            log.debug("Cache semantic hit (%.3f): '%s'", sim, cq)
                            return self._cache[cq]
                except Exception as e:
                    log.warning("Semantic cache lookup failed: %s", e)
        return None

    def set(self, query: str, result: dict, embedder=None) -> None:
        with self._lock:
            self._cache[query] = result
            if embedder:
                try:
                    self._embeddings[query] = embedder.encode([query], show_progress_bar=False)[0]
                except Exception as e:
                    log.warning("Cache embedding failed: %s", e)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
        log.info("Query cache cleared.")


query_cache = QueryCache()
