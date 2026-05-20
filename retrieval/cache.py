import threading
import numpy as np

class QueryCache:
    """
    Thread-safe query cache storing query results.
    Supports both exact match and semantic similarity matching on queries.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}  # Map: query_text -> results_dict
        self._embeddings = {}  # Map: query_text -> embedding_vector

    def get(self, query: str, embedder=None, similarity_threshold: float = 0.95) -> list[dict] | None:
        """
        Retrieves cached results for a query.
        1. Checks for exact match.
        2. Checks for semantic similarity match if embedder is provided.
        """
        query_clean = query.strip().lower()
        
        with self._lock:
            # 1. Exact Match
            for cached_query in self._cache:
                if cached_query.strip().lower() == query_clean:
                    print(f"[Cache Hit] Exact match found for query: '{query}'")
                    return self._cache[cached_query]
            
            # 2. Semantic Match
            if embedder is not None and len(self._cache) > 0:
                try:
                    query_emb = embedder.encode([query], show_progress_bar=False)[0]
                    
                    for cached_query, cached_emb in self._embeddings.items():
                        # Compute cosine similarity
                        norm1 = np.linalg.norm(query_emb)
                        norm2 = np.linalg.norm(cached_emb)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(query_emb, cached_emb) / (norm1 * norm2)
                        else:
                            sim = 0.0
                            
                        if sim >= similarity_threshold:
                            print(f"[Cache Hit] Semantic match found (similarity {sim:.3f}): '{cached_query}' for query '{query}'")
                            return self._cache[cached_query]
                except Exception as e:
                    print(f"[Cache Warning] Semantic cache check failed: {e}")
                    
        return None

    def set(self, query: str, results: list[dict], embedder=None) -> None:
        """
        Caches the results of a query.
        """
        with self._lock:
            self._cache[query] = results
            if embedder is not None:
                try:
                    self._embeddings[query] = embedder.encode([query], show_progress_bar=False)[0]
                except Exception as e:
                    print(f"[Cache Warning] Failed to compute embedding for cache query: {e}")

    def clear(self) -> None:
        """
        Clears all cached queries.
        """
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            print("[Cache] Cache cleared.")

# Global query cache instance
query_cache = QueryCache()
