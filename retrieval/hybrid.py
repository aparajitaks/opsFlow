from retrieval.bm25 import bm25_search
from retrieval.vector_store import semantic_retrieve

def hybrid_retrieve(query: str, embedder, collection, bm25_index, chunks: list[dict], top_k: int = 10, rrf_k: int = 60) -> list[dict]:
    """
    Merges semantic search (ChromaDB) and BM25 search results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Run semantic search -> returns top_k chunks
    semantic_chunks = semantic_retrieve(query, embedder, collection, k=top_k)
    
    # 2. Run BM25 search -> returns top_k elements in the format [{"chunk": c, "bm25_score": score}]
    bm25_results = bm25_search(query, bm25_index, chunks, top_k=top_k)
    
    # Standardize BM25 output chunks list
    bm25_chunks = []
    for item in bm25_results:
        c = item["chunk"].copy()
        c["bm25_score"] = item["bm25_score"]
        bm25_chunks.append(c)
        
    # Map from chunk_index to a combined dictionary
    fused_map = {}
    
    # Build maps of index -> rank (1-indexed)
    semantic_rank_map = {c["chunk_index"]: idx + 1 for idx, c in enumerate(semantic_chunks)}
    bm25_rank_map = {c["chunk_index"]: idx + 1 for idx, c in enumerate(bm25_chunks)}
    
    # Retrieve base chunk structures from semantic search
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
        
    # Retrieve or augment base chunk structures from BM25 search
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
            
    # Calculate RRF score for every chunk in either list
    for idx, c in fused_map.items():
        sem_rank = semantic_rank_map.get(idx, float('inf'))
        bm_rank = bm25_rank_map.get(idx, float('inf'))
        
        score_sem = 1.0 / (rrf_k + sem_rank) if sem_rank != float('inf') else 0.0
        score_bm = 1.0 / (rrf_k + bm_rank) if bm_rank != float('inf') else 0.0
        
        c["rrf_score"] = score_sem + score_bm
        # Set main score to rrf_score for compatibility
        c["score"] = c["rrf_score"]
        
    # Sort descending by rrf_score
    fused_chunks = list(fused_map.values())
    fused_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return fused_chunks[:top_k]

def print_hybrid_comparison(query: str, semantic_chunks: list[dict], bm25_results: list[dict], hybrid_chunks: list[dict]):
    """
    Prints a comparison showing which chunks came from semantic only,
    BM25 only, or both.
    """
    semantic_ids = {c["chunk_index"] for c in semantic_chunks}
    bm25_ids = {item["chunk"]["chunk_index"] for item in bm25_results}
    
    semantic_only = []
    bm25_only = []
    both = []
    
    for c in hybrid_chunks:
        idx = c["chunk_index"]
        desc = f"'{c['doc_name']}' (Chunk {idx})"
        if idx in semantic_ids and idx in bm25_ids:
            both.append(desc)
        elif idx in semantic_ids:
            semantic_only.append(desc)
        elif idx in bm25_ids:
            bm25_only.append(desc)
            
    print("\n[Hybrid Retrieval Fusion Breakdown]")
    print(f" - Semantic Search Only retrieved: {', '.join(semantic_only) if semantic_only else 'None'}")
    print(f" - BM25 Keyword Search Only retrieved: {', '.join(bm25_only) if bm25_only else 'None'}")
    print(f" - Retrieved by BOTH (Intersection):  {', '.join(both) if both else 'None'}")

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
