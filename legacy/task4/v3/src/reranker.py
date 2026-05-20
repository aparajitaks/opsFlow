from sentence_transformers import CrossEncoder

def get_rerank_model() -> CrossEncoder:
    """
    Loads cross-encoder model ms-marco-MiniLM-L-6-v2 once at startup.
    """
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_n: int = 3) -> list[dict]:
    """
    Step 3: Scores each candidate (query, text) pair using a CrossEncoder
    and returns the top_n re-ranked chunks.
    """
    if not chunks:
        return []
        
    # Store pre-rerank ranks (1-indexed based on hybrid rank)
    for idx, c in enumerate(chunks):
        c["pre_rerank_rank"] = idx + 1
        
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    
    # Pair scores with chunks
    for idx, score in enumerate(scores):
        chunks[idx]["cross_score"] = float(score)
        # Store for compatibility with print elements
        chunks[idx]["score"] = float(score)
        
    # Sort descending by cross_score
    reranked = sorted(chunks, key=lambda x: x["cross_score"], reverse=True)
    
    # Store post-rerank ranks (1-indexed based on sorted order)
    for idx, c in enumerate(reranked):
        c["post_rerank_rank"] = idx + 1
        
    return reranked[:top_n]

def print_rerank_comparison(query: str, pre_rerank_chunks: list[dict], post_rerank_chunks: list[dict]):
    """
    Prints a comparison of document rank shifts before and after re-ranking.
    """
    print("\n[Cross-Encoder Re-Ranking Position Shifts]")
    
    # Map post-rerank chunks by chunk_index to easily check where they ended up
    post_map = {c["chunk_index"]: c for c in post_rerank_chunks}
    
    for idx, c in enumerate(pre_rerank_chunks):
        c_idx = c["chunk_index"]
        doc = c["doc_name"]
        
        pre_pos = idx + 1
        if c_idx in post_map:
            post_pos = post_map[c_idx]["post_rerank_rank"]
            if pre_pos != post_pos:
                print(f" • '{doc}' (Chunk {c_idx}) moved from rank {pre_pos} → rank {post_pos} (RE-RANKED TOP 3)")
            else:
                print(f" • '{doc}' (Chunk {c_idx}) stayed at rank {pre_pos} (RE-RANKED TOP 3)")
        else:
            # Shifted outside top 3
            print(f" • '{doc}' (Chunk {c_idx}) moved from rank {pre_pos} → shifted out of Top 3")

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
