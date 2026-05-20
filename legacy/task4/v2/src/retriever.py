def retrieve(query: str, embedder, collection, k: int = 3) -> list[dict]:
    """
    Step 5: Embeds the incoming user query and searches ChromaDB for
    the top k most semantically similar chunks.
    """
    # 1. Embed query locally using SentenceTransformer
    query_vector = embedder.encode([query])[0].tolist()
    
    # 2. Query ChromaDB collection
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )
    
    retrieved_chunks = []
    
    if results and "documents" in results and len(results["documents"]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[0.0] * len(documents)])[0]
        
        for idx in range(len(documents)):
            meta = metadatas[idx]
            # ChromaDB default distance metric is L2 squared.
            # To get an intuitive similarity score: score = 1.0 - distance (or L2 derived distance)
            distance = float(distances[idx])
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
