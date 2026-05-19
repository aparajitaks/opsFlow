import os

def chunk_documents(docs_dir: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Step 1: Performs overlap chunking across all text documents in docs_dir.
    Returns a list of dictionaries, each representing a chunk with text and detailed metadata.
    """
    chunks = []
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
        
    filenames = sorted([f for f in os.listdir(docs_dir) if f.endswith('.txt')])
    
    global_chunk_idx = 0
    for filename in filenames:
        file_path = os.path.join(docs_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        words = text.split()
        n_words = len(words)
        
        step = chunk_size - overlap
        
        # Handle small files
        if n_words <= chunk_size:
            chunk_metadata = {
                "doc_name": filename,
                "chunk_index": global_chunk_idx,
                "start_word": 0,
                "end_word": max(0, n_words - 1),
                "word_count": n_words,
                "text": " ".join(words)
            }
            chunks.append(chunk_metadata)
            global_chunk_idx += 1
            continue
            
        # Standard sliding window with overlap
        for start_idx in range(0, n_words, step):
            end_idx = start_idx + chunk_size
            chunk_words = words[start_idx:end_idx]
            
            if not chunk_words:
                break
                
            chunk_metadata = {
                "doc_name": filename,
                "chunk_index": global_chunk_idx,
                "start_word": start_idx,
                "end_word": start_idx + len(chunk_words) - 1,
                "word_count": len(chunk_words),
                "text": " ".join(chunk_words)
            }
            chunks.append(chunk_metadata)
            global_chunk_idx += 1
            
            # If the current chunk reached the end of the file, stop
            if end_idx >= n_words:
                break
                
    return chunks

def get_overlap_explanation() -> str:
    return (
        "--- WHY OVERLAP CHUNKING IS REQUIRED ---\n"
        "1. Prevents Context Fragmentation: When documents are split arbitrarily, a key fact or multi-step instruction "
        "can be cut exactly down the middle across a boundary. This separates related keywords or facts, preventing "
        "either chunk from having sufficient context to answer a query.\n"
        "2. Concrete Example: In LOTO instructions, a sentence might read:\n"
        "   'Standard LOTO requires cutting electrical power, and [boundary] bleeding residual pneumatic pressure.'\n"
        "   If the user asks 'What are the electrical and pneumatic steps of LOTO?', a fixed-size partition splits this "
        "   instruction. Chunk 1 has only electrical context; Chunk 2 has only pneumatic. A semantic search on either "
        "   chunk alone will miss the full combined procedure. By overlapping by 50 words, the boundary zone is "
        "   retained in both chunks, ensuring the full semantic context is kept whole in at least one retrieved chunk."
    )
