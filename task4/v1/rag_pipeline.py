import os
import sys
import time
import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# =================================================================
# STEP 1 — Maintenance Documents Ingestion
# =================================================================
def verify_documents_exist(docs_dir: str):
    """
    Step 1: Ensure all 5 text maintenance documents exist in the docs folder.
    """
    required_docs = [
        "maintenance_guide.txt",
        "equipment_manual.txt",
        "troubleshooting_faq.txt",
        "safety_procedures.txt",
        "preventive_maintenance.txt"
    ]
    print("\n=================================================================")
    print("      STEP 1: CREATING & INGESTING MAINTENANCE DOCUMENTS        ")
    print("=================================================================")
    
    os.makedirs(docs_dir, exist_ok=True)
    
    missing_docs = []
    for doc in required_docs:
        doc_path = os.path.join(docs_dir, doc)
        if not os.path.exists(doc_path):
            missing_docs.append(doc)
        else:
            # Print physical file stats
            words_count = len(open(doc_path, 'r', encoding='utf-8').read().split())
            print(f"Loaded Doc: {doc:<28} | Word Count: {words_count:<4} | Path: {doc_path}")
            
    if missing_docs:
        print(f"Warning: The following required documents are missing: {missing_docs}")
        sys.exit(1)

# =================================================================
# STEP 2 — Document Chunking
# =================================================================
def chunk_documents(docs_dir: str, chunk_size: int = 300):
    """
    Step 2: Fixed-size word-based chunking of text files.
    Groups text into chunks of exactly 300 words with no overlap.
    """
    print("\n=================================================================")
    print("      STEP 2: RUNNING DOCUMENT CHUNKING                         ")
    print("=================================================================")
    
    chunks = []
    filenames = sorted([f for f in os.listdir(docs_dir) if f.endswith('.txt')])
    
    for filename in filenames:
        file_path = os.path.join(docs_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        words = text.split()
        doc_words = len(words)
        
        # Group into blocks of 300 words (no overlap)
        doc_chunk_idx = 0
        for i in range(0, doc_words, chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_metadata = {
                "doc_name": filename,
                "chunk_index": len(chunks), # global chunk index
                "doc_chunk_index": doc_chunk_idx,
                "word_count": len(chunk_words),
                "text": chunk_text
            }
            chunks.append(chunk_metadata)
            doc_chunk_idx += 1
            
    print(f"Successfully processed {len(filenames)} documents.")
    print(f"Total number of chunks created: {len(chunks)}")
    
    # -------------------------------------------------------------
    # Why Chunking is Necessary Explanation
    # -------------------------------------------------------------
    print("\n--- WHY CHUNKING IS NECESSARY ---")
    print("1. Embedding Model Constraints: Modern dense transformer models (such as all-MiniLM-L6-v2)")
    print("   have a strict maximum sequence length (usually 256 or 512 tokens). Passing an entire")
    print("   multi-page document causes silent truncation, losing critical troubleshooting or safety details.")
    print("2. Avoid Semantic Dilution: Embedding a massive document generates a single vector that tries")
    print("   to represent everything. This 'blurs' the coordinates, yielding weak similarity matches.")
    print("   Chunking isolates specific sub-topics into dense, specialized semantic vectors.")
    print("3. LLM Context Limitations & Token Cost: Feeding a complete user manual to an LLM for every")
    print("   query spikes API latencies, costs, and introduces the 'lost-in-the-middle' retrieval issue.")
    
    return chunks

# =================================================================
# STEP 3 — Generate Embeddings
# =================================================================
def generate_embeddings(chunks: list, model_name: str = "all-MiniLM-L6-v2"):
    """
    Step 3: Instantiates the local SentenceTransformer model and computes
    embeddings for all text chunks.
    """
    print("\n=================================================================")
    print("      STEP 3: COMPUTING CHUNK EMBEDDINGS LOCALLY                ")
    print("=================================================================")
    
    print(f"Loading local SentenceTransformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    chunk_texts = [c["text"] for c in chunks]
    print(f"Computing embeddings for {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    
    # Extract dimensions
    embedding_dim = embeddings.shape[1]
    print(f"Embedding Generation Complete!")
    print(f"Embedding Vector Matrix Shape: {embeddings.shape}")
    print(f"Embedding Vector Dimension:    {embedding_dim} (Expected: 384 for all-MiniLM-L6-v2)")
    
    # -------------------------------------------------------------
    # What an Embedding is Explanation
    # -------------------------------------------------------------
    print("\n--- WHAT IS A TEXT EMBEDDING? ---")
    print("An embedding is a dense mathematical vector that represents the semantic meaning of a text segment.")
    print("The model maps words and sentences into a continuous high-dimensional vector space (e.g., 384 dimensions).")
    print("Because the model is trained on diverse corporate and general linguistics data, it locates terms of similar")
    print("semantic contexts (like 'excessive winding heat' and 'motor overheating') at close physical proximity")
    print("within this vector space, enabling conceptual search beyond simple keyword matching.")
    
    return model, embeddings

# =================================================================
# STEP 4 — FAISS Vector Store with Cosine Similarity
# =================================================================
def build_faiss_index(embeddings: np.ndarray):
    """
    Step 4: Normalizes vectors and stores them in a FAISS inner product index
    to resolve similarity searches via Cosine Similarity.
    """
    print("\n=================================================================")
    print("      STEP 4: INITIALIZING FAISS VECTOR STORAGE                  ")
    print("=================================================================")
    
    # L2 Normalization makes Inner Product (IP) search identical to Cosine Similarity
    print("Applying L2 Normalization to embedding vectors...")
    embeddings_normalized = embeddings.copy().astype('float32')
    faiss.normalize_L2(embeddings_normalized)
    
    dim = embeddings.shape[1]
    
    # Build IndexFlatIP
    print("Building FAISS IndexFlatIP (Inner Product) index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_normalized)
    
    print(f"Total vectors stored in the FAISS index: {index.ntotal}")
    
    # -------------------------------------------------------------
    # Why FAISS is Preferred over Simple For-Loops Explanation
    # -------------------------------------------------------------
    print("\n--- WHY FAISS IS PREFERRED OVER SIMPLE FOR-LOOPS ---")
    print("1. Operational Speed & Scale: A naive for-loop similarity scan runs at O(N * D) complexity.")
    print("   As files grow, linearly scanning all vectors stalls search performance. FAISS is written in C++")
    print("   and utilizes multi-threaded SIMD instructions to calculate cosine distances in microseconds.")
    print("2. Approximate Nearest Neighbors (ANN): FAISS provides powerful index architectures (like HNSW, IVF,")
    print("   and Product Quantization) which trade absolute precision for massive speed boosts, achieving logarithmic")
    print("   search time O(log N) suitable for billion-scale databases.")
    
    return index

# =================================================================
# STEP 5 & 6 — Retrieve, Generate & Log Audit Trail
# =================================================================
def execute_rag_query(
    query: str, 
    model, 
    index: faiss.Index, 
    chunks: list, 
    anthropic_client: Anthropic,
    outputs_dir: str = "outputs",
    k: int = 3
):
    """
    Step 5 & Step 6: 
    - Retrieve top k matching chunks using FAISS.
    - Generate answer via Anthropic API (Claude-3-5-sonnet).
    - Append detailed retrieval audit log to retrieved_chunks.log.
    """
    print(f"\n=================================================================")
    print(f"  PROCESSING QUERY: \"{query}\"")
    print(f"=================================================================")
    
    # 1. Embed query and normalize
    query_emb = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_emb)
    
    # 2. Search FAISS index
    scores, indices = index.search(query_emb, k)
    
    # Extract matching chunks
    retrieved_chunks_list = []
    print("\n--- RETRIEVED CONTEXT CHUNKS ---")
    for i in range(k):
        chunk_idx = int(indices[0][i])
        score = float(scores[0][i])
        chunk = chunks[chunk_idx]
        
        retrieved_chunks_list.append({
            "doc_name": chunk["doc_name"],
            "chunk_index": chunk["chunk_index"],
            "score": score,
            "text": chunk["text"]
        })
        
        print(f"Match {i+1} | Doc: {chunk['doc_name']} | Global Index: {chunk['chunk_index']} | Cosine Similarity Score: {score:.4f}")
        print(f"Excerpt: \"{chunk['text'][:150]}...\"\n")
        
    # 3. Format Prompt
    context_str = ""
    for idx, c in enumerate(retrieved_chunks_list):
        context_str += f"[{idx+1}] Doc: {c['doc_name']} | Index: {c['chunk_index']}\nContent: {c['text']}\n\n"
        
    prompt = f"""You are a maintenance assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context_str.strip()}

Question: {query}

Answer:"""

    # 4. Generate Answer via Anthropic SDK
    print("Calling Anthropic Claude API for Answer Generation...")
    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", # Claude 3.5 Sonnet
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = message.content[0].text
    except Exception as e:
        answer = f"Error during generation: {e}"
        print(f"API Error: {e}")
        
    print("\n--- CLAUDE GENERATED ANSWER ---")
    print(answer)
    print("-------------------------------------------------------------")
    
    # 5. Log retrieved chunks audit trail (Step 6)
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, "retrieved_chunks.log")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"=====================================\n")
        f.write(f"Query: {query}\n")
        f.write(f"Timestamp: {timestamp}\n")
        for idx, c in enumerate(retrieved_chunks_list):
            f.write(f"-------------------------------------\n")
            f.write(f"Chunk {idx+1} | Doc: {c['doc_name']} | Index: {c['chunk_index']} | Score: {c['score']:.4f}\n")
            f.write(f"{c['text']}\n")
        f.write(f"-------------------------------------\n")
        f.write(f"=====================================\n\n")
        
    print(f"Saved audit log to: {log_path}")
    return answer

# =================================================================
# MAIN PIPELINE ORCHESTRATOR & DEMO RUNS
# =================================================================
def main():
    # Folder paths relative to task4/v1
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "docs")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    # 1. Step 1 Documents Check
    verify_documents_exist(docs_dir)
    
    # 2. Step 2 Word-based Chunking
    chunks = chunk_documents(docs_dir, chunk_size=300)
    
    # 3. Step 3 Generate Embeddings
    model, embeddings = generate_embeddings(chunks)
    
    # 4. Step 4 FAISS index build
    index = build_faiss_index(embeddings)
    
    # Ensure Anthropic API Key is active
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[WARNING] ANTHROPIC_API_KEY environment variable is not set!")
        print("Please export your API key: export ANTHROPIC_API_KEY='your-key'")
        print("The pipeline will run local retrieval, but LLM generation will be skipped/mocked.")
        api_key = "dummy-key" # Avoid instantiation crash
        
    anthropic_client = Anthropic(api_key=api_key)
    
    # 5. Run the 3 Demo Queries
    demo_queries = [
        "What are the signs of motor overheating?",
        "How often should lubrication be performed on rotating equipment?",
        "What should I do if the rotational speed exceeds the safe threshold?"
    ]
    
    print("\n=================================================================")
    print("      EXECUTING TASK 4 DEMO QUERIES                              ")
    print("=================================================================")
    
    for query in demo_queries:
        execute_rag_query(query, model, index, chunks, anthropic_client, outputs_dir)
        time.sleep(1) # Prevent rapid rate limits

if __name__ == "__main__":
    main()
