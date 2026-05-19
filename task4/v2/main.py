import os
import sys
import time

# Append the v2 directory to the path so absolute/relative src imports work flawlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chunker import chunk_documents, get_overlap_explanation
from src.embedder import get_embedder
from src.vector_store import build_or_load_store, get_chromadb_vs_faiss_explanation
from src.retriever import retrieve
from src.generator import generate_answer, get_groq_explanation
from src.logger import log_query, print_sources_to_terminal, get_source_logging_explanation

def load_env_file():
    # Walk up directories starting from this file's folder to find and load .env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir:
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k:
                            os.environ[k] = v
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

def run_v2_rag_orchestrator():
    load_env_file()
    print("=================================================================")
    print("      TASK 4 — RETRIEVAL-BASED AI ASSISTANT (RAG): V2            ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "docs")
    persist_dir = os.path.join(base_dir, "outputs", "chroma_db")
    log_path = os.path.join(base_dir, "outputs", "retrieved_chunks.log")
    
    # 1. Step 1: Overlap Chunking
    chunks = chunk_documents(docs_dir, chunk_size=300, overlap=50)
    print(f"[Step 1] Overlap Chunking Complete. Total Chunks Created: {len(chunks)}")
    print(get_overlap_explanation())
    
    # 2. Step 2 & 3: Load local embedding model and set up persistent ChromaDB
    print("\n[Step 2 & 3] Loading local SentenceTransformer embedder...")
    embedder = get_embedder("all-MiniLM-L6-v2")
    
    # Build or Load persistent database
    collection = build_or_load_store(chunks, embedder, persist_dir)
    print(get_chromadb_vs_faiss_explanation())
    
    # Print Groq Llama 3 usage detail
    print("\n" + get_groq_explanation())
    
    # Print Source logging significance
    print("\n" + get_source_logging_explanation())
    
    # 3. Step 5: Demo Queries
    demo_queries = [
        "What are the warning signs of bearing failure?",
        "What PPE is required during equipment maintenance?",
        "How do I perform lockout tagout procedure?",
        "What is the meaning of life?"
    ]
    
    print("\n=================================================================")
    print("      RUNNING MANDATORY RAG V2 DEMO QUERIES                      ")
    print("=================================================================")
    
    for query in demo_queries:
        # Pipeline: Retrieve -> Ground -> Generate -> Log
        retrieved = retrieve(query, embedder, collection, k=3)
        answer = generate_answer(query, retrieved)
        log_query(query, retrieved, answer, log_path)
        print_sources_to_terminal(query, retrieved, answer)
        time.sleep(3)
        
    # 4. Interactive CLI Loop
    print("\n=================================================================")
    print("      ENTER INTERACTIVE RAG V2 CLI LOOP                          ")
    print("=================================================================")
    print("Maintenance RAG Assistant v2")
    print("Type your question and press Enter. Type 'quit' or 'exit' to exit.\n")
    
    # We support both interactive keyboard prompts and standard piped streams for easy automated verification
    try:
        if sys.stdin.isatty():
            while True:
                query = input("Your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting RAG v2 Assistant. Goodbye!")
                    break
                if not query:
                    continue
                    
                retrieved = retrieve(query, embedder, collection, k=3)
                answer = generate_answer(query, retrieved)
                log_query(query, retrieved, answer, log_path)
                print_sources_to_terminal(query, retrieved, answer)
        else:
            print("[System Note] Piped standard input channel detected. Processing queries:")
            for line in sys.stdin:
                query = line.strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting RAG v2 Assistant. Goodbye!")
                    break
                print(f"\nProcessing Piped Query: {query}")
                retrieved = retrieve(query, embedder, collection, k=3)
                answer = generate_answer(query, retrieved)
                log_query(query, retrieved, answer, log_path)
                print_sources_to_terminal(query, retrieved, answer)
    except KeyboardInterrupt:
        print("\nExiting RAG v2 Assistant. Goodbye!")

if __name__ == "__main__":
    run_v2_rag_orchestrator()
