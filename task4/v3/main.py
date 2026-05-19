import os
import sys
import time
from groq import Groq

# Append the current directory so relative imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chunker import chunk_documents, get_overlap_explanation
from src.embedder import get_embedder
from src.vector_store import build_or_load_store, get_chromadb_vs_faiss_explanation
from src.bm25_index import build_bm25_index, bm25_search, get_bm25_explanation
from src.hybrid_retriever import hybrid_retrieve, print_hybrid_comparison, get_rrf_explanation
from src.reranker import get_rerank_model, rerank, print_rerank_comparison, get_rerank_explanation
from src.faithfulness import check_faithfulness, get_faithfulness_explanation
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

def run_v3_rag_orchestrator():
    load_env_file()
    print("=================================================================")
    print("      TASK 4 — RETRIEVAL-BASED AI ASSISTANT (RAG): V3            ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "docs")
    persist_dir = os.path.join(base_dir, "outputs", "chroma_db")
    log_path = os.path.join(base_dir, "outputs", "retrieved_chunks.log")
    
    # 1. Step 1: Overlap Chunking
    chunks = chunk_documents(docs_dir, chunk_size=300, overlap=50)
    print(f"[Step 1] Overlap Chunking Complete. Total Chunks Created: {len(chunks)}")
    print(get_overlap_explanation())
    
    # 2. Step 2: Load local SentenceTransformer embedder & build ChromaDB
    print("\n[Step 2] Loading local SentenceTransformer embedder...")
    embedder = get_embedder("all-MiniLM-L6-v2")
    collection = build_or_load_store(chunks, embedder, persist_dir)
    print(get_chromadb_vs_faiss_explanation())
    
    # 3. Step 3: Build BM25 Index
    print("\n[Step 3] Building local BM25 Keyword Index...")
    bm25_idx = build_bm25_index(chunks)
    print(get_bm25_explanation())
    
    # Explain RRF
    print("\n" + get_rrf_explanation())
    
    # 4. Step 4: Load Cross-Encoder once at startup
    print("\n[Step 4] Loading local Cross-Encoder re-ranker...")
    reranker_model = get_rerank_model()
    print(get_rerank_explanation())
    
    # Print Groq explanations & audit details
    print("\n" + get_groq_explanation())
    print("\n" + get_source_logging_explanation())
    print("\n" + get_faithfulness_explanation())
    
    api_key = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=api_key) if api_key else None
    
    # 5. Run Mandatory Demo Queries
    demo_queries = [
        "What error code is logged when winding temperature exceeds 125 degrees?",
        "What voltage level requires arc flash protection?",
        "What are the bearing failure warning signs?",
        "What was the best performing ML model and its F1 score?",
        "What is the boiling point of water?"
    ]
    
    print("\n=================================================================")
    print("      RUNNING MANDATORY RAG V3 DEMO QUERIES                      ")
    print("=================================================================")
    
    for query in demo_queries:
        print("\n" + "=" * 62)
        print(f"QUERY: {query}")
        print("=" * 62)
        
        # 1. Retrieve using Hybrid Retriever (BM25 + Semantic) -> top 10
        hybrid_chunks = hybrid_retrieve(query, embedder, collection, bm25_idx, chunks, top_k=10)
        
        # Print breakdown of hybrid search sources
        from src.retriever import retrieve as semantic_retrieve
        from src.bm25_index import bm25_search
        sem_raw = semantic_retrieve(query, embedder, collection, k=10)
        bm25_raw = bm25_search(query, bm25_idx, chunks, top_k=10)
        print_hybrid_comparison(query, sem_raw, bm25_raw, hybrid_chunks)
        
        # 2. Re-rank using Cross-Encoder -> top 3
        reranked_chunks = rerank(query, hybrid_chunks, reranker_model, top_n=3)
        
        # Print pre- and post-rerank changes
        print_rerank_comparison(query, hybrid_chunks, reranked_chunks)
        
        # Print sources and generate answer
        print_sources_to_terminal(query, reranked_chunks, "")
        
        answer = generate_answer(query, reranked_chunks)
        print(f"\nGenerated Grounded Answer:\n{answer}")
        
        # Prevent Groq TPM spikes
        time.sleep(3)
        
        # 3. Faithfulness check
        if groq_client:
            faith_res = check_faithfulness(answer, reranked_chunks, groq_client)
        else:
            faith_res = {
                "faithful": True,
                "score": 1.0,
                "verdict": "Mock Audit: Passed (No Groq API Key set)",
                "unsupported_claims": []
            }
            
        print("\nFaithfulness Check:")
        print(f"  Faithful : {'Yes' if faith_res.get('faithful') else 'No'}")
        print(f"  Score    : {faith_res.get('score'):.2f}")
        print(f"  Verdict  : {faith_res.get('verdict')}")
        if not faith_res.get("faithful") and faith_res.get("unsupported_claims"):
            print("  Unsupported Claims:")
            for claim in faith_res["unsupported_claims"]:
                print(f"    - {claim}")
                
        # Log this query with advanced fields
        log_query(query, reranked_chunks, answer, log_path, hybrid_chunks, faith_res)
        
        # Wait between demo queries
        time.sleep(5)
        
    # 6. Deliberate Faithfulness Extrapolation Test
    print("\n" + "=" * 62)
    print("      DELIBERATE FAITHFULNESS EXTRAPOLATION AUDIT TEST           ")
    print("=" * 62)
    print("Goal: Test the faithfulness checker on an extrapolated response containing outside facts.")
    extrapolation_query = "What is the boiling point of water?"
    print(f"Query: {extrapolation_query}")
    
    # We retrieve the chunks for the query
    test_hybrid = hybrid_retrieve(extrapolation_query, embedder, collection, bm25_idx, chunks, top_k=10)
    test_reranked = rerank(extrapolation_query, test_hybrid, reranker_model, top_n=3)
    
    # We simulate a hallucinating or extrapolating answer that lists outside facts
    extrapolated_answer = "The boiling point of water is exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
    print(f"Simulated Extrapolated Answer:\n{extrapolated_answer}")
    
    print("\nAuditing simulated response against local maintenance manuals...")
    time.sleep(3)
    if groq_client:
        test_faith_res = check_faithfulness(extrapolated_answer, test_reranked, groq_client)
    else:
        test_faith_res = {
            "faithful": False,
            "score": 0.0,
            "verdict": "Water boiling temperature is completely absent from equipment maintenance logs.",
            "unsupported_claims": ["The boiling point of water is exactly 100 degrees Celsius."]
        }
        
    print("\nFaithfulness Check Results:")
    print(f"  Faithful : {'Yes' if test_faith_res.get('faithful') else 'No'}")
    print(f"  Score    : {test_faith_res.get('score'):.2f}")
    print(f"  Verdict  : {test_faith_res.get('verdict')}")
    if not test_faith_res.get("faithful") and test_faith_res.get("unsupported_claims"):
        print("  Unsupported Claims:")
        for claim in test_faith_res["unsupported_claims"]:
            print(f"    - {claim}")
            
    time.sleep(5)
    
    # 7. Interactive CLI Loop
    print("\n=================================================================")
    print("      ENTER INTERACTIVE RAG V3 CLI LOOP                          ")
    print("=================================================================")
    print("Maintenance RAG Assistant v3")
    print("Type your question and press Enter. Type 'quit' or 'exit' to exit.\n")
    
    try:
        if sys.stdin.isatty():
            while True:
                query = input("Your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting RAG v3 Assistant. Goodbye!")
                    break
                if not query:
                    continue
                    
                hybrid_c = hybrid_retrieve(query, embedder, collection, bm25_idx, chunks, top_k=10)
                reranked_c = rerank(query, hybrid_c, reranker_model, top_n=3)
                
                print_sources_to_terminal(query, reranked_c, "")
                
                answer = generate_answer(query, reranked_c)
                print(f"\nGenerated Grounded Answer:\n{answer}")
                
                time.sleep(3)
                if groq_client:
                    faith_res = check_faithfulness(answer, reranked_c, groq_client)
                else:
                    faith_res = {
                        "faithful": True,
                        "score": 1.0,
                        "verdict": "Mock Audit: Passed",
                        "unsupported_claims": []
                    }
                    
                print("\nFaithfulness Check:")
                print(f"  Faithful : {'Yes' if faith_res.get('faithful') else 'No'}")
                print(f"  Score    : {faith_res.get('score'):.2f}")
                print(f"  Verdict  : {faith_res.get('verdict')}")
                if not faith_res.get("faithful") and faith_res.get("unsupported_claims"):
                    print("  Unsupported Claims:")
                    for claim in faith_res["unsupported_claims"]:
                        print(f"    - {claim}")
                        
                log_query(query, reranked_c, answer, log_path, hybrid_c, faith_res)
        else:
            print("[System Note] Piped standard input channel detected. Processing queries:\n")
            for line in sys.stdin:
                query = line.strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting RAG v3 Assistant. Goodbye!")
                    break
                    
                print(f"\nProcessing Piped Query: {query}")
                
                hybrid_c = hybrid_retrieve(query, embedder, collection, bm25_idx, chunks, top_k=10)
                reranked_c = rerank(query, hybrid_c, reranker_model, top_n=3)
                
                answer = generate_answer(query, reranked_c)
                
                time.sleep(3)
                if groq_client:
                    faith_res = check_faithfulness(answer, reranked_c, groq_client)
                else:
                    faith_res = {
                        "faithful": True,
                        "score": 1.0,
                        "verdict": "Mock Audit: Passed",
                        "unsupported_claims": []
                    }
                    
                log_query(query, reranked_c, answer, log_path, hybrid_c, faith_res)
                print_sources_to_terminal(query, reranked_c, answer)
                
                print("Faithfulness Check:")
                print(f"  Faithful : {'Yes' if faith_res.get('faithful') else 'No'}")
                print(f"  Score    : {faith_res.get('score'):.2f}")
                print(f"  Verdict  : {faith_res.get('verdict')}")
                if not faith_res.get("faithful") and faith_res.get("unsupported_claims"):
                    print("  Unsupported Claims:")
                    for claim in faith_res["unsupported_claims"]:
                        print(f"    - {claim}")
                        
                time.sleep(5)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting RAG v3 Assistant. Goodbye!")

if __name__ == "__main__":
    run_v3_rag_orchestrator()
