import os
import sys
import argparse
import json

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_banner():
    print("==================================================================")
    print("    ⚙️  opsFlow: INDUSTRIAL AI MAINTENANCE ARCHITECTURE PLATFORM   ")
    print("==================================================================")

def explain_chromadb():
    from rag import get_chromadb_vs_faiss_explanation
    print(get_chromadb_vs_faiss_explanation())

def explain_bm25():
    from rag import get_bm25_explanation
    print(get_bm25_explanation())

def explain_rrf():
    from rag import get_rrf_explanation
    print(get_rrf_explanation())

def explain_rerank():
    from rag import get_rerank_explanation
    print(get_rerank_explanation())

def explain_faithfulness():
    from rag import get_relevance_vs_faithfulness_explanation
    print(get_relevance_vs_faithfulness_explanation())

def explain_logging():
    from rag import get_source_logging_explanation
    print(get_source_logging_explanation())

def main():
    parser = argparse.ArgumentParser(
        description="opsFlow Unified CLI: A Technical Production ML & RAG Assessment System.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ML Pipeline Actions
    ml_group = parser.add_argument_group("Task 3 — Predictive Maintenance ML Pipeline")
    ml_group.add_argument("--train", action="store_true", help="Ingest data, engineer features, scale, tune and train ML classifiers.")
    ml_group.add_argument("--evaluate", action="store_true", help="Perform holdout evaluations comparing balanced/SMOTE recall strategies and generate SHAP / PR / ROC plots.")
    ml_group.add_argument("--predict", type=str, metavar="'{JSON}'", help="Classify machine failure probability from a JSON telemetry string.")
    
    # RAG Pipeline Actions
    rag_group = parser.add_argument_group("Task 4 — Retrieval-Augmented Generation (RAG) Assistant")
    rag_group.add_argument("--query", type=str, metavar='"<question>"', help="Submit a question to the grounded industrial maintenance RAG assistant.")
    rag_group.add_argument("--groq-key", type=str, metavar="KEY", help="Pass a Groq API Key dynamically (overrides environment variable).")
    rag_group.add_argument("--use-semantic-chunking", action="store_true", help="Apply sentence cosine semantic chunking during knowledge base indexing.")
    rag_group.add_argument("--clear-cache", action="store_true", help="Purge the persistent semantic query cache database.")
    
    # Reviewer Explanations
    exp_group = parser.add_argument_group("Architectural Explanations (Technical Recruiter Review)")
    exp_group.add_argument("--explain-chromadb", action="store_true", help="FAISS library vs SQLite persistent ChromaDB Vector Database.")
    exp_group.add_argument("--explain-bm25", action="store_true", help="Lexical keyword matching vs Dense vector representations.")
    exp_group.add_argument("--explain-rrf", action="store_true", help="Reciprocal Rank Fusion (RRF) vs basic Score Averaging.")
    exp_group.add_argument("--explain-rerank", action="store_true", help="Bi-Encoder embeddings vs deep Cross-Encoder re-ranking.")
    exp_group.add_argument("--explain-faithfulness", action="store_true", help="Subject relevance vs factual faithfulness LLM auditing.")
    exp_group.add_argument("--explain-logging", action="store_true", help="Chunk source logging compliance for safety auditing.")

    # Parse arguments
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()

    # Handle Architectural Explanations (Fast response, zero heavy ML/RAG package imports)
    if args.explain_chromadb:
        explain_chromadb()
        sys.exit(0)
    if args.explain_bm25:
        explain_bm25()
        sys.exit(0)
    if args.explain_rrf:
        explain_rrf()
        sys.exit(0)
    if args.explain_rerank:
        explain_rerank()
        sys.exit(0)
    if args.explain_faithfulness:
        explain_faithfulness()
        sys.exit(0)
    if args.explain_logging:
        explain_logging()
        sys.exit(0)

    # Handle ML Actions
    if args.train:
        print_banner()
        from models.train import run_train
        run_train()
        sys.exit(0)
        
    if args.evaluate:
        print_banner()
        from models.evaluate import run_evaluation
        run_evaluation()
        sys.exit(0)
        
    if args.predict:
        print_banner()
        from models.predict import TelemetryPredictor
        try:
            raw_input = json.loads(args.predict)
        except json.JSONDecodeError as e:
            print(f"Error parsing --predict input JSON: {e}")
            sys.exit(1)
        predictor = TelemetryPredictor()
        res = predictor.predict(raw_input)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    # Handle RAG Actions
    if args.clear_cache:
        print_banner()
        from rag import query_cache
        query_cache.clear()
        sys.exit(0)

    if args.query:
        print_banner()
        from rag import rag_pipeline
        
        print(f"[RAG] Executing grounded generation query: '{args.query}'...")
        if args.use_semantic_chunking:
            print("[RAG] Enabling sentence-level cosine semantic chunking strategy.")
            
        # Initialize pipeline explicitly with user settings
        rag_pipeline.initialize_pipeline(use_semantic_chunking=args.use_semantic_chunking)
        
        # Run query
        res = rag_pipeline.run_query(args.query, groq_api_key=args.groq_key)
        
        # Display output beautifully
        print("\n" + "="*66)
        print(f"QUESTION: {args.query}")
        print("="*66)
        
        if res.get("blocked"):
            print(f"⚠️  QUERY BLOCKED: {res.get('answer')}")
        else:
            print("Retrieved Reference Manual Chunks:")
            for idx, c in enumerate(res.get("retrieved_chunks", [])):
                print(f" [{idx+1}] Doc: {c['doc_name']:<24} | Chunk ID: {c['chunk_index']:<2} | Words: {c['start_word']:>3}-{c['end_word']:<3} | Similarity Score: {c.get('score', 0.0):.4f}")
            
            print("\nGenerated Grounded Answer:")
            print("-" * 66)
            print(res.get("answer"))
            print("-" * 66)
            
            faith = res.get("faithfulness", {})
            print(f"\nFactual Faithfulness Audit:")
            print(f" • Faithful to Context: {'✅ YES' if faith.get('faithful') else '❌ NO'}")
            print(f" • Grounding Score    : {faith.get('score', 0.0)*100:.1f}%")
            print(f" • Auditor Verdict    : \"{faith.get('verdict')}\"")
            if faith.get("unsupported_claims"):
                print(f" • Hallucinated Claims: {faith.get('unsupported_claims')}")
                
            print(f" • Query Cache Hit    : {'✅ YES' if res.get('cached') else '❌ NO'}")
            print(f" • Retrieval Confidence: {res.get('confidence_score', 0.0):.4f}")
            
        print("="*66)
        sys.exit(0)

if __name__ == "__main__":
    main()
