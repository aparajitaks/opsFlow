import os
import sys
import argparse
import json
import time

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
    rag_group.add_argument("--interactive", "-i", action="store_true", help="Start a continuous interactive RAG terminal loop.")
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

    if args.interactive:
        print_banner()
        from rag import rag_pipeline
        from core.config import settings

        print("[RAG] Starting consolidated RAG pipeline initialization for interactive session...")
        if args.use_semantic_chunking:
            print("[RAG] Enabling sentence-level cosine semantic chunking strategy.")

        # Initialize the pipeline ONCE here, avoiding reloading embedding/rerank models inside loop
        rag_pipeline.initialize_pipeline(use_semantic_chunking=args.use_semantic_chunking)

        # Prepare metrics and history trackers
        history = []
        history_transcripts = []
        session_start_time = time.time()

        print("\n\033[1;32m==================================================================")
        print(" 🛠️  opsFlow: INDUSTRIAL MAINTENANCE RAG ASSISTANT INTERACTIVE MODE")
        print("==================================================================")
        print(" • Type your maintenance question below and press Enter.")
        print(" • Special Commands:")
        print("   - 'exit', 'quit', or 'q'  -> Exit session gracefully.")
        print("   - '/clear'                 -> Clear the screen.")
        print("   - '/history'               -> Show query history in this session.")
        print("   - '/save'                  -> Save session conversation log to file.")
        print("   - '/help'                  -> Show this guide again.")
        print("==================================================================\033[0m")

        while True:
            try:
                # Prompt with premium cyan terminal color formatting
                query = input("\n\033[1;36m⚙️ opsFlow>\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n\033[1;33m[Session Interrupted] Safely terminating session. Goodbye!\033[0m")
                break

            if not query:
                continue

            # Graceful exit triggers
            if query.lower() in ("exit", "quit", "q"):
                session_duration = time.time() - session_start_time
                print(f"\n\033[1;32m[Session Terminated] Session duration: {session_duration:.1f}s | Safe shutdown of RAG models. Goodbye!\033[0m")
                break

            # Process special slash commands
            if query.startswith("/"):
                cmd = query.lower()
                if cmd in ("/help", "help"):
                    print("\n\033[1;35m--- SPECIAL COMMAND REFERENCE ---")
                    print("  /help      - Display this help reference.")
                    print("  /clear     - Clear the terminal screen.")
                    print("  /history   - View queried questions in this session.")
                    print("  /save      - Save current conversation transcript to logs/ directory.")
                    print("  exit/quit  - Terminate interactive session.\033[0m")
                elif cmd == "/clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_banner()
                    print("\033[1;32mTerminal cleared. opsFlow RAG interactive session is running.\033[0m")
                elif cmd == "/history":
                    if not history:
                        print("\n\033[1;33mNo queries recorded in this session yet.\033[0m")
                    else:
                        print("\n\033[1;35m--- SESSION QUERY HISTORY ---")
                        for idx, q in enumerate(history, 1):
                            print(f"  {idx}. {q}")
                        print("\033[0m")
                elif cmd == "/save":
                    log_name = f"session_transcript_{int(time.time())}.txt"
                    log_path = settings.LOGS_DIR / log_name
                    try:
                        with open(log_path, "w", encoding="utf-8") as f:
                            f.write("==================================================================\n")
                            f.write("       ⚙️ opsFlow: INDUSTRIAL RAG INTERACTIVE SESSION LOG\n")
                            f.write("==================================================================\n")
                            f.write(f"Session Start: {time.asctime(time.localtime(session_start_time))}\n")
                            f.write(f"Session Saved: {time.asctime(time.localtime(time.time()))}\n")
                            f.write(f"Total Queries: {len(history)}\n")
                            f.write("==================================================================\n\n")
                            for idx, (q, a) in enumerate(history_transcripts, 1):
                                f.write(f"[{idx}] USER QUERY:\n{q}\n\n")
                                f.write(f"[{idx}] RAG SYSTEM ANSWER:\n{a}\n")
                                f.write("-" * 66 + "\n\n")
                        print(f"\n\033[1;32m✅ Conversation log successfully saved to: {log_path}\033[0m")
                    except Exception as e:
                        print(f"\n\033[1;31m⚠️ Error saving conversation log: {e}\033[0m")
                else:
                    print(f"\n\033[1;31m⚠️ Unknown command '{query}'. Type '/help' for a list of valid commands.\033[0m")
                continue

            # Execute RAG query with timing metrics and robust error handling
            print("\033[1;33m⌛ Grounding query against reference manuals...\033[0m")
            start_time = time.time()
            try:
                # Reuses the exact same retriever/generator session instantiated in memory
                res = rag_pipeline.run_query(query, groq_api_key=args.groq_key)
            except Exception as e:
                print(f"\n\033[1;31m⚠️ Failure during processing: {e}\033[0m")
                continue

            elapsed = time.time() - start_time
            history.append(query)
            history_transcripts.append((query, res.get("answer", "")))

            # Print output block dynamically
            print("\n\033[1;34m" + "="*66 + "\033[0m")
            if res.get("blocked"):
                print(f"\033[1;31m⚠️  QUERY BLOCKED BY PROMPT INJECTION FIREWALL:\033[0m")
                print(f"\033[31m{res.get('answer')}\033[0m")
            else:
                print("\033[1;35mRetrieved Reference Manual Chunks (Hybrid RRF + Cross-Encoder):\033[0m")
                for idx, c in enumerate(res.get("retrieved_chunks", [])):
                    print(f" \033[33m[{idx+1}]\033[0m Doc: \033[1m{c['doc_name']:<24}\033[0m | Chunk ID: {c['chunk_index']:<2} | Words: {c['start_word']:>3}-{c['end_word']:<3} | Score: \033[32m{c.get('score', 0.0):.4f}\033[0m")
                
                print("\n\033[1;32mGenerated Grounded Answer:\033[0m")
                print("-" * 66)
                print(res.get("answer"))
                print("-" * 66)
                
                faith = res.get("faithfulness", {})
                is_faithful = faith.get("faithful")
                faithful_color = "\033[1;32m" if is_faithful else "\033[1;31m"
                print(f"\n\033[1mFactual Faithfulness Audit:\033[0m")
                print(f" • Grounded in Context  : {faithful_color}{'✅ YES' if is_faithful else '❌ NO'}\033[0m")
                print(f" • Grounding Score      : \033[32m{faith.get('score', 0.0)*100:.1f}%\033[0m")
                print(f" • Auditor Verdict      : \033[3m\"{faith.get('verdict')}\"\033[0m")
                if faith.get("unsupported_claims"):
                    print(f" • Hallucinated Claims  : \033[31m{faith.get('unsupported_claims')}\033[0m")
                    
                print(f" • Query Cache Hit      : {'✅ YES' if res.get('cached') else '❌ NO'}")
                print(f" • Retrieval Confidence  : \033[32m{res.get('confidence_score', 0.0):.4f}\033[0m")
                print(f" • RAG Retrieval Latency : \033[35m{elapsed*1000:.2f} ms\033[0m")
                
            print("\033[1;34m" + "="*66 + "\033[0m")
            
        sys.exit(0)

if __name__ == "__main__":
    main()
