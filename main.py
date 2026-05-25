"""
main.py — opsFlow unified CLI entry point.

Task 3 — Predictive Maintenance ML Pipeline:
  --train       Ingest data, engineer features, tune and train ML classifiers.
  --evaluate    Holdout evaluation with SHAP, PR/ROC curves, and SMOTE comparison.
  --predict     Classify a single telemetry observation from a JSON string.

Task 4 — RAG Maintenance Assistant:
  --query       Single grounded query against the industrial knowledge base.
  --interactive Start a continuous interactive RAG session.
  --groq-key    Override the GROQ_API_KEY environment variable at runtime.
  --use-semantic-chunking  Use cosine-similarity sentence chunking (default: fixed-size).
  --clear-cache Purge the in-memory semantic query cache.
"""
import os
import sys
import argparse
import json
import time

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="opsFlow: Industrial AI Maintenance Assessment System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task 3
    ml_group = parser.add_argument_group("Task 3 — Predictive Maintenance ML Pipeline")
    ml_group.add_argument("--train", action="store_true",
                          help="Train LR and RF classifiers with GridSearchCV + 5-fold CV.")
    ml_group.add_argument("--evaluate", action="store_true",
                          help="Holdout evaluation: SHAP importance, PR/ROC curves, SMOTE recall comparison.")
    ml_group.add_argument("--predict", type=str, metavar="'{JSON}'",
                          help="Predict failure probability from a JSON telemetry string.")

    # Task 4
    rag_group = parser.add_argument_group("Task 4 — RAG Maintenance Assistant")
    rag_group.add_argument("--query", type=str, metavar='"<question>"',
                           help="Submit a question to the grounded maintenance RAG assistant.")
    rag_group.add_argument("--interactive", "-i", action="store_true",
                           help="Start a continuous interactive RAG session.")
    rag_group.add_argument("--groq-key", type=str, metavar="KEY",
                           help="Pass a Groq API key at runtime (overrides .env).")
    rag_group.add_argument("--use-semantic-chunking", action="store_true",
                           help="Use cosine-similarity sentence chunking instead of fixed-size chunking.")
    rag_group.add_argument("--clear-cache", action="store_true",
                           help="Clear the in-memory semantic query cache.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # ── Task 3: ML ────────────────────────────────────────────────────────────

    if args.train:
        from ml.train import run_train
        run_train()
        sys.exit(0)

    if args.evaluate:
        from ml.evaluate import run_evaluation
        run_evaluation()
        sys.exit(0)

    if args.predict:
        from ml.predict import TelemetryPredictor
        try:
            raw_input = json.loads(args.predict)
        except json.JSONDecodeError as e:
            print(f"Error: could not parse --predict JSON: {e}", file=sys.stderr)
            sys.exit(1)
        predictor = TelemetryPredictor()
        result = predictor.predict(raw_input)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # ── Task 4: RAG ───────────────────────────────────────────────────────────

    if args.clear_cache:
        from rag import query_cache
        query_cache.clear()
        print("Query cache cleared.")
        sys.exit(0)

    if args.query:
        from rag import rag_pipeline
        rag_pipeline.initialize_pipeline(use_semantic_chunking=args.use_semantic_chunking)
        res = rag_pipeline.run_query(args.query, groq_api_key=args.groq_key)

        print("\n" + "=" * 60)
        print(f"QUESTION: {args.query}")
        print("=" * 60)

        if res.get("blocked"):
            print(f"BLOCKED: {res.get('answer')}")
        else:
            print("Retrieved Chunks (Hybrid RRF + Cross-Encoder):")
            for i, c in enumerate(res.get("retrieved_chunks", [])):
                print(
                    f"  [{i+1}] {c['doc_name']:<24} | chunk {c['chunk_index']:<3} "
                    f"| words {c['start_word']}-{c['end_word']} | score {c.get('score', 0.0):.4f}"
                )
            print("\nAnswer:")
            print("-" * 60)
            print(res.get("answer"))
            print("-" * 60)

            faith = res.get("faithfulness", {})
            print(f"\nFaithfulness Audit:")
            print(f"  Faithful     : {'YES' if faith.get('faithful') else 'NO'}")
            print(f"  Score        : {faith.get('score', 0.0) * 100:.1f}%")
            print(f"  Verdict      : \"{faith.get('verdict')}\"")
            if faith.get("unsupported_claims"):
                print(f"  Unsupported  : {faith.get('unsupported_claims')}")
            print(f"  Cache hit    : {'YES' if res.get('cached') else 'NO'}")
            print(f"  Confidence   : {res.get('confidence_score', 0.0):.4f}")

        print("=" * 60)
        sys.exit(0)

    if args.interactive:
        from rag import rag_pipeline
        from core.config import settings

        rag_pipeline.initialize_pipeline(use_semantic_chunking=args.use_semantic_chunking)
        history: list[tuple[str, str]] = []
        conversation_history: list[dict] = []

        print("=" * 60)
        print("  opsFlow RAG Interactive Session")
        print("  Type 'exit' or 'quit' to end.")
        print("=" * 60)

        while True:
            try:
                query = input("\nopsFlow> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("Session ended.")
                break

            start = time.time()
            try:
                res = rag_pipeline.run_query(
                    query, groq_api_key=args.groq_key, conversation_history=conversation_history
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

            elapsed_ms = (time.time() - start) * 1000
            history.append((query, res.get("answer", "")))

            if not res.get("blocked"):
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": res.get("answer", "")})

            print("\n" + "=" * 60)
            if res.get("blocked"):
                print(f"BLOCKED: {res.get('answer')}")
            else:
                print("Retrieved Chunks:")
                for i, c in enumerate(res.get("retrieved_chunks", [])):
                    print(
                        f"  [{i+1}] {c['doc_name']:<24} | chunk {c['chunk_index']:<3} "
                        f"| score {c.get('score', 0.0):.4f}"
                    )
                print("\nAnswer:")
                print("-" * 60)
                print(res.get("answer"))
                print("-" * 60)

                faith = res.get("faithfulness", {})
                print(f"Faithful: {'YES' if faith.get('faithful') else 'NO'} "
                      f"({faith.get('score', 0.0)*100:.1f}%) — \"{faith.get('verdict')}\"")
                print(f"Cache: {'HIT' if res.get('cached') else 'MISS'} | "
                      f"Confidence: {res.get('confidence_score', 0.0):.4f} | "
                      f"Latency: {elapsed_ms:.0f}ms")
            print("=" * 60)

        sys.exit(0)


if __name__ == "__main__":
    main()
