import os
import datetime

def log_query(query: str, retrieved_chunks: list[dict], answer: str, log_path: str, pre_rerank_chunks: list[dict], faithfulness_res: dict):
    """
    Step 4: Appends query, retrieved chunk metadata, hybrid retrieval method,
    pre- and post-rerank rank distributions, faithfulness audit metrics,
    and the generated grounded response to outputs/retrieved_chunks.log.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    k = len(retrieved_chunks)
    
    pre_ranks_str = "[" + ", ".join([f"{c['doc_name']} rank{c['pre_rerank_rank']}" for c in retrieved_chunks]) + "]"
    post_ranks_str = "[" + ", ".join([f"{c['doc_name']} rank{c['post_rerank_rank']}" for c in retrieved_chunks]) + "]"
    
    faithful_str = "Yes" if faithfulness_res.get("faithful") else "No"
    faith_score = faithfulness_res.get("score", 0.0)
    faith_verdict = faithfulness_res.get("verdict", "")
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("=====================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Total chunks retrieved: {k}\n")
        f.write("-------------------------------------\n")
        f.write("Retrieval Method : Hybrid (BM25 + Semantic, RRF fusion)\n")
        f.write(f"Pre-rerank Ranks : {pre_ranks_str}\n")
        f.write(f"Post-rerank Ranks: {post_ranks_str}\n")
        f.write(f"Faithfulness     : {faithful_str} ({faith_score}) — \"{faith_verdict}\"\n")
        f.write("-------------------------------------\n")
        
        for idx, c in enumerate(retrieved_chunks):
            preview = c["text"].replace("\n", " ")[:100]
            f.write(f"[Chunk {idx+1}]\n")
            f.write(f"  Document : {c['doc_name']}\n")
            f.write(f"  Chunk ID : {c['chunk_index']}\n")
            f.write(f"  Words    : {c['start_word']}–{c['end_word']}\n")
            # Show cross-encoder re-rank score
            f.write(f"  Score    : {c['score']:.4f}\n")
            f.write(f"  Preview  : {preview}...\n")
            f.write("-------------------------------------\n")
            
        f.write("[LLM Answer]\n")
        f.write(f"{answer}\n")
        f.write("=====================================\n\n")

def print_sources_to_terminal(query: str, retrieved_chunks: list[dict], answer: str):
    """
    Step 4: Prints the retrieval results and sources directly to the CLI.
    """
    print("\n-------------------------------------------------------------")
    print(f"QUESTION: {query}")
    print("-------------------------------------------------------------")
    print("Retrieved Sources used for Grounding:")
    for idx, c in enumerate(retrieved_chunks):
        print(f" [{idx+1}] Doc: {c['doc_name']:<24} | Chunk ID: {c['chunk_index']:<2} | Words: {c['start_word']:>3}-{c['end_word']:<3} | Similarity: {c['score']:.4f}")
    print("\nGenerated Grounded Answer:")
    print(answer)
    print("-------------------------------------------------------------")

def get_source_logging_explanation() -> str:
    return (
        "--- WHY CHUNK SOURCE LOGGING MATTERS FOR AUDITABILITY ---\n"
        "1. Safety Compliance & Verification: In physical industrial plants, executing incorrect maintenance "
        "   (such as operating out-of-spec torque regimes or skipping isolated LOTO safety locks) can cause physical "
        "   injury or severe equipment damage. Log records prove exactly which operational manual was consulted.\n"
        "2. Hallucination Guardrails: If an LLM response sounds ambiguous or incorrect, engineers can trace "
        "   the prediction back to specific sections in the original manual, allowing rapid auditing and model tuning."
    )
