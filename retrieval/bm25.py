from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """
    Tokenizes text by lowercasing and splitting on whitespace.
    """
    return text.lower().split()

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25Okapi index from a list of chunks.
    Tokenizes each chunk's text by lowercasing and splitting on whitespace.
    """
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)

def bm25_search(query: str, bm25_index: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """
    Searches the BM25 index for the top_k closest matching chunks.
    Returns results formatted as [{"chunk": c, "bm25_score": score}].
    """
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    ranked_results = []
    for idx, score in enumerate(scores):
        ranked_results.append({
            "chunk": chunks[idx],
            "bm25_score": float(score)
        })
        
    ranked_results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return ranked_results[:top_k]

def get_bm25_explanation() -> str:
    return (
        "--- WHY BM25 CATCHES QUERIES THAT SEMANTIC SEARCH MISSES ---\n"
        "1. Precise Keyword Matching: BM25 is a term-frequency/inverse document frequency (TF-IDF) derived\n"
        "   lexical search algorithm. It scores documents based on exact keyword occurrences.\n"
        "2. Cosine Distance Coordinate Smoothing: Dense transformer models map terms into low-dimensional semantic\n"
        "   embedding spaces. For highly specific industrial codes, technical identifiers, or exact numerical values\n"
        "   (such as error codes like 'ERR-101' or voltage numbers like '480V'), the dense vector representation\n"
        "   smooths these values with surrounding words, yielding a near-zero similarity score for exact numeric lookups.\n"
        "3. Concrete Example:\n"
        "   If a technician queries 'How to resolve ERR-101?', a semantic search model might retrieve chunks related\n"
        "   to general faults (like general warnings or standard errors) due to the similarity of surrounding words\n"
        "   like 'resolve' and 'error'. However, BM25 targets the high-IDF rare keyword 'ERR-101', instantly boosting\n"
        "   the exact troubleshooting manual chunk containing that exact code to the top of the list."
    )
