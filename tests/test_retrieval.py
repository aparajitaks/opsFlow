import pytest
import numpy as np
from unittest.mock import MagicMock

from retrieval.chunker import split_into_sentences, semantic_chunk_text
from retrieval.bm25 import build_bm25_index, bm25_search
from retrieval.hybrid import hybrid_retrieve
from retrieval.reranker import rerank
from retrieval.cache import QueryCache

def test_sentence_split():
    """Asserts that text is correctly split into sentences on punctuation bounds."""
    text = "This is sentence one. This is sentence two? Yes it is."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is sentence one."
    assert sentences[1] == "This is sentence two?"
    assert sentences[2] == "Yes it is."

def test_bm25_lexical_search():
    """Asserts BM25 indexes and matches technical codes and exact keywords."""
    chunks = [
        {"chunk_index": 0, "text": "Procedure for high-voltage breaker shutdown ERR-101 code."},
        {"chunk_index": 1, "text": "Standard lubricating checklist for rotational gears."},
        {"chunk_index": 2, "text": "Thermal overload threshold is defined at 120C degrees."}
    ]
    
    bm25_index = build_bm25_index(chunks)
    results = bm25_search("ERR-101", bm25_index, chunks, top_k=2)
    
    assert len(results) > 0
    assert results[0]["chunk"]["chunk_index"] == 0
    assert results[0]["bm25_score"] > 0.0

def test_reciprocal_rank_fusion():
    """Validates that RRF ranks intersecting items above unique channel occurrences."""
    # Mock search outcomes
    semantic_results = [
        {"chunk_index": 1, "doc_name": "A", "start_word": 0, "end_word": 10, "word_count": 10, "text": "one", "score": 0.9},
        {"chunk_index": 2, "doc_name": "A", "start_word": 11, "end_word": 20, "word_count": 10, "text": "two", "score": 0.8}
    ]
    
    bm25_results = [
        {"chunk": {"chunk_index": 2, "doc_name": "A", "start_word": 11, "end_word": 20, "word_count": 10, "text": "two"}, "bm25_score": 10.0},
        {"chunk": {"chunk_index": 3, "doc_name": "A", "start_word": 21, "end_word": 30, "word_count": 10, "text": "three"}, "bm25_score": 5.0}
    ]
    
    # Mock dependencies
    embedder = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "documents": [[r["text"] for r in semantic_results]],
        "metadatas": [[{
            "doc_name": r["doc_name"],
            "chunk_index": r["chunk_index"],
            "start_word": r["start_word"],
            "end_word": r["end_word"],
            "word_count": r["word_count"]
        } for r in semantic_results]],
        "distances": [[1.0 - r["score"] for r in semantic_results]]
    }
    
    bm25_index = MagicMock()
    bm25_index.get_scores.return_value = [0.0, 0.0, 10.0, 5.0]
    
    chunks = [
        {"chunk_index": 0, "doc_name": "A", "start_word": 0, "end_word": 0, "word_count": 1, "text": "zero"},
        {"chunk_index": 1, "doc_name": "A", "start_word": 0, "end_word": 10, "word_count": 10, "text": "one"},
        {"chunk_index": 2, "doc_name": "A", "start_word": 11, "end_word": 20, "word_count": 10, "text": "two"},
        {"chunk_index": 3, "doc_name": "A", "start_word": 21, "end_word": 30, "word_count": 10, "text": "three"}
    ]
    
    fused = hybrid_retrieve(
        query="test query",
        embedder=embedder,
        collection=collection,
        bm25_index=bm25_index,
        chunks=chunks,
        top_k=3,
        rrf_k=60
    )
    
    assert len(fused) > 0
    # Chunk index 2 is retrieved by both BM25 (rank 1) and Semantic (rank 2).
    # Its RRF score should be highest, making it the top chunk.
    assert fused[0]["chunk_index"] == 2

def test_query_cache():
    """Checks exact cache hits and semantic similarity cache retrievals."""
    cache = QueryCache()
    mock_results = [{"answer": "Test answer", "retrieved_chunks": []}]
    
    # 1. Test Exact cache setting/getting
    cache.set("What is the power limit?", mock_results)
    hit1 = cache.get("What is the power limit?")
    assert hit1 == mock_results
    
    # Check lowercase case-insensitivity
    hit2 = cache.get("  WHAT is the POWER limit? ")
    assert hit2 == mock_results
    
    # 2. Test Semantic cache getting
    embedder = MagicMock()
    # Return identical vectors to simulate semantic matches
    embedder.encode.return_value = [np.array([1.0, 0.0, 0.0])]
    cache._embeddings["What is the power limit?"] = np.array([1.0, 0.0, 0.0])
    
    hit_semantic = cache.get("Verify the power boundaries?", embedder=embedder, similarity_threshold=0.90)
    assert hit_semantic == mock_results
