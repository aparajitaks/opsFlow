"""Shared typed structures for ML and RAG pipelines."""
from typing import Any, NotRequired, TypedDict


class ChunkMetadata(TypedDict):
    doc_name: str
    chunk_index: int
    start_word: int
    end_word: int
    word_count: NotRequired[int]
    text: str


class RetrievedChunk(ChunkMetadata, total=False):
    score: float
    cross_score: float
    pre_rerank_rank: int
    post_rerank_rank: int


class FaithfulnessResult(TypedDict, total=False):
    faithful: bool
    score: float
    verdict: str
    unsupported_claims: list[str]


class RAGQueryResult(TypedDict, total=False):
    answer: str
    retrieved_chunks: list[dict[str, Any]]
    faithfulness: FaithfulnessResult
    confidence_score: float
    cached: bool
    blocked: bool


class PredictionResult(TypedDict, total=False):
    prediction: int
    probability: float
    model_used: str
    status: str
    explanation: str
    error: str
