from pydantic import BaseModel, Field
from typing import List, Optional, Any

# RAG Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The maintenance query text.")
    groq_api_key: Optional[str] = Field(None, description="Optional custom Groq API Key.")

class ChunkMetadata(BaseModel):
    doc_name: str
    chunk_index: int
    start_word: int
    end_word: int
    word_count: int
    score: float
    text: str

class FaithfulnessResult(BaseModel):
    faithful: bool
    score: float
    unsupported_claims: List[str]
    verdict: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[ChunkMetadata]
    faithfulness: FaithfulnessResult
    confidence_score: float
    cached: bool
    blocked: bool

# ML Schemas
class PredictRequest(BaseModel):
    Type: str = Field("L", description="Equipment Type (H, L, M)")
    Air_temperature: float = Field(300.0, description="Air temperature [K]")
    Process_temperature: float = Field(310.0, description="Process temperature [K]")
    Rotational_speed: float = Field(1500.0, description="Rotational speed [rpm]")
    Torque: float = Field(40.0, description="Torque [Nm]")
    Tool_wear: float = Field(100.0, description="Tool wear [min]")
    model_type: str = Field("random_forest", description="Model type: 'random_forest' or 'logistic_regression'")

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_used: str
    engineered_features: dict[str, float]

class ModelStatusResponse(BaseModel):
    run_timestamp: Optional[str] = None
    best_model: str
    best_f1: float
    best_roc_auc: float
    best_params: dict[str, Any]
    top_features: List[str]
    failure_rate_in_dataset: float
    status: Optional[str] = None

# System Schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ReindexResponse(BaseModel):
    status: str
    chunks_count: int
