from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from services.rag_service import rag_service
from retrieval.cache import query_cache

router = APIRouter(prefix="/api", tags=["RAG Query Engine"])

@router.post("/query", response_model=QueryResponse)
async def query_rag_engine(req: QueryRequest):
    """
    Run RAG query answering execution. Matches incoming queries against semantic
    and keyword indexes, re-ranks using a Cross-Encoder, calls Groq LLM,
    and performs factual faithfulness checks.
    """
    try:
        # Run query through RAG Orchestrator
        res = rag_service.run_query(req.query, req.groq_api_key)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal RAG query error: {str(e)}")

@router.post("/clear_cache")
async def clear_query_cache():
    """
    Clears the query semantic/exact cache.
    """
    query_cache.clear()
    return {"status": "success", "message": "Query cache cleared successfully."}
