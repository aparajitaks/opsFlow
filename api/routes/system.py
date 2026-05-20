import os
import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from api.schemas import HealthResponse, ReindexResponse
from services.rag_service import rag_service
from core.config import settings

router = APIRouter(tags=["System & Operations"])

class ReindexRequest(BaseModel):
    use_semantic_chunking: bool = False

@router.get("/health", response_model=HealthResponse)
async def check_health():
    """
    Returns system status and current ISO timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.post("/api/system/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(req: ReindexRequest):
    """
    Forces the RAG engine to clean and reload all documentation files from settings.DOCS_DIR
    and re-build the ChromaDB vector database and BM25 index.
    """
    try:
        rag_service.initialize_pipeline(force=True, use_semantic_chunking=req.use_semantic_chunking)
        return {
            "status": "success",
            "chunks_count": len(rag_service._chunks) if rag_service._chunks else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-index documents: {str(e)}")

@router.get("/api/system/logs")
async def get_retrieval_logs(lines: int = Query(50, ge=1, le=500)):
    """
    Returns the tail end of the RAG auditing logs file.
    """
    log_path = str(settings.LOG_FILE_PATH)
    if not os.path.exists(log_path):
        return {
            "log_path": log_path,
            "exists": False,
            "content": "No queries logged yet."
        }
        
    try:
        with open(log_path, 'r', encoding='utf-8') as lf:
            content_lines = lf.readlines()
        tail = content_lines[-lines:] if len(content_lines) > lines else content_lines
        return {
            "log_path": log_path,
            "exists": True,
            "line_count": len(tail),
            "content": "".join(tail)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")
