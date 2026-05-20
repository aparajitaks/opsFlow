import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from api.routes import query, ml, system
from services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmstart heavier deep learning models (sentence embedder and cross-encoder reranker)
    # This prevents the first API query from taking 5+ seconds.
    print("[FastAPI Startup] Warm starting retrieval models and caching database...")
    try:
        rag_service.initialize_pipeline()
    except Exception as e:
        print(f"[FastAPI Startup Warning] Failed to warm up RAG components: {e}. Make sure CSV dataset and docs are set up.")
    yield
    print("[FastAPI Shutdown] Stopping services...")

app = FastAPI(
    title="opsFlow API",
    description="Production-Grade REST API for Predictive Maintenance Telemetry and RAG Operations.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS middleware for secure communication with Streamlit client or external scripts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains (e.g. settings.ALLOWED_ORIGINS)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(query.router)
app.include_router(ml.router)
app.include_router(system.router)

@app.get("/")
async def root():
    return {
        "app": "opsFlow",
        "description": "Welcome to the opsFlow predictive maintenance API.",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    # Standard entrypoint for local execution
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
