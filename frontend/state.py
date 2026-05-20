import os
import threading
import streamlit as st

# Direct imports from core services and utilities
from services.rag_service import rag_service
from services.ml_service import ml_service
from retrieval.cache import query_cache
from core.config import settings
from models.pipeline import run_ml_pipeline

def query_rag(query_text: str, custom_api_key: str = None) -> dict | None:
    """Directly invokes the RAG pipeline to answer user questions."""
    try:
        # Direct Python function execution
        return rag_service.run_query(query_text, custom_api_key)
    except Exception as e:
        st.error(f"RAG query execution failure: {e}")
    return None

def clear_query_cache() -> bool:
    """Clears the local in-memory RAG query cache."""
    try:
        query_cache.clear()
        return True
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")
        return False

def predict_ml(telemetry_data: dict, model_type: str = "random_forest") -> dict | None:
    """Preprocesses telemetry attributes and runs machine failure classification directly."""
    try:
        # Convert incoming UI attributes to the precise features expected by ml_service
        features = {
            "Type": telemetry_data["Type"],
            "Air temperature [K]": telemetry_data["Air_temperature"],
            "Process temperature [K]": telemetry_data["Process_temperature"],
            "Rotational speed [rpm]": telemetry_data["Rotational_speed"],
            "Torque [Nm]": telemetry_data["Torque"],
            "Tool wear [min]": telemetry_data["Tool_wear"]
        }
        
        # Direct execution of ML failure prediction
        return ml_service.predict_failure(features, model_type=model_type)
    except FileNotFoundError as fnf:
        st.error(str(fnf))
    except Exception as e:
        st.error(f"ML Inference processing failure: {e}")
    return None

def get_model_status() -> dict | None:
    """Retrieves standard ML model parameters and F1 training metrics directly."""
    try:
        return ml_service.get_model_status()
    except Exception as e:
        st.error(f"Failed to retrieve model status: {e}")
    return None

def trigger_retraining() -> dict | None:
    """Asynchronously triggers the ML hyperparameter tuning and model retraining pipeline."""
    try:
        # Prevent UI freezing by spinning up retraining in a background thread
        bg_thread = threading.Thread(target=run_ml_pipeline, daemon=True)
        bg_thread.start()
        
        return {
            "status": "training_started",
            "message": "Equipment failure ML retraining pipeline triggered in the background. Check logs or model status for completions."
        }
    except Exception as e:
        st.error(f"Failed to start retraining pipeline: {e}")
    return None

def force_reindex(use_semantic: bool = False) -> bool:
    """Forces direct document chunking, indexing, and persistent database build."""
    try:
        # Re-initialize the entire RAG pipeline from local files
        rag_service.initialize_pipeline(force=True, use_semantic_chunking=use_semantic)
        return True
    except Exception as e:
        st.error(f"Failed to re-index documents: {e}")
        return False

def fetch_logs(lines: int = 50) -> dict | None:
    """Directly tails the RAG auditing logs from the local filesystem."""
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
        st.error(f"Failed to read local logs: {e}")
    return None
