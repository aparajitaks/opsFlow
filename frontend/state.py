import requests
import streamlit as st

BACKEND_URL = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

def check_backend_health() -> bool:
    """Checks if the FastAPI backend is running and healthy."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def query_rag(query_text: str, custom_api_key: str = None) -> dict | None:
    """Sends a question to the RAG endpoint."""
    try:
        payload = {"query": query_text}
        if custom_api_key:
            payload["groq_api_key"] = custom_api_key
        r = requests.post(f"{BACKEND_URL}/api/query", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Backend Query Error ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
    return None

def clear_query_cache() -> bool:
    """Clears the query cache on the backend."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/clear_cache", timeout=5)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")
        return False

def predict_ml(telemetry_data: dict, model_type: str = "random_forest") -> dict | None:
    """Predicts equipment failure based on telemetry metrics."""
    try:
        payload = {**telemetry_data, "model_type": model_type}
        r = requests.post(f"{BACKEND_URL}/api/model/predict", json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Backend Prediction Error ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
    return None

def get_model_status() -> dict | None:
    """Retrieves ML training status and performance summaries."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/model/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def trigger_retraining() -> dict | None:
    """Triggers ML model retraining in the background."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/model/retrain", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to trigger retraining: {e}")
    return None

def force_reindex(use_semantic: bool = False) -> bool:
    """Triggers RAG document reindexing on the backend."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/system/reindex", json={"use_semantic_chunking": use_semantic}, timeout=30)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Failed to re-index: {e}")
        return False

def fetch_logs(lines: int = 50) -> dict | None:
    """Fetches retrieval auditing logs from the backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/system/logs", params={"lines": lines}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
    return None
