import threading
from sentence_transformers import SentenceTransformer
import numpy as np

# Global cache for the embedding model instance
_embedder_instance = None
_embedder_lock = threading.Lock()

def get_embedder(model_name: str = None) -> SentenceTransformer:
    from core.config import settings
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    """
    Thread-safe global singleton caching of SentenceTransformer embedding model
    to prevent redundant initialization and save CPU cycles.
    """
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                print(f"[RAG] Initializing embedding model '{model_name}' on CPU...")
                _embedder_instance = SentenceTransformer(model_name, device="cpu")
    return _embedder_instance

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Generates embedding vectors for a list of texts."""
    return embedder.encode(texts, show_progress_bar=False)
