from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    return _model

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Generates embedding vectors for a list of texts.
    """
    return embedder.encode(texts, show_progress_bar=False)
