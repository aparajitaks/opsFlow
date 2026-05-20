from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Instantiates and returns the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Generates embedding vectors for a list of texts.
    """
    return embedder.encode(texts, show_progress_bar=False)
