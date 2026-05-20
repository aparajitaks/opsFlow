import numpy as np
from sentence_transformers import SentenceTransformer

def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Step 3: Instantiates the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Generates embedding vectors for a list of texts.
    """
    return embedder.encode(texts, show_progress_bar=False)
