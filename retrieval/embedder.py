import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

@st.cache_resource
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        device="cpu"
    )

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Generates embedding vectors for a list of texts.
    """
    return embedder.encode(texts, show_progress_bar=False)
