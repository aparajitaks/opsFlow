"""
rag/chunking.py — Document loading and text segmentation for Task 4.

Chunking strategies:
  - Fixed-size with overlap (default): deterministic, fast, no model dependency.
  - Semantic (optional): splits at cosine-similarity boundaries between sentences.
"""
import os
import json
import re
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".txt", ".json")


def extract_text_from_file(file_path: Path) -> str:
    """Read plain text or JSON files and return their string content."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return json.dumps(data, indent=2)
        if isinstance(data, list):
            return "\n\n".join(
                json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                for item in data
            )
        return str(data)
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Sentence splitting (for semantic chunking)
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences on punctuation boundaries."""
    boundary = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
    return [s.strip() for s in boundary.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def semantic_chunk_text(
    text: str,
    embedder,
    similarity_threshold: float = 0.6,
    max_chunk_size: int = 400,
) -> list[str]:
    """
    Splits text into semantically coherent chunks.

    Encodes each sentence and groups consecutive sentences until cosine
    similarity drops below threshold or the chunk exceeds max_chunk_size words.
    Falls back to returning the full text as one chunk on error.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    try:
        embeddings = embedder.encode(sentences, show_progress_bar=False)

        similarities = []
        for i in range(len(embeddings) - 1):
            v1, v2 = embeddings[i], embeddings[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            sim = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            similarities.append(sim)

        chunks, current, word_count = [], [sentences[0]], len(sentences[0].split())
        for i, sim in enumerate(similarities):
            next_s = sentences[i + 1]
            next_w = len(next_s.split())
            if sim < similarity_threshold or (word_count + next_w > max_chunk_size):
                chunks.append(" ".join(current))
                current, word_count = [next_s], next_w
            else:
                current.append(next_s)
                word_count += next_w

        if current:
            chunks.append(" ".join(current))
        return chunks

    except Exception as e:
        print(f"[Chunking] Semantic chunking failed: {e}. Returning full text.")
        return [" ".join(sentences)]


# ---------------------------------------------------------------------------
# Main chunking entrypoint
# ---------------------------------------------------------------------------

def chunk_documents(
    docs_dir: str,
    embedder=None,
    chunk_size: int = 300,
    overlap: int = 50,
    use_semantic: bool = False,
) -> list[dict]:
    """
    Loads all supported documents from docs_dir and segments them into chunks.

    Each chunk is a dict with: doc_name, chunk_index, start_word, end_word,
    word_count, text. chunk_index is globally unique across all documents.

    Chunking strategies:
      - Fixed-size (default): sliding window of chunk_size words with overlap.
      - Semantic (use_semantic=True): cosine-similarity-based sentence grouping.
    """
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    filenames = sorted([
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    chunks = []
    global_idx = 0

    for filename in filenames:
        file_path = Path(docs_dir) / filename
        try:
            text = extract_text_from_file(file_path)
        except Exception as e:
            print(f"[Chunking] Skipping {filename}: {e}")
            continue

        if use_semantic and embedder is not None:
            for raw in semantic_chunk_text(text, embedder):
                words = raw.split()
                if words:
                    chunks.append({
                        "doc_name": filename,
                        "chunk_index": global_idx,
                        "start_word": 0,
                        "end_word": len(words) - 1,
                        "word_count": len(words),
                        "text": raw,
                    })
                    global_idx += 1
        else:
            words = text.split()
            n = len(words)
            step = chunk_size - overlap

            if n <= chunk_size:
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": global_idx,
                    "start_word": 0,
                    "end_word": max(0, n - 1),
                    "word_count": n,
                    "text": " ".join(words),
                })
                global_idx += 1
                continue

            for start in range(0, n, step):
                end = start + chunk_size
                cwords = words[start:end]
                if not cwords:
                    break
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": global_idx,
                    "start_word": start,
                    "end_word": start + len(cwords) - 1,
                    "word_count": len(cwords),
                    "text": " ".join(cwords),
                })
                global_idx += 1
                if end >= n:
                    break

    return chunks
