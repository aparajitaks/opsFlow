import os
import json
import re
import numpy as np

def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences using regex boundary detection."""
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunk_text(text: str, embedder, similarity_threshold: float = 0.6, max_chunk_size: int = 400) -> list[str]:
    """
    Advanced semantic chunker:
    1. Splits the document into sentences.
    2. Encodes each sentence.
    3. Computes cosine similarity between consecutive sentences.
    4. Splices sentences into chunks when similarity drops below threshold.
    5. Enforces max_chunk_size (in words) to prevent overly large chunks.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences
        
    try:
        # Encode all sentences
        embeddings = embedder.encode(sentences, show_progress_bar=False)
        
        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)
            
        chunks = []
        current_chunk = [sentences[0]]
        current_word_count = len(sentences[0].split())
        
        for idx, sim in enumerate(similarities):
            next_sentence = sentences[idx + 1]
            next_words = len(next_sentence.split())
            
            # Check if we should split
            if sim < similarity_threshold or (current_word_count + next_words > max_chunk_size):
                chunks.append(" ".join(current_chunk))
                current_chunk = [next_sentence]
                current_word_count = next_words
            else:
                current_chunk.append(next_sentence)
                current_word_count += next_words
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    except Exception as e:
        print(f"[Chunking Warning] Semantic chunking failed: {e}. Falling back to sentences join.")
        return [" ".join(sentences)]

def chunk_documents(docs_dir: str, embedder=None, chunk_size: int = 300, overlap: int = 50, use_semantic: bool = False) -> list[dict]:
    """
    Performs chunking across all text/JSON documents in docs_dir.
    Supports either overlap chunking or advanced semantic chunking.
    Returns a list of dictionaries with text and metadata.
    """
    chunks = []
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
        
    filenames = sorted([f for f in os.listdir(docs_dir) if f.endswith('.txt') or f.endswith('.json')])
    
    global_chunk_idx = 0
    for filename in filenames:
        file_path = os.path.join(docs_dir, filename)
        if filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                
                # Make dynamic summary text from task JSON metrics
                text = (
                    f"Machine Learning Model Summary and Training Results:\n"
                    f"Training Timestamp: {data.get('run_timestamp')}\n"
                    f"Best Performing Model: The best performing model is the {data.get('best_model')}.\n"
                    f"Best F1 Score: The best F1 score achieved by the model is {data.get('best_f1')}.\n"
                    f"Best ROC-AUC Score: The best ROC-AUC score is {data.get('best_roc_auc')}.\n"
                    f"Best Model Hyperparameters: The best hyperparameters for the model are {data.get('best_params')}.\n"
                    f"Top Features for Failure Prediction: The top 3 most important features for predicting equipment failure are {', '.join(data.get('top_features', []))}.\n"
                    f"Dataset Failure Rate: The failure rate in the predictive maintenance dataset is {data.get('failure_rate_in_dataset') * 100:.1f}% (or {data.get('failure_rate_in_dataset')}).\n"
                )
            except Exception:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
        # Select chunking strategy
        if use_semantic and embedder is not None:
            raw_chunks = semantic_chunk_text(text, embedder)
            for rc in raw_chunks:
                words = rc.split()
                if not words:
                    continue
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": global_chunk_idx,
                    "start_word": 0,
                    "end_word": len(words) - 1,
                    "word_count": len(words),
                    "text": rc
                })
                global_chunk_idx += 1
        else:
            # Traditional Sliding Window with Overlap
            words = text.split()
            n_words = len(words)
            step = chunk_size - overlap
            
            if n_words <= chunk_size:
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": global_chunk_idx,
                    "start_word": 0,
                    "end_word": max(0, n_words - 1),
                    "word_count": n_words,
                    "text": " ".join(words)
                })
                global_chunk_idx += 1
                continue
                
            for start_idx in range(0, n_words, step):
                end_idx = start_idx + chunk_size
                chunk_words = words[start_idx:end_idx]
                if not chunk_words:
                    break
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": global_chunk_idx,
                    "start_word": start_idx,
                    "end_word": start_idx + len(chunk_words) - 1,
                    "word_count": len(chunk_words),
                    "text": " ".join(chunk_words)
                })
                global_chunk_idx += 1
                if end_idx >= n_words:
                    break
                    
    return chunks
