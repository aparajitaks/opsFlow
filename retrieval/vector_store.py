import os
import chromadb
import streamlit as st
from core.config import settings

@st.cache_resource
def get_chroma_client(persist_dir: str):
    return chromadb.PersistentClient(path=persist_dir)

def build_or_load_store(chunks: list[dict], embedder, persist_dir: str = None) -> chromadb.Collection:
    """
    Initializes a persistent ChromaDB client, check if collection
    already exists on disk, and populates it if it is a fresh run.
    """
    if persist_dir is None:
        persist_dir = str(settings.DATABASE_DIR)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize persistent client using cached helper
    client = get_chroma_client(persist_dir)
    collection_name = "maintenance_kb"
    
    loaded_from_disk = False
    
    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() > 0 and collection.count() == len(chunks):
            print(f"\n[ChromaDB] Collection '{collection_name}' successfully loaded from disk.")
            print(f"[ChromaDB] Total archived chunks: {collection.count()}")
            loaded_from_disk = True
        else:
            if collection.count() > 0:
                print(f"\n[ChromaDB] Collection count mismatch ({collection.count()} vs {len(chunks)} chunks). Rebuilding database for fresh updates...")
                client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
            loaded_from_disk = False
    except Exception:
        print(f"\n[ChromaDB] Collection '{collection_name}' not found. Creating a new one...")
        collection = client.create_collection(name=collection_name)
        loaded_from_disk = False
        
    if not loaded_from_disk:
        print(f"[ChromaDB] Computing embeddings and populating database for {len(chunks)} chunks...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for c in chunks:
            ids.append(f"chunk_{c['chunk_index']}")
            
            # Embed text using local embedder
            chunk_embedding = embedder.encode([c['text']])[0].tolist()
            embeddings.append(chunk_embedding)
            
            metadata = {
                "doc_name": c["doc_name"],
                "chunk_index": c["chunk_index"],
                "start_word": c["start_word"],
                "end_word": c["end_word"],
                "word_count": c["word_count"]
            }
            metadatas.append(metadata)
            documents.append(c["text"])
            
        # Add in batch to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"[ChromaDB] Successfully stored {collection.count()} chunks in the persistent vector database.")
        
    return collection

def semantic_retrieve(query: str, embedder, collection, k: int = 3, where_filter: dict = None) -> list[dict]:
    """
    Embeds the incoming user query and searches ChromaDB for
    the top k most semantically similar chunks, with optional metadata filtering.
    """
    query_vector = embedder.encode([query])[0].tolist()
    
    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": k
    }
    if where_filter:
        query_kwargs["where"] = where_filter
        
    results = collection.query(**query_kwargs)
    
    retrieved_chunks = []
    
    if results and "documents" in results and len(results["documents"]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[0.0] * len(documents)])[0]
        
        for idx in range(len(documents)):
            meta = metadatas[idx]
            distance = float(distances[idx])
            # Conversions from L2 distances: score = 1.0 - distance
            score = 1.0 - distance
            
            retrieved_chunks.append({
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "start_word": meta["start_word"],
                "end_word": meta["end_word"],
                "word_count": meta["word_count"],
                "score": score,
                "text": documents[idx]
            })
            
    return retrieved_chunks

def get_chromadb_vs_faiss_explanation() -> str:
    return (
        "--- CHROMADB VS FAISS COMPARISON ---\n"
        "1. Complete Database vs. Core Vector Library: FAISS is a lightweight search library focused strictly "
        "   on in-memory similarity operations. It does not manage raw text, IDs, or metadata internally; developers "
        "   must maintain separate lookup tables. ChromaDB is a feature-rich, serverless vector database that "
        "   handles vectors, text contents, and metadata in an integrated SQL-backed store.\n"
        "2. Persistent Lifecycle: In FAISS, persistence requires dumping and loading index files manually. ChromaDB "
        "   manages persistence out of the box using a standardized SQLite-backed engine via PersistentClient.\n"
        "3. Metadata Filtering & CRUD: FAISS does not support direct semantic metadata filtering during indexing. "
        "   ChromaDB natively supports metadata filtering (e.g., retrieving only safety manual chunks) using standard "
        "   where clauses at search time, alongside simple CRUD operations to update or delete vectors dynamically."
    )
