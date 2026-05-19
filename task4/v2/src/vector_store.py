import os
import chromadb
import numpy as np

def build_or_load_store(chunks: list[dict], embedder, persist_dir: str) -> chromadb.Collection:
    """
    Step 2: Initializes a persistent ChromaDB client, checks if collection
    already exists on disk, and populates it if it is a fresh run.
    """
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize persistent client
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "maintenance_kb"
    
    loaded_from_disk = False
    
    try:
        # Check if the collection already exists and has documents
        collection = client.get_collection(name=collection_name)
        if collection.count() > 0:
            print(f"\n[ChromaDB] Collection '{collection_name}' successfully loaded from disk.")
            print(f"[ChromaDB] Total archived chunks: {collection.count()}")
            loaded_from_disk = True
        else:
            print(f"\n[ChromaDB] Collection '{collection_name}' is empty. Initiating fresh build...")
            loaded_from_disk = False
    except Exception:
        # Collection does not exist at all
        print(f"\n[ChromaDB] Collection '{collection_name}' not found. Creating a new one...")
        collection = client.create_collection(name=collection_name)
        loaded_from_disk = False
        
    if not loaded_from_disk:
        print(f"[ChromaDB] Computing embeddings and populating database for {len(chunks)} chunks...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        # Extract metadata and document content
        for c in chunks:
            ids.append(f"chunk_{c['chunk_index']}")
            
            # Embed text using local embedder
            chunk_embedding = embedder.encode([c['text']])[0].tolist()
            embeddings.append(chunk_embedding)
            
            # ChromaDB metadata must only have simple types (str, int, float, bool)
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
