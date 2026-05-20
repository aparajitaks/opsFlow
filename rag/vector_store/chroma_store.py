"""ChromaDB-backed persistent vector store."""
from __future__ import annotations

import threading
from typing import Any

import chromadb

from core.config import settings
from core.logger import get_logger

log = get_logger("rag.vector_store")

_chroma_client: chromadb.PersistentClient | None = None
_chroma_lock = threading.Lock()


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


class ChromaVectorStore:
    """Encapsulates Chroma collection lifecycle and semantic search."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or str(settings.DATABASE_DIR)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self._collection: chromadb.Collection | None = None

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call load_or_build first.")
        return self._collection

    def load_or_build(self, chunks: list[dict], embedder) -> chromadb.Collection:
        import os

        os.makedirs(self.persist_dir, exist_ok=True)
        client = _get_client(self.persist_dir)
        loaded = False

        try:
            col = client.get_collection(name=self.collection_name)
            if col.count() > 0 and col.count() == len(chunks):
                log.info(
                    "Chroma collection '%s' loaded (%d chunks).",
                    self.collection_name,
                    col.count(),
                )
                self._collection = col
                return col
            if col.count() > 0:
                log.info(
                    "Chroma count mismatch (%d vs %d). Rebuilding.",
                    col.count(),
                    len(chunks),
                )
                client.delete_collection(name=self.collection_name)
            col = client.create_collection(name=self.collection_name)
        except Exception:
            log.info("Creating Chroma collection '%s'.", self.collection_name)
            col = client.create_collection(name=self.collection_name)

        log.info("Embedding %d chunks into Chroma...", len(chunks))
        ids, embeddings, metadatas, documents = [], [], [], []
        for c in chunks:
            ids.append(f"chunk_{c['chunk_index']}")
            embeddings.append(embedder.encode([c["text"]])[0].tolist())
            metadatas.append({
                "doc_name": c["doc_name"],
                "chunk_index": c["chunk_index"],
                "start_word": c["start_word"],
                "end_word": c["end_word"],
                "word_count": c["word_count"],
            })
            documents.append(c["text"])

        col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        log.info("Chroma populated with %d chunks.", col.count())
        self._collection = col
        return col

    def semantic_search(
        self,
        query: str,
        embedder,
        k: int = 3,
        where_filter: dict | None = None,
    ) -> list[dict]:
        col = self.collection
        query_vector = embedder.encode([query])[0].tolist()
        kwargs: dict[str, Any] = {"query_embeddings": [query_vector], "n_results": k}
        if where_filter:
            kwargs["where"] = where_filter
        results = col.query(**kwargs)

        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            chunks.append({
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "start_word": meta["start_word"],
                "end_word": meta["end_word"],
                "text": results["documents"][0][i],
                "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0),
            })
        return chunks


def build_or_load_store(chunks: list[dict], embedder, persist_dir: str | None = None):
    """Backward-compatible factory used by pipeline and retriever."""
    store = ChromaVectorStore(persist_dir=persist_dir)
    return store.load_or_build(chunks, embedder)
