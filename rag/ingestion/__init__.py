"""Document text extraction for RAG chunking."""
from rag.ingestion.parsers import extract_text_from_file

__all__ = [
    "extract_text_from_file",
    "ingest_uploaded_file",
    "IngestionService",
    "rebuild_index_after_upload",
]
