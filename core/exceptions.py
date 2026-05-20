"""Domain exceptions for opsFlow."""


class OpsFlowError(Exception):
    """Base exception for application errors."""


class DatasetValidationError(OpsFlowError, ValueError):
    """Raised when the predictive maintenance dataset fails integrity checks."""


class ModelArtifactError(OpsFlowError, FileNotFoundError):
    """Raised when required ML artifacts are missing or corrupt."""


class RAGInitializationError(OpsFlowError):
    """Raised when the RAG pipeline is used before initialization."""


class IngestionError(OpsFlowError, ValueError):
    """Raised when document ingestion or parsing fails (subclasses ValueError for API mapping)."""
