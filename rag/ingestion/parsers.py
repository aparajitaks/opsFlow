"""Document text extraction for multi-format ingestion."""
import json
from pathlib import Path

from core.exceptions import IngestionError


def extract_text_from_file(file_path: Path) -> str:
    """Extract plain text from txt, json, pdf, or docx files."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        with open(file_path, encoding="utf-8") as jf:
            data = json.load(jf)
        return (
            f"Machine Learning Model Summary and Training Results:\n"
            f"Training Timestamp: {data.get('run_timestamp')}\n"
            f"Best Performing Model: {data.get('best_model')}\n"
            f"Best F1 Score: {data.get('best_f1')}\n"
            f"Best ROC-AUC Score: {data.get('best_roc_auc')}\n"
            f"Best Model Hyperparameters: {data.get('best_params')}\n"
            f"Top Features: {', '.join(data.get('top_features', []))}\n"
            f"Dataset Failure Rate: {data.get('failure_rate_in_dataset')}\n"
        )
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise IngestionError("PDF support requires pypdf.") from e
        reader = PdfReader(str(file_path))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise IngestionError("PDF contains no extractable text.")
        return text
    if suffix == ".docx":
        try:
            import docx
        except ImportError as e:
            raise IngestionError("DOCX support requires python-docx.") from e
        document = docx.Document(str(file_path))
        text = "\n".join(p.text for p in document.paragraphs if p.text).strip()
        if not text:
            raise IngestionError("DOCX contains no extractable text.")
        return text
    raise IngestionError(f"Unsupported file type: {suffix}")
