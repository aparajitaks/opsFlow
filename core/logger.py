import os
import logging
import logging.handlers
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured with both console and rotating file handlers.
    First call for a given name sets up handlers; subsequent calls return the same logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Load settings to get log config (lazy import to avoid circular deps)
    from core.config import settings

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    fmt = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATEFMT
    )

    # --- Console Handler ---
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # --- Rotating File Handler ---
    log_path = BASE_DIR / settings.LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
    return logger
