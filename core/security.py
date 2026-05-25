"""
core/security.py — input sanitization and prompt injection detection.

Intentionally minimal: this is a CLI tool, not a web server.
- No rate limiting (single-user CLI)
- No HTML escaping (terminal input, not rendered HTML)
- Prompt injection regex checks: genuinely useful for a grounded RAG tool
"""
import re

from core.config import settings

# Patterns that indicate attempts to override the grounding constraints
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:the\s+)?(?:all|any|prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all|any|previous|prior)(?:\s+previous)?\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(?:system|security|safety|instructions)", re.IGNORECASE),
    re.compile(r"(?:acting|act)\s+as", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+leak", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"override\s+(?:grounding|instructions|controls)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?(?:developer|admin|root)\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]


def sanitize_input(text: str) -> str:
    """Strips whitespace and truncates to the configured maximum query length."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) > settings.MAX_QUERY_LENGTH:
        cleaned = cleaned[: settings.MAX_QUERY_LENGTH]
    return cleaned


def check_prompt_injection(text: str) -> bool:
    """Returns True if the input matches a known prompt-injection pattern."""
    if not text:
        return False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
