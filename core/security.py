import re
import html
import time
from collections import defaultdict
from core.config import settings

# Prompt injection regular expressions
PROMPT_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(?:the\s+)?(?:all|any|prior|previous|above)\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all|any|previous|prior)(?:\s+previous)?\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(r"bypass\s+(?:system|security|safety|instructions)", re.IGNORECASE),
    re.compile(r"(?:acting|act)\s+as", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+leak", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"override\s+(?:grounding|instructions|controls)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?(?:developer|admin|root)\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

class TokenBucketRateLimiter:
    """
    In-memory thread-safe token bucket rate limiter.
    """
    def __init__(self, rate_limit: int = 60, period: float = 60.0):
        self.rate_limit = rate_limit
        self.period = period
        self.buckets = defaultdict(lambda: (rate_limit, time.time()))

    def is_allowed(self, client_id: str) -> bool:
        tokens, last_update = self.buckets[client_id]
        now = time.time()
        
        # Calculate elapsed time and add tokens proportionally
        elapsed = now - last_update
        tokens_to_add = elapsed * (self.rate_limit / self.period)
        
        new_tokens = min(self.rate_limit, tokens + tokens_to_add)
        
        if new_tokens >= 1.0:
            self.buckets[client_id] = (new_tokens - 1.0, now)
            return True
        else:
            self.buckets[client_id] = (new_tokens, now)
            return False

# Global instances
rate_limiter = TokenBucketRateLimiter(rate_limit=settings.RATE_LIMIT_PER_MINUTE)

def sanitize_input(text: str) -> str:
    """
    Cleans inputs: strips whitespace, escapes HTML tags, limits character size.
    """
    if not text:
        return ""
    
    # Strip whitespace
    cleaned = text.strip()
    
    # Limit length
    if len(cleaned) > settings.MAX_QUERY_LENGTH:
        cleaned = cleaned[:settings.MAX_QUERY_LENGTH]
        
    # Escape HTML to prevent injection
    cleaned = html.escape(cleaned)
    return cleaned

def check_prompt_injection(text: str) -> bool:
    """
    Scans the input text for patterns common in LLM prompt injection attacks.
    Returns True if an injection risk is identified.
    """
    if not text:
        return False
    
    # Scan regexes
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
            
    return False
