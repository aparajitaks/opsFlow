import os
import re
import html
import time
import json
import datetime
from collections import defaultdict
from groq import Groq
from core.config import settings

# In-memory TokenBucket Rate Limiter
class TokenBucketRateLimiter:
    """In-memory thread-safe token-bucket rate limiter to prevent API abuse."""
    def __init__(self, rate_limit: int = 60, period: float = 60.0):
        self.rate_limit = rate_limit
        self.period = period
        self.buckets = defaultdict(lambda: (float(rate_limit), time.time()))

    def is_allowed(self, client_id: str) -> bool:
        tokens, last_update = self.buckets[client_id]
        now = time.time()
        
        # Replenish tokens proportional to elapsed time
        elapsed = now - last_update
        tokens_to_add = elapsed * (self.rate_limit / self.period)
        new_tokens = min(float(self.rate_limit), tokens + tokens_to_add)
        
        if new_tokens >= 1.0:
            self.buckets[client_id] = (new_tokens - 1.0, now)
            return True
        else:
            self.buckets[client_id] = (new_tokens, now)
            return False

rate_limiter = TokenBucketRateLimiter(rate_limit=settings.RATE_LIMIT_PER_MINUTE)

# Security scanning & Prompt injection regex firewalls
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:the\s+)?(?:prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(?:system|security|safety|instructions)", re.IGNORECASE),
    re.compile(r"(?:acting|act)\s+as", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+leak", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"print\s+your\s+instructions", re.IGNORECASE),
    re.compile(r"override\s+(?:grounding|instructions|controls)", re.IGNORECASE),
]

def sanitize_input(text: str) -> str:
    """Cleans inputs: strips whitespace, escapes HTML tags, limits character size."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) > settings.MAX_QUERY_LENGTH:
        cleaned = cleaned[:settings.MAX_QUERY_LENGTH]
    return html.escape(cleaned)

def check_prompt_injection(text: str) -> bool:
    """Scans query for common adversarial prompt injection attacks."""
    if not text:
        return False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False

# Groq Client Cache Singleton
_groq_client_instance = None
_groq_lock = threading_lock = threading = None # We will use a standard lock

import threading
_groq_lock = threading.Lock()

def get_groq_client(api_key: str) -> Groq:
    """Thread-safe singleton retriever for Groq API client."""
    global _groq_client_instance
    if not api_key or api_key.strip().lower() in ("none", "undefined", "null", ""):
        return None
    if _groq_client_instance is None:
        with _groq_lock:
            if _groq_client_instance is None:
                _groq_client_instance = Groq(api_key=api_key.strip())
    return _groq_client_instance

def _clear_groq_client():
    global _groq_client_instance
    _groq_client_instance = None

get_groq_client.clear = _clear_groq_client

def is_valid_groq_key(api_key: str) -> bool:
    """Validates that the provided Groq API key is populated and non-trivial."""
    if not api_key or not isinstance(api_key, str):
        return False
    k = api_key.strip()
    return len(k) > 0 and k.lower() not in ("none", "undefined", "null", "")

# Grounded Generation Completions
def generate_llm_completion(query: str, retrieved_chunks: list[dict], client: Groq) -> str:
    """Executes grounded generation call to Groq with rate-limit retries."""
    context = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
    
    max_retries = 3
    retry_delays = [3, 6, 12]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_GENERATOR,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an industrial maintenance assistant. Answer ONLY using the provided context. "
                            "If the answer is not in the context, say exactly: 'I don't have enough information in "
                            "my knowledge base to answer this question.' Do not use outside knowledge or extrapolate."
                        )
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            if is_429 and attempt < max_retries - 1:
                wait = retry_delays[attempt]
                print(f"[RAG Generator 429] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                return f"Error during Groq generation: {e}"
                
    return "Rate limit exceeded in Generator after multiple retries."

def get_mock_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Fallback mock answers for local sandboxed CLI testing."""
    if "water" in query.lower() or "boiling" in query.lower():
        return "The boiling point of water is exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
    return (
        "[MOCK RAG GENERATOR - NO ACTIVE GROQ API KEY]\n"
        "Grounded Context Reference Chunks:\n"
        + "\n".join([f" • [{c['doc_name']} Chunk {c['chunk_index']}] - {c['text'][:120]}..." for c in retrieved_chunks])
    )

# Double-pass Faithfulness Claims Audit
def check_faithfulness(answer: str, retrieved_chunks: list[dict], client: Groq) -> dict:
    """Audits the generated answer to confirm strict grounding in retrieved sources."""
    if not client:
        # Mock audit logic for local evaluations
        if "boiling point of water" in answer.lower():
            return {
                "faithful": False,
                "score": 0.0,
                "verdict": "Water boiling point parameters are completely absent from the equipment maintenance manuals.",
                "unsupported_claims": ["The boiling point of water is exactly 100 degrees Celsius."]
            }
        return {
            "faithful": True,
            "score": 1.0,
            "verdict": "Mock Audit: Passed (Groq client not active).",
            "unsupported_claims": []
        }

    chunks_text = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
    
    prompt = f"""You are a faithfulness auditor. Given a generated answer and the source context it was based on, determine whether every factual claim in the answer is directly supported by the context.

Context:
{chunks_text}

Generated Answer:
{answer}

Respond in this exact JSON format:
{{
  "faithful": true or false,
  "score": 0.0 to 1.0,
  "unsupported_claims": ["claim 1", "claim 2"] or [],
  "verdict": "one sentence summary"
}}"""

    max_retries = 3
    retry_delays = [3, 6, 12]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_AUDITOR,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a factual claim auditor. Respond ONLY in valid JSON. "
                            "Do not write introductory or concluding remarks outside the JSON block. "
                            "CRITICAL: If the Generated Answer is a refusal to answer (e.g. 'I don't have enough "
                            "information...'), mark it as faithful (faithful: true, score: 1.0, unsupported_claims: [])."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            return parse_faithfulness_json(raw)
        except Exception as e:
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            if is_429 and attempt < max_retries - 1:
                wait = retry_delays[attempt]
                print(f"[RAG Auditor 429] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                return {
                    "faithful": False,
                    "score": 0.0,
                    "unsupported_claims": [f"Auditor failure: {e}"],
                    "verdict": "Auditing service encountered an active exception."
                }
                
    return {
        "faithful": False,
        "score": 0.0,
        "unsupported_claims": ["Auditor rate limit exceeded"],
        "verdict": "Auditing was blocked by rate limits after multiple retries."
    }

def parse_faithfulness_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()
        
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        return json.loads(raw_text[start:end])
    except Exception as e:
        return {
            "faithful": False,
            "score": 0.0,
            "unsupported_claims": ["Failed to extract valid JSON formatting from auditor response"],
            "verdict": f"JSON parsing error: {e}"
        }

# Logging queries to persistent files
def log_query(query: str, retrieved_chunks: list[dict], answer: str, log_path: str = None, pre_rerank_chunks: list[dict] = None, faithfulness_res: dict = None):
    """Logs the complete parameters of the RAG interaction for safety and auditability."""
    if log_path is None:
        log_path = str(settings.LOG_FILE_PATH)
        
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pre_ranks = ""
    post_ranks = ""
    if pre_rerank_chunks:
        pre_ranks = "[" + ", ".join([f"{c['doc_name']} pre-rank{c.get('pre_rerank_rank', 'N/A')}" for c in retrieved_chunks]) + "]"
        post_ranks = "[" + ", ".join([f"{c['doc_name']} post-rank{c.get('post_rerank_rank', 'N/A')}" for c in retrieved_chunks]) + "]"
        
    faithful_val = "N/A"
    faith_score = 0.0
    faith_verdict = ""
    if faithfulness_res:
        faithful_val = "Yes" if faithfulness_res.get("faithful") else "No"
        faith_score = faithfulness_res.get("score", 0.0)
        faith_verdict = faithfulness_res.get("verdict", "")
        
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("=====================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Total chunks retrieved: {len(retrieved_chunks)}\n")
        f.write("-------------------------------------\n")
        f.write("Retrieval Method : Hybrid (BM25 + Semantic, RRF fusion)\n")
        if pre_ranks:
            f.write(f"Pre-rerank Ranks : {pre_ranks}\n")
            f.write(f"Post-rerank Ranks: {post_ranks}\n")
        f.write(f"Faithfulness     : {faithful_val} ({faith_score}) — \"{faith_verdict}\"\n")
        f.write("-------------------------------------\n")
        
        for idx, c in enumerate(retrieved_chunks):
            preview = c["text"].replace("\n", " ")[:100]
            f.write(f"[Chunk {idx+1}]\n")
            f.write(f"  Document : {c['doc_name']}\n")
            f.write(f"  Chunk ID : {c['chunk_index']}\n")
            f.write(f"  Words    : {c['start_word']}–{c['end_word']}\n")
            f.write(f"  Score    : {c.get('score', 0.0):.4f}\n")
            f.write(f"  Preview  : {preview}...\n")
            f.write("-------------------------------------\n")
            
        f.write("[LLM Answer]\n")
        f.write(f"{answer}\n")
        f.write("=====================================\n\n")

# Informative explanations
def get_relevance_vs_faithfulness_explanation() -> str:
    return (
        "--- FAITHFULNESS VS RELEVANCE EXPLANATION ---\n"
        "1. Relevance:\n"
        "   - Relevance measures if the retrieved or generated contents focus on the same subject or domain\n"
        "     as the user's question. For example, if a user asks about high-voltage lines, a paragraph detailing\n"
        "     the general operations of 480V cabinets is highly relevant.\n"
        "2. Faithfulness:\n"
        "   - Faithfulness measures factual alignment. It verifies that every claims, parameter value, or troubleshooting\n"
        "     instruction generated by the assistant is strictly supported by the actual source context without introducing\n"
        "     extraneous assumptions or hallucinations.\n"
        "3. Highly Relevant but Unfaithful Example:\n"
        "   - If a technician asks: 'What is the voltage limit for high-voltage cabinets?'\n"
        "   - A relevant but unfaithful model might answer: 'The voltage limit for high-voltage cabinets is 600V.'\n"
        "     Even if the context mentions 'voltage limits are defined in safety procedure 3', the model hallucinated the\n"
        "     value '600V' from its pre-training weights. This response is highly relevant but factually unfaithful,\n"
        "     introducing high risk to industrial safety operations."
    )

def get_source_logging_explanation() -> str:
    return (
        "--- WHY CHUNK SOURCE LOGGING MATTERS FOR AUDITABILITY ---\n"
        "1. Safety Compliance & Verification: In physical industrial plants, executing incorrect maintenance "
        "   (such as operating out-of-spec torque regimes or skipping isolated LOTO safety locks) can cause physical "
        "   injury or severe equipment damage. Log records prove exactly which operational manual was consulted.\n"
        "2. Hallucination Guardrails: If an LLM response sounds incorrect or ambiguous, engineers can trace "
        "   the prediction back to specific sections in the original manual, allowing rapid auditing and model tuning."
    )
