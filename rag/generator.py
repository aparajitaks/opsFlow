import os
import json
import datetime
import groq
from groq import Groq
from core.config import settings
from core.logger import get_logger
from core.security import (
    TokenBucketRateLimiter,
    rate_limiter,
    sanitize_input,
    check_prompt_injection,
    PROMPT_INJECTION_PATTERNS,
)

log = get_logger("rag.generator")

import threading

_groq_clients: dict[str, Groq] = {}
_groq_lock = threading.Lock()


def get_groq_client(api_key: str) -> Groq | None:
    """Thread-safe per-key Groq client cache (supports key rotation)."""
    if not is_valid_groq_key(api_key):
        return None
    key = api_key.strip()
    with _groq_lock:
        if key not in _groq_clients:
            _groq_clients[key] = Groq(api_key=key)
        return _groq_clients[key]


def _clear_groq_client():
    with _groq_lock:
        _groq_clients.clear()


get_groq_client.clear = _clear_groq_client

def is_valid_groq_key(api_key: str) -> bool:
    """Validates that the provided Groq API key is populated and non-trivial."""
    if not api_key or not isinstance(api_key, str):
        return False
    k = api_key.strip()
    return len(k) > 0 and k.lower() not in ("none", "undefined", "null", "")

# Grounded Generation Completions
def generate_llm_completion(query: str, retrieved_chunks: list[dict], client: Groq, conversation_history: list = None) -> str:
    """Executes grounded generation call to Groq with rate-limit retries and conversational memory support."""
    context = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
    
    # Construct message sequence
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an industrial maintenance assistant. Answer ONLY using the provided context. "
                f"If the answer is not in the context, say exactly: '{settings.KB_REFUSAL_MESSAGE}' "
                "Do not use outside knowledge or extrapolate."
            )
        }
    ]
    
    # If conversation history is present, inject it before the current question
    if conversation_history:
        for turn in conversation_history[-settings.MEMORY_MAX_TURNS:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
    # Add current question with context
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    
    max_retries = settings.GEN_MAX_RETRIES
    retry_delays = settings.GEN_RETRY_DELAYS
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_GENERATOR,
                messages=messages,
                temperature=settings.GEN_TEMPERATURE
            )
            
            # Log token usage
            if hasattr(response, "usage") and response.usage:
                log.info(
                    f"Generator Groq API usage - Prompt tokens: {response.usage.prompt_tokens} | "
                    f"Completion tokens: {response.usage.completion_tokens} | "
                    f"Total tokens: {response.usage.total_tokens}"
                )
                
            return response.choices[0].message.content
        except groq.APIConnectionError as e:
            # ConnectionError: fail fast
            err_msg = f"Groq Connection Error: Could not connect to Groq API. Please check your network connection. Detail: {e}"
            log.error(err_msg)
            return err_msg
        except groq.RateLimitError as e:
            # RateLimitError (429): retry
            if attempt < max_retries - 1:
                wait = retry_delays[attempt]
                log.warning(f"[RAG Generator 429] Rate limit hit. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                err_msg = "Rate limit exceeded in Generator after multiple retries."
                log.error(err_msg)
                return err_msg
        except Exception as e:
            # Check for other errors
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            is_conn = "connection" in str(e).lower() or "connect" in str(e).lower()
            if is_conn:
                err_msg = f"Groq Connection Error: {e}"
                log.error(err_msg)
                return err_msg
            elif is_429 and attempt < max_retries - 1:
                wait = retry_delays[attempt]
                log.warning(f"[RAG Generator 429] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                err_msg = f"Error during Groq generation: {e}"
                log.error(err_msg, exc_info=True)
                return err_msg
                
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

    max_retries = settings.GEN_MAX_RETRIES
    retry_delays = settings.GEN_RETRY_DELAYS
    
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
                            "CRITICAL: If the Generated Answer is a refusal to answer (e.g. 'I could not find this "
                            "information...'), mark it as faithful (faithful: true, score: 1.0, unsupported_claims: [])."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.GEN_TEMPERATURE
            )
            
            # Log token usage
            if hasattr(response, "usage") and response.usage:
                log.info(
                    f"Auditor Groq API usage - Prompt tokens: {response.usage.prompt_tokens} | "
                    f"Completion tokens: {response.usage.completion_tokens} | "
                    f"Total tokens: {response.usage.total_tokens}"
                )
                
            raw = response.choices[0].message.content.strip()
            return parse_faithfulness_json(raw)
        except groq.APIConnectionError as e:
            # ConnectionError: fail fast
            err_msg = f"Auditor Connection Error: Could not connect to Groq API. Please check your network connection. Detail: {e}"
            log.error(err_msg)
            return {
                "faithful": False,
                "score": 0.0,
                "unsupported_claims": [err_msg],
                "verdict": "Auditing service connection failed."
            }
        except groq.RateLimitError as e:
            # RateLimitError (429): retry
            if attempt < max_retries - 1:
                wait = retry_delays[attempt]
                log.warning(f"[RAG Auditor 429] Rate limit hit. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                err_msg = "Rate limit exceeded in Auditor after multiple retries."
                log.error(err_msg)
                return {
                    "faithful": False,
                    "score": 0.0,
                    "unsupported_claims": [err_msg],
                    "verdict": "Auditing was blocked by rate limits after multiple retries."
                }
        except Exception as e:
            # Check for other errors
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            is_conn = "connection" in str(e).lower() or "connect" in str(e).lower()
            if is_conn:
                err_msg = f"Auditor Connection Error: {e}"
                log.error(err_msg)
                return {
                    "faithful": False,
                    "score": 0.0,
                    "unsupported_claims": [err_msg],
                    "verdict": "Auditing service connection failed."
                }
            elif is_429 and attempt < max_retries - 1:
                wait = retry_delays[attempt]
                log.warning(f"[RAG Auditor 429] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                err_msg = f"Auditor failure: {e}"
                log.error(err_msg, exc_info=True)
                return {
                    "faithful": False,
                    "score": 0.0,
                    "unsupported_claims": [err_msg],
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
