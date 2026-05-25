"""
rag/generator.py — Task 4: LLM generation + faithfulness auditing

Handles:
  - Groq client lifecycle (singleton per key)
  - Grounded generation with conversational memory
  - Double-pass faithfulness auditing
  - Structured JSON output parsing
  - RAG interaction logging
"""
import os
import time
import json
import datetime
import groq
from groq import Groq

from core.config import settings
from core.logger import get_logger

log = get_logger("rag.generator")


# ---------------------------------------------------------------------------
# Groq client management
# ---------------------------------------------------------------------------

_groq_clients: dict[str, Groq] = {}


def get_groq_client(api_key: str) -> Groq | None:
    """Returns a cached Groq client for the given key, or None if key is invalid."""
    if not is_valid_groq_key(api_key):
        return None
    key = api_key.strip()
    if key not in _groq_clients:
        _groq_clients[key] = Groq(api_key=key)
    return _groq_clients[key]


def _clear_groq_client():
    _groq_clients.clear()


get_groq_client.clear = _clear_groq_client


def is_valid_groq_key(api_key: str) -> bool:
    """Returns True if the key is a non-empty, non-placeholder string."""
    if not api_key or not isinstance(api_key, str):
        return False
    k = api_key.strip()
    return len(k) > 0 and k.lower() not in ("none", "undefined", "null", "")


# ---------------------------------------------------------------------------
# Grounded generation
# ---------------------------------------------------------------------------

def generate_llm_completion(
    query: str,
    retrieved_chunks: list[dict],
    client: Groq,
    conversation_history: list = None,
) -> str:
    """
    Calls Groq with retrieved context + optional conversational history.
    Retries on rate-limit errors with exponential back-off.
    """
    context = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an industrial maintenance assistant. Answer ONLY using the provided context. "
                f"If the answer is not in the context, say exactly: '{settings.KB_REFUSAL_MESSAGE}' "
                "Do not use outside knowledge or extrapolate."
            ),
        }
    ]

    if conversation_history:
        for turn in conversation_history[-settings.MEMORY_MAX_TURNS:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    for attempt in range(settings.GEN_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_GENERATOR,
                messages=messages,
                temperature=settings.GEN_TEMPERATURE,
            )
            if hasattr(response, "usage") and response.usage:
                log.debug(
                    "Generator tokens — prompt: %d, completion: %d",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            return response.choices[0].message.content

        except groq.APIConnectionError as e:
            msg = f"Groq connection error: {e}"
            log.error(msg)
            return msg

        except groq.RateLimitError:
            if attempt < settings.GEN_MAX_RETRIES - 1:
                wait = settings.GEN_RETRY_DELAYS[attempt]
                log.warning("Generator rate-limited. Retrying in %ds (attempt %d).", wait, attempt + 2)
                time.sleep(wait)
            else:
                msg = "Rate limit exceeded in generator after all retries."
                log.error(msg)
                return msg

        except Exception as e:
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            is_conn = "connection" in str(e).lower()
            if is_conn:
                msg = f"Groq connection error: {e}"
                log.error(msg)
                return msg
            elif is_429 and attempt < settings.GEN_MAX_RETRIES - 1:
                wait = settings.GEN_RETRY_DELAYS[attempt]
                log.warning("Generator rate-limited (generic). Retrying in %ds.", wait)
                time.sleep(wait)
            else:
                msg = f"Generator error: {e}"
                log.error(msg, exc_info=True)
                return msg

    return "Rate limit exceeded in generator after all retries."


def get_mock_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Fallback used when no valid Groq API key is available."""
    return (
        "[MOCK ANSWER — no active Groq API key]\n"
        "Retrieved chunks:\n"
        + "\n".join([
            f"  [{c['doc_name']} Chunk {c['chunk_index']}] {c['text'][:120]}..."
            for c in retrieved_chunks
        ])
    )


# ---------------------------------------------------------------------------
# Faithfulness auditing
# ---------------------------------------------------------------------------

def check_faithfulness(answer: str, retrieved_chunks: list[dict], client: Groq) -> dict:
    """
    Second-pass LLM call: audits the generated answer for grounding in the retrieved context.
    Returns structured dict with faithful bool, score, verdict, and any unsupported claims.
    """
    if answer.strip() == settings.KB_REFUSAL_MESSAGE.strip():
        return {
            "faithful": True,
            "score": 1.0,
            "verdict": "Factual refusal is faithful.",
            "unsupported_claims": [],
        }

    if not client:
        return {
            "faithful": True,
            "score": 1.0,
            "verdict": "Mock audit passed (no Groq client).",
            "unsupported_claims": [],
        }

    chunks_text = "\n\n".join([
        f"[{c['doc_name']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    prompt = (
        "You are a faithfulness auditor. Given a generated answer and the source context it was based on, "
        "determine whether every factual claim in the answer is directly supported by the context.\n\n"
        f"Context:\n{chunks_text}\n\nGenerated Answer:\n{answer}\n\n"
        "Respond in this exact JSON format:\n"
        '{"faithful": true or false, "score": 0.0 to 1.0, '
        '"unsupported_claims": ["claim 1"] or [], "verdict": "one sentence summary"}'
    )

    for attempt in range(settings.GEN_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_AUDITOR,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a factual claim auditor. Respond ONLY in valid JSON. "
                            "If the answer is a refusal, mark it as faithful (score: 1.0, unsupported_claims: [])."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.GEN_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            return parse_faithfulness_json(raw)

        except groq.APIConnectionError as e:
            msg = f"Auditor connection error: {e}"
            log.error(msg)
            return {"faithful": False, "score": 0.0, "unsupported_claims": [msg], "verdict": "Auditor connection failed."}

        except groq.RateLimitError:
            if attempt < settings.GEN_MAX_RETRIES - 1:
                wait = settings.GEN_RETRY_DELAYS[attempt]
                log.warning("Auditor rate-limited. Retrying in %ds.", wait)
                time.sleep(wait)
            else:
                msg = "Auditor rate limit exceeded."
                log.error(msg)
                return {"faithful": False, "score": 0.0, "unsupported_claims": [msg], "verdict": msg}

        except Exception as e:
            is_429 = "429" in str(e) or "rate limit" in str(e).lower()
            if is_429 and attempt < settings.GEN_MAX_RETRIES - 1:
                wait = settings.GEN_RETRY_DELAYS[attempt]
                log.warning("Auditor rate-limited (generic). Retrying in %ds.", wait)
                time.sleep(wait)
            else:
                msg = f"Auditor error: {e}"
                log.error(msg, exc_info=True)
                return {"faithful": False, "score": 0.0, "unsupported_claims": [msg], "verdict": "Auditor failed."}

    return {
        "faithful": False,
        "score": 0.0,
        "unsupported_claims": ["Auditor rate limit exceeded after all retries."],
        "verdict": "Auditing blocked by rate limits.",
    }


def parse_faithfulness_json(raw_text: str) -> dict:
    """Strips markdown fences and extracts valid JSON from auditor response."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
        raw_text = "\n".join(lines).strip()

    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        return json.loads(raw_text[start:end])
    except Exception as e:
        return {
            "faithful": False,
            "score": 0.0,
            "unsupported_claims": ["Failed to parse auditor JSON response."],
            "verdict": f"JSON parsing error: {e}",
        }


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------

def log_query(
    query: str,
    retrieved_chunks: list[dict],
    answer: str,
    log_path: str = None,
    pre_rerank_chunks: list[dict] = None,
    faithfulness_res: dict = None,
):
    """Appends a structured RAG interaction record to the retrieval log file."""
    if log_path is None:
        log_path = str(settings.LOG_FILE_PATH)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    faithful_str = "N/A"
    faith_score = 0.0
    faith_verdict = ""
    if faithfulness_res:
        faithful_str = "Yes" if faithfulness_res.get("faithful") else "No"
        faith_score = faithfulness_res.get("score", 0.0)
        faith_verdict = faithfulness_res.get("verdict", "")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Query     : {query}\n")
        f.write(f"Chunks    : {len(retrieved_chunks)}\n")
        f.write(f"Faithful  : {faithful_str} ({faith_score:.2f}) — \"{faith_verdict}\"\n")
        f.write("-" * 40 + "\n")
        for idx, c in enumerate(retrieved_chunks):
            preview = c["text"].replace("\n", " ")[:100]
            f.write(f"[{idx+1}] {c['doc_name']} chunk {c['chunk_index']} | score {c.get('score', 0.0):.4f}\n")
            f.write(f"    {preview}...\n")
        f.write(f"[Answer]\n{answer}\n")
        f.write("=" * 40 + "\n\n")
