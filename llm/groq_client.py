import json
import time

import requests

from config.settings import (
    GENERATION_TEMPERATURE,
    GROQ_API_KEY,
    LLM_MAX_ATTEMPTS,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
)
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

URL = "https://api.groq.com/openai/v1/chat/completions"

# Rate limits and transient server errors are worth retrying; anything else
# (bad request, bad key) will not get better on a second attempt.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the Groq API call fails after all retry attempts."""


def generate(prompt, temperature=GENERATION_TEMPERATURE, json_mode=False):
    """Call the Groq chat-completions API and return the response text.

    Retries rate limits (429) and transient 5xx errors with exponential
    backoff. Set `json_mode=True` to force a JSON-object response (used by
    the router/grader so their decisions parse reliably).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL_NAME,
        "temperature": temperature,
        "max_completion_tokens": 300,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            response = requests.post(
                URL, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            last_error = f"request failed: {exc}"
            _log_retry(attempt, last_error)
            _backoff(attempt)
            continue

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        if response.status_code in _RETRYABLE_STATUS_CODES:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            _log_retry(attempt, last_error)
            _backoff(attempt)
            continue

        if response.status_code != 200:
            raise LLMError(f"Groq API error (HTTP {response.status_code}): {response.text[:500]}")

        body = response.json()
        usage = body.get("usage", {})
        log_stage(
            logger,
            "llm_call",
            model=MODEL_NAME,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return body["choices"][0]["message"]["content"]

    raise LLMError(f"Groq API call failed after {LLM_MAX_ATTEMPTS} attempts: {last_error}")


def parse_json_response(raw):
    """Parse a JSON-mode LLM response, returning None instead of raising on
    malformed output so callers can fall back to plain-text parsing."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _log_retry(attempt, error):
    log_stage(logger, "llm_retry", attempt=attempt, max_attempts=LLM_MAX_ATTEMPTS, error=error)


def _backoff(attempt):
    if attempt < LLM_MAX_ATTEMPTS:
        time.sleep(2 ** (attempt - 1))
