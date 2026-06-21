import time

import requests

from config.settings import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

URL = "https://api.groq.com/openai/v1/chat/completions"


def generate(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL_NAME,
        "temperature": 0.7,
        "max_completion_tokens": 300,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    start = time.perf_counter()
    response = requests.post(URL, headers=headers, json=payload)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    body = response.json()
    usage = body.get("usage", {})
    log_stage(
        logger,
        "llm_call",
        model=MODEL_NAME,
        elapsed_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
    return body["choices"][0]["message"]["content"]
