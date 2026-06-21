"""Structured logging for the RAG pipeline.

Each pipeline stage logs one line via `log_stage`, giving a readable trace of a
run: router decision -> rewritten query -> retrieval scores -> retries -> answer.
"""

import functools
import logging
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Third-party libraries (sentence-transformers, httpx, huggingface_hub) log at INFO
# and would otherwise drown out the pipeline trace.
_PACKAGE_LOGGER = "agents", "workflow", "retrieval", "llm"


def get_logger(name):
    logger = logging.getLogger(name)
    if name.split(".")[0] in _PACKAGE_LOGGER:
        logger.setLevel(logging.INFO)
    return logger


def log_stage(logger, stage, **fields):
    """Log one pipeline stage as a single structured line, e.g.
    log_stage(logger, "retrieve", query=q, n_chunks=3, retry=0)
    -> "retrieve | query='...' n_chunks=3 retry=0"
    """
    rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("%s | %s", stage, rendered)


def time_stage(stage_name):
    """Decorator that logs how long a pipeline stage took, e.g.
    @time_stage("rewrite") -> "timing | stage='rewrite' elapsed_ms=123.4"
    """

    def decorator(func):
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            log_stage(logger, "timing", stage_name=stage_name, elapsed_ms=elapsed_ms)
            return result

        return wrapper

    return decorator
