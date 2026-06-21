from functools import lru_cache

from agents import evaluate_retrieval, rewrite_query, route_query, generate_answer
from llm import generate
from retrieval.vector_store import ChromaStore
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)
store = ChromaStore()


def run_workflow(query, return_trace=False):
    """Cached entry point: identical (query, return_trace) pairs skip the whole
    router/retrieve/grade/generate pipeline and return the prior result."""
    before_hits = _run_workflow_cached.cache_info().hits
    result = _run_workflow_cached(query, return_trace)
    cache_hit = _run_workflow_cached.cache_info().hits > before_hits
    log_stage(logger, "workflow_cache", query=query, cache_hit=cache_hit)
    return result


def clear_workflow_cache():
    _run_workflow_cached.cache_clear()


@lru_cache(maxsize=128)
def _run_workflow_cached(query, return_trace=False):

    needs_retrieval = route_query(query)

    if not needs_retrieval:
        answer = generate(query)
        if return_trace:
            return {
                "query": query,
                "needs_retrieval": False,
                "rewritten_query": None,
                "retrieved_chunks": [],
                "retry_count": 0,
                "answer": answer,
            }
        return answer

    rewritten = rewrite_query(query)

    context = store.retrieve_hybrid(rewritten)

    is_relevant = evaluate_retrieval(rewritten, context)

    retries = 0
    while not is_relevant and retries < 3:
        retries += 1
        log_stage(logger, "retry_retrieval", attempt=retries, query=query)
        rewritten = rewrite_query(query)
        context = store.retrieve_hybrid(rewritten)
        is_relevant = evaluate_retrieval(rewritten, context)

    if not is_relevant:
        log_stage(logger, "retrieval_failed", query=query, retries=retries)
        answer = generate(query)
        if return_trace:
            return {
                "query": query,
                "needs_retrieval": True,
                "rewritten_query": rewritten,
                "retrieved_chunks": context,
                "retry_count": retries,
                "answer": answer,
            }
        return answer

    answer = generate_answer(rewritten, context)
    if return_trace:
        return {
            "query": query,
            "needs_retrieval": True,
            "rewritten_query": rewritten,
            "retrieved_chunks": context,
            "retry_count": retries,
            "answer": answer,
        }
    return answer
