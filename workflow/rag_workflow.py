from functools import lru_cache

from agents import evaluate_retrieval, generate_answer, rewrite_query, route_query
from config.settings import MAX_RETRIEVAL_RETRIES
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


def _trace(query, answer, needs_retrieval=False, rewritten_query=None, retrieved_chunks=(),
           retry_count=0, grader_feedback=None):
    return {
        "query": query,
        "needs_retrieval": needs_retrieval,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": list(retrieved_chunks),
        "retry_count": retry_count,
        "grader_feedback": grader_feedback,
        "answer": answer,
    }


@lru_cache(maxsize=128)
def _run_workflow_cached(query, return_trace=False):

    needs_retrieval = route_query(query)

    if not needs_retrieval:
        answer = generate(query)
        if return_trace:
            return _trace(query, answer)
        return answer

    rewritten = rewrite_query(query)
    context = store.retrieve_hybrid(rewritten)
    is_relevant, feedback = evaluate_retrieval(rewritten, context)

    # Self-correction loop: each retry tells the rewriter which rewrites already
    # failed and why, so it produces a genuinely different query instead of
    # re-rolling the same one.
    failed_rewrites = []
    retries = 0
    while not is_relevant and retries < MAX_RETRIEVAL_RETRIES:
        retries += 1
        failed_rewrites.append(rewritten)
        log_stage(logger, "retry_retrieval", attempt=retries, query=query, feedback=feedback)
        rewritten = rewrite_query(query, failed_rewrites=failed_rewrites, feedback=feedback)
        context = store.retrieve_hybrid(rewritten)
        is_relevant, feedback = evaluate_retrieval(rewritten, context)

    if not is_relevant:
        log_stage(logger, "retrieval_failed", query=query, retries=retries)
        answer = generate(query)
        if return_trace:
            return _trace(
                query, answer, needs_retrieval=True, rewritten_query=rewritten,
                retrieved_chunks=context, retry_count=retries, grader_feedback=feedback,
            )
        return answer

    answer = generate_answer(rewritten, context)
    if return_trace:
        return _trace(
            query, answer, needs_retrieval=True, rewritten_query=rewritten,
            retrieved_chunks=context, retry_count=retries, grader_feedback=feedback,
        )
    return answer
