from agents import evaluate_retrieval, rewrite_query, route_query, generate_answer
from llm import generate
from retrieval.vector_store import ChromaStore
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)
store = ChromaStore()


def run_workflow(query):

    if not route_query(query):
        return generate(query)

    rewritten = rewrite_query(query)

    context = store.retrieve(rewritten)

    is_relevant = evaluate_retrieval(rewritten, context)

    retries = 0
    while not is_relevant and retries < 3:
        retries += 1
        log_stage(logger, "retry_retrieval", attempt=retries, query=query)
        rewritten = rewrite_query(query)
        context = store.retrieve(rewritten)
        is_relevant = evaluate_retrieval(rewritten, context)

    if not is_relevant:
        log_stage(logger, "retrieval_failed", query=query, retries=retries)
        return "The retrieved context is not relevant after multiple attempts."

    return generate_answer(rewritten, context)
