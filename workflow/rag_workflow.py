from agents import evaluate_retrieval, rewrite_query, route_query, generate_answer
from llm import generate
from retrieval.vector_store import ChromaStore

store = ChromaStore()


def run_workflow(query):

    if not route_query(query):
        return generate(query)

    rewritten = rewrite_query(query)

    context = store.retrieve(rewritten)

    is_relevant = evaluate_retrieval(rewritten, context)

    retries = 0
    while not is_relevant and retries < 3:
        print("The retrieved context is not relevant. Retrying retrieval...")
        rewritten = rewrite_query(query)
        context = store.retrieve(rewritten)
        is_relevant = evaluate_retrieval(rewritten, context)
        retries += 1

    if not is_relevant:
        return "The retrieved context is not relevant after multiple attempts."

    return generate_answer(rewritten, context)
