from agents import evaluate_retrieval, rewrite_query, route_query, retreival_evaluation
from llm import generate
from retriever import retrieve


def run_workflow(query):

    if not route_query(query):
        return generate(query)

    rewritten = rewrite_query(query)

    context = retrieve(rewritten)

    is_relevant = evaluate_retrieval(rewritten, context)

    retries = 0
    while not is_relevant and retries < 3:
        print("The retrieved context is not relevant. Retrying retrieval...")
        rewritten = rewrite_query(query)
        context = retrieve(rewritten)
        is_relevant = evaluate_retrieval(rewritten, context)
        retries += 1

    if not is_relevant:
        return "The retrieved context is not relevant after multiple attempts."

    prompt = f"Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {rewritten}\n\nAnswer:"

    return generate(prompt)
