from prompts import QUERY_REWRITE_PROMPT, RETRIEVAL_EVALUATION_PROMPT, ROUTING_PROMPT

from llm import generate


def multi_query_retrieval(query):
    pass


def rewrite_query(query):
    rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten_query = generate(rewrite_prompt)
    return rewritten_query.strip()


def route_query(query):
    routing_prompt = ROUTING_PROMPT.format(query=query)
    routing_decision = generate(routing_prompt)
    return routing_decision.strip().upper() == "YES"


def evaluate_retrieval(query, relevant_chunks):
    retrieval_prompt = RETRIEVAL_EVALUATION_PROMPT.format(
        query=query, context=relevant_chunks
    )
    evaluation = generate(retrieval_prompt)
    return 1 if evaluation == "YES" else 0


def reflection_agent(answer):
    pass
