from llm.prompts import QUERY_REWRITE_PROMPT

from llm.groq_client import generate


def rewrite_query(query):
    rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten_query = generate(rewrite_prompt)
    return rewritten_query.strip()
