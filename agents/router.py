from llm.prompts import ROUTING_PROMPT

from llm.groq_client import generate


def route_query(query):
    routing_prompt = ROUTING_PROMPT.format(query=query)
    routing_decision = generate(routing_prompt)
    return routing_decision.strip().upper() == "YES"
