from llm.prompts import ROUTING_PROMPT

from llm.groq_client import generate
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


@time_stage("route")
def route_query(query):
    routing_prompt = ROUTING_PROMPT.format(query=query)
    routing_decision = generate(routing_prompt)
    needs_retrieval = routing_decision.strip().upper() == "YES"
    log_stage(logger, "route", query=query, needs_retrieval=needs_retrieval)
    return needs_retrieval
