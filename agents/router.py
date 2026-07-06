from llm.groq_client import generate, parse_json_response
from llm.prompts import ROUTING_PROMPT
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


@time_stage("route")
def route_query(query):
    routing_prompt = ROUTING_PROMPT.format(query=query)
    routing_decision = generate(routing_prompt, temperature=0.0, json_mode=True)
    parsed = parse_json_response(routing_decision)
    if parsed is not None and "needs_retrieval" in parsed:
        needs_retrieval = bool(parsed["needs_retrieval"])
    else:
        # Fallback for models that ignore JSON mode and answer in plain text.
        needs_retrieval = "YES" in routing_decision.strip().upper()
    log_stage(logger, "route", query=query, needs_retrieval=needs_retrieval)
    return needs_retrieval
