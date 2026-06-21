from llm.prompts import QUERY_REWRITE_PROMPT

from llm.groq_client import generate
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


@time_stage("rewrite")
def rewrite_query(query):
    rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten_query = generate(rewrite_prompt).strip()
    log_stage(logger, "rewrite", original=query, rewritten=rewritten_query)
    return rewritten_query
