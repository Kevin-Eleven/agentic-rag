from llm.groq_client import generate
from llm.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_RETRY_PROMPT
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


@time_stage("rewrite")
def rewrite_query(query, failed_rewrites=None, feedback=None):
    """Rewrite the query for semantic search.

    On retries, `failed_rewrites` (earlier attempts that retrieved bad context)
    and `feedback` (the grader's reason) steer the model away from repeating
    the same rewrite.
    """
    if failed_rewrites:
        rewrite_prompt = QUERY_REWRITE_RETRY_PROMPT.format(
            query=query,
            failed_rewrites="\n".join(f"- {attempt}" for attempt in failed_rewrites),
            feedback=feedback or "no feedback available",
        )
    else:
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten_query = generate(rewrite_prompt).strip()
    log_stage(logger, "rewrite", original=query, rewritten=rewritten_query)
    return rewritten_query
