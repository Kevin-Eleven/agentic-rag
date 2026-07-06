from llm.groq_client import generate, parse_json_response
from llm.prompts import RETRIEVAL_EVALUATION_PROMPT
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


def _format_chunks(chunks):
    return "\n\n".join(
        chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in chunks
    )


@time_stage("grade_retrieval")
def evaluate_retrieval(query, retrieved_chunks):
    """Grade whether the retrieved chunks can answer the query.

    Returns (is_relevant, reason); the reason feeds back into the query
    rewriter when the workflow retries a failed retrieval.
    """
    retrieval_prompt = RETRIEVAL_EVALUATION_PROMPT.format(
        query=query, context=_format_chunks(retrieved_chunks)
    )
    evaluation = generate(retrieval_prompt, temperature=0.0, json_mode=True)
    parsed = parse_json_response(evaluation)
    if parsed is not None and "relevant" in parsed:
        is_relevant = bool(parsed["relevant"])
        reason = parsed.get("reason", "")
    else:
        # Fallback for models that ignore JSON mode and answer in plain text.
        is_relevant = "YES" in evaluation.strip().upper()
        reason = evaluation.strip()
    log_stage(
        logger,
        "grade_retrieval",
        query=query,
        n_chunks=len(retrieved_chunks),
        is_relevant=is_relevant,
        reason=reason,
    )
    return is_relevant, reason
