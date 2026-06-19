from llm.prompts import RETRIEVAL_EVALUATION_PROMPT

from llm.groq_client import generate
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


def evaluate_retrieval(query, relevant_chunks):
    retrieval_prompt = RETRIEVAL_EVALUATION_PROMPT.format(
        query=query, context=relevant_chunks
    )
    evaluation = generate(retrieval_prompt)
    is_relevant = evaluation.strip().upper() == "YES"
    log_stage(
        logger, "grade_retrieval", query=query, n_chunks=len(relevant_chunks), is_relevant=is_relevant
    )
    return 1 if is_relevant else 0
