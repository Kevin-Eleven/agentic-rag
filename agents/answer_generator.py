from llm.prompts import ANSWER_GENERATION_PROMPT

from llm.groq_client import generate
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


def generate_answer(query, context):
    answer_prompt = ANSWER_GENERATION_PROMPT.format(query=query, context=context)
    answer = generate(answer_prompt)
    log_stage(logger, "generate_answer", query=query, answer_length=len(answer))
    return answer
