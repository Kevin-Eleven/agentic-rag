from llm.groq_client import generate
from llm.prompts import ANSWER_GENERATION_PROMPT
from utils.logger import get_logger, log_stage, time_stage

logger = get_logger(__name__)


def format_context(chunks):
    """Render chunks as numbered sources so the model can cite them as [1], [2]…"""
    rendered = []
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            rendered.append(f"[{i}] ({source}, page {page})\n{chunk['text']}")
        else:
            rendered.append(f"[{i}]\n{chunk}")
    return "\n\n".join(rendered)


@time_stage("generate_answer")
def generate_answer(query, context):
    answer_prompt = ANSWER_GENERATION_PROMPT.format(query=query, context=format_context(context))
    answer = generate(answer_prompt)
    log_stage(logger, "generate_answer", query=query, answer_length=len(answer))
    return answer
