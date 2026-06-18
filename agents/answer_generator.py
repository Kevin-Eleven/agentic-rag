from llm.prompts import ANSWER_GENERATION_PROMPT

from llm.groq_client import generate


def generate_answer(query, context):
    answer_prompt = ANSWER_GENERATION_PROMPT.format(query=query, context=context)
    return generate(answer_prompt)
