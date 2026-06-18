from llm.prompts import RETRIEVAL_EVALUATION_PROMPT

from llm.groq_client import generate


def evaluate_retrieval(query, relevant_chunks):
    retrieval_prompt = RETRIEVAL_EVALUATION_PROMPT.format(
        query=query, context=relevant_chunks
    )
    evaluation = generate(retrieval_prompt)
    return 1 if evaluation.strip().upper() == "YES" else 0
