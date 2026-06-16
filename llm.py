from utils import ask_llm


def generate(prompt):
    response = ask_llm(prompt)
    return response["choices"][0]["message"]["content"]
