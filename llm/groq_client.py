import requests

from config.settings import GROQ_API_KEY, MODEL_NAME

URL = "https://api.groq.com/openai/v1/chat/completions"


def generate(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL_NAME,
        "temperature": 0.7,
        "max_completion_tokens": 300,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    response = requests.post(URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
