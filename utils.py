import requests
import os
import dotenv

dotenv.load_dotenv()

# Your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}",
}


def ask_llm(question):
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "max_completion_tokens": 300,
        "top_p": 1,
        "stream": False,  # change to True for streaming
        "stop": None,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()
