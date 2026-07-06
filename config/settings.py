import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_chunks")

# How many chunks to hand to the answer generator.
TOP_K = int(os.getenv("TOP_K", "3"))

# LLM call behaviour.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))

# How many times the workflow re-rewrites the query after a failed retrieval grade.
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "3"))
