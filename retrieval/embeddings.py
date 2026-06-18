"""Wraps sentence-transformers so embedding logic lives in one place instead of
relying on ChromaDB's implicit default model."""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    """Embed a list of strings, returns a list of embedding vectors (lists of floats)."""
    return _get_model().encode(list(texts)).tolist()


def embed_query(text):
    """Embed a single query string, returns a single embedding vector."""
    return _get_model().encode([text])[0].tolist()
