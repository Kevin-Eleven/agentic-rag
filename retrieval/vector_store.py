import chromadb

from config.settings import CHROMA_PATH, TOP_K
from retrieval.embeddings import embed_query


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("pdf_chunks")

    def retrieve(self, query, n_results=TOP_K):
        results = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=n_results
        )
        return results["documents"][0]

    def retrieve_with_scores(self, query, n_results=TOP_K):
        """Like `retrieve`, but pairs each chunk with its distance score (for display/eval)."""
        results = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=n_results
        )
        return list(zip(results["documents"][0], results["distances"][0]))
