import chromadb

from config.settings import CHROMA_PATH, TOP_K


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection("pdf_chunks")

    def retrieve(self, query, n_results=TOP_K):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0]
