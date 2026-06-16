import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")


def retrieve(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]
