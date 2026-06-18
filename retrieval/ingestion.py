"""Ingest a PDF: extract text, chunk it, embed it, and store it in ChromaDB."""

import pymupdf
import chromadb

from config.settings import CHROMA_PATH
from retrieval.chunker import chunk_text
from retrieval.embeddings import embed_texts


def ingest_pdf(pdf_path, collection_name="pdf_chunks"):
    doc = pymupdf.open(pdf_path)

    page_texts = [page.get_text().replace("\n", " ").strip() for page in doc]
    full_text = " ".join(page_texts)

    chunks = chunk_text(full_text)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    collection.add(
        documents=chunks,
        embeddings=embed_texts(chunks),
        ids=[str(i) for i in range(len(chunks))],
    )

    return len(chunks)


if __name__ == "__main__":
    count = ingest_pdf("data/Placement Guidelines- 2026-27.pdf")
    print(f"Ingested {count} chunks.")
