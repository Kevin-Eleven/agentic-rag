"""Ingest a PDF: extract text, chunk it per page, embed it, and store it in ChromaDB."""

import argparse
import os

import chromadb
import pymupdf

from config.settings import CHROMA_PATH, COLLECTION_NAME
from retrieval.chunker import chunk_pages
from retrieval.embeddings import embed_texts


def ingest_pdf(pdf_path, collection_name=COLLECTION_NAME):
    doc = pymupdf.open(pdf_path)

    page_texts = [page.get_text().replace("\n", " ").strip() for page in doc]

    chunks = chunk_pages(page_texts)
    source = os.path.basename(pdf_path)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embed_texts([chunk["text"] for chunk in chunks]),
        metadatas=[{"source": source, "page": chunk["page"]} for chunk in chunks],
        ids=[f"{source}:{i}" for i in range(len(chunks))],
    )

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})",
    )
    args = parser.parse_args()

    count = ingest_pdf(args.pdf_path, collection_name=args.collection)
    print(f"Ingested {count} chunks from {args.pdf_path}.")


if __name__ == "__main__":
    main()
