"""Ingest script for processing PDF documents and storing them in ChromaDB"""

# Purpose of file:
# Extract text from pdf, clean it, chunk it, embed it and store in chroma db


import pymupdf
import chromadb

doc = pymupdf.open("data/Placement Guidelines- 2026-27.pdf")

# create text chunks
chunks = []
for page in doc:
    text = page.get_text()
    chunks.append(text)


# Refine the chunks
cleaned_chunks = [chunk.replace("\n", " ").strip() for chunk in chunks]


# Create a ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection in ChromaDB
collection = client.create_collection(name="pdf_chunks")

collection.add(
    documents=cleaned_chunks, ids=[str(i) for i in range(len(cleaned_chunks))]
)
