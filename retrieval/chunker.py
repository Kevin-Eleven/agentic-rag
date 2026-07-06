"""Creates chunks from a file"""


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_pages(page_texts, chunk_size=500, overlap=100):
    """Chunk each page separately so every chunk keeps its page provenance,
    which is what lets answers cite "source.pdf, page 4".

    Args:
        page_texts (list[str]): One string per page, in page order.

    Returns:
        list[dict]: Chunks shaped like {"text": str, "page": int} (1-indexed pages).
    """
    chunks = []
    for page_number, text in enumerate(page_texts, start=1):
        for piece in chunk_text(text, chunk_size, overlap):
            if piece.strip():
                chunks.append({"text": piece, "page": page_number})
    return chunks
