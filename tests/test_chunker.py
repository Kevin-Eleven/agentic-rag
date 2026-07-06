from retrieval.chunker import chunk_pages, chunk_text


def test_chunk_text_splits_with_overlap():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "".join(chunks).count("a") >= 1200  # overlap duplicates characters


def test_chunk_text_short_input_is_single_chunk():
    assert chunk_text("short text") == ["short text"]


def test_chunk_pages_tags_each_chunk_with_its_page():
    chunks = chunk_pages(["page one text", "page two text"])
    assert chunks == [
        {"text": "page one text", "page": 1},
        {"text": "page two text", "page": 2},
    ]


def test_chunk_pages_skips_empty_pages():
    chunks = chunk_pages(["content", "", "   "])
    assert chunks == [{"text": "content", "page": 1}]


def test_chunk_pages_splits_long_pages_and_keeps_page_number():
    chunks = chunk_pages(["x" * 1200], chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(chunk["page"] == 1 for chunk in chunks)
