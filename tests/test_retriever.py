from unittest.mock import MagicMock, patch

from retrieval.vector_store import ChromaStore


@patch("retrieval.vector_store.embed_query", return_value=[0.1, 0.2, 0.3])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_retrieve_embeds_query_and_returns_chunks_with_metadata(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[{"source": "doc.pdf", "page": 1}, {"source": "doc.pdf", "page": 2}]],
    }
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    results = store.retrieve("what are the off-campus rules?")

    mock_embed_query.assert_called_once_with("what are the off-campus rules?")
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]], n_results=3
    )
    assert results == [
        {"text": "chunk one", "source": "doc.pdf", "page": 1},
        {"text": "chunk two", "source": "doc.pdf", "page": 2},
    ]


@patch("retrieval.vector_store.embed_query", return_value=[0.1, 0.2, 0.3])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_retrieve_handles_legacy_chunks_without_metadata(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [["chunk one", "chunk two"]]}
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    results = store.retrieve("what are the off-campus rules?")

    assert results == [{"text": "chunk one"}, {"text": "chunk two"}]


@patch("retrieval.vector_store.embed_query", return_value=[0.1])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_retrieve_respects_custom_n_results(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [["a", "b", "c", "d", "e"]]}
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    store.retrieve("query", n_results=5)

    mock_collection.query.assert_called_once_with(query_embeddings=[[0.1]], n_results=5)


@patch("retrieval.vector_store.embed_query", return_value=[0.1, 0.2, 0.3])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_retrieve_hybrid_fuses_bm25_and_embedding_rankings(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["0", "1", "2"],
        "documents": [
            "off-campus placement rules",
            "on-campus drive schedule",
            "eligibility criteria",
        ],
        "metadatas": [{"page": 1}, {"page": 2}, {"page": 3}],
    }
    mock_collection.query.return_value = {
        "documents": [["eligibility criteria", "off-campus placement rules"]],
        "ids": [["2", "0"]],
    }
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    results = store.retrieve_hybrid("off-campus rules", n_results=2)

    texts = [chunk["text"] for chunk in results]
    assert "off-campus placement rules" in texts
    assert len(results) == 2
    assert all("page" in chunk for chunk in results)


@patch("retrieval.vector_store.embed_query", return_value=[0.1, 0.2, 0.3])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_retrieve_hybrid_builds_bm25_index_only_once(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["0", "1"],
        "documents": ["off-campus placement rules", "on-campus drive schedule"],
    }
    mock_collection.query.return_value = {
        "documents": [["off-campus placement rules"]],
        "ids": [["0"]],
    }
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    store.retrieve_hybrid("off-campus", n_results=1)
    store.retrieve_hybrid("on-campus", n_results=1)

    mock_collection.get.assert_called_once()


@patch("retrieval.vector_store.embed_query", return_value=[0.1, 0.2, 0.3])
@patch("retrieval.vector_store.chromadb.PersistentClient")
def test_refresh_reopens_collection_and_drops_bm25_index(mock_client_cls, mock_embed_query):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["0"],
        "documents": ["off-campus placement rules"],
    }
    mock_collection.query.return_value = {
        "documents": [["off-campus placement rules"]],
        "ids": [["0"]],
    }
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    store = ChromaStore()
    store.retrieve_hybrid("off-campus", n_results=1)
    store.refresh()
    store.retrieve_hybrid("off-campus", n_results=1)

    assert mock_collection.get.call_count == 2
