from unittest.mock import patch

from agents.retrieval_grader import evaluate_retrieval


@patch(
    "agents.retrieval_grader.generate",
    return_value='{"relevant": true, "reason": "chunks cover the query"}',
)
def test_evaluate_retrieval_relevant(mock_generate):
    is_relevant, reason = evaluate_retrieval("query", [{"text": "relevant chunk"}])
    assert is_relevant is True
    assert reason == "chunks cover the query"


@patch(
    "agents.retrieval_grader.generate",
    return_value='{"relevant": false, "reason": "chunks are about a different topic"}',
)
def test_evaluate_retrieval_irrelevant(mock_generate):
    is_relevant, reason = evaluate_retrieval("query", [{"text": "unrelated chunk"}])
    assert is_relevant is False
    assert reason == "chunks are about a different topic"


@patch("agents.retrieval_grader.generate", return_value="YES")
def test_evaluate_retrieval_falls_back_to_plain_text(mock_generate):
    is_relevant, _ = evaluate_retrieval("query", [{"text": "relevant chunk"}])
    assert is_relevant is True


@patch("agents.retrieval_grader.generate", return_value='{"relevant": true, "reason": ""}')
def test_evaluate_retrieval_formats_chunk_text_into_prompt(mock_generate):
    evaluate_retrieval("query", [{"text": "chunk body", "source": "doc.pdf", "page": 2}])
    prompt_used = mock_generate.call_args[0][0]
    assert "chunk body" in prompt_used
