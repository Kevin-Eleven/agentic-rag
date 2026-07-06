from unittest.mock import patch

from agents.query_rewriter import rewrite_query


@patch(
    "agents.query_rewriter.generate",
    return_value="  off-campus job rules for CCDC students",
)
def test_rewrite_query_strips_whitespace(mock_generate):
    assert (
        rewrite_query("can i apply off campus")
        == "off-campus job rules for CCDC students"
    )


@patch("agents.query_rewriter.generate")
def test_rewrite_query_passes_original_query_into_prompt(mock_generate):
    mock_generate.return_value = "rewritten"
    rewrite_query("original question")
    prompt_used = mock_generate.call_args[0][0]
    assert "original question" in prompt_used


@patch("agents.query_rewriter.generate", return_value="a different rewrite")
def test_rewrite_query_retry_prompt_includes_failed_attempts_and_feedback(mock_generate):
    rewrite_query(
        "original question",
        failed_rewrites=["first failed rewrite"],
        feedback="chunks were about a different topic",
    )
    prompt_used = mock_generate.call_args[0][0]
    assert "original question" in prompt_used
    assert "first failed rewrite" in prompt_used
    assert "chunks were about a different topic" in prompt_used
