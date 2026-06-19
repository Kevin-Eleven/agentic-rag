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
