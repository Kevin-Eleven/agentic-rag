from unittest.mock import patch

from agents.retrieval_grader import evaluate_retrieval


@patch("agents.retrieval_grader.generate", return_value="YES")
def test_evaluate_retrieval_relevant(mock_generate):
    assert evaluate_retrieval("query", ["relevant chunk"]) == 1


@patch("agents.retrieval_grader.generate", return_value="NO")
def test_evaluate_retrieval_irrelevant(mock_generate):
    assert evaluate_retrieval("query", ["unrelated chunk"]) == 0
