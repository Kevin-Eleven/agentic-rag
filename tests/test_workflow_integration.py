from unittest.mock import patch

from workflow import rag_workflow


@patch("workflow.rag_workflow.route_query", return_value=False)
@patch("workflow.rag_workflow.generate", return_value="4")
def test_general_knowledge_query_skips_retrieval(mock_generate, mock_route_query):
    answer = rag_workflow.run_workflow("What is 2+2?")
    assert answer == "4"
    mock_generate.assert_called_once_with("What is 2+2?")


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=1)
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten query")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_relevant_retrieval_generates_answer_without_retry(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.return_value = ["relevant chunk"]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "Final answer"
    mock_store.retrieve_hybrid.assert_called_once_with("rewritten query")
    mock_generate_answer.assert_called_once_with("rewritten query", ["relevant chunk"])


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch("workflow.rag_workflow.evaluate_retrieval", side_effect=[0, 1])
@patch("workflow.rag_workflow.rewrite_query", side_effect=["rewrite 1", "rewrite 2"])
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_self_corrects_after_one_bad_retrieval(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.side_effect = [["irrelevant chunk"], ["relevant chunk"]]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "Final answer"
    assert mock_store.retrieve_hybrid.call_count == 2
    mock_generate_answer.assert_called_once_with("rewrite 2", ["relevant chunk"])


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate", return_value="answered without context")
@patch("workflow.rag_workflow.generate_answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=0)
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_falls_back_to_no_context_generation_after_three_retries(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_generate, mock_store
):
    mock_store.retrieve_hybrid.return_value = ["irrelevant chunk"]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "answered without context"
    assert mock_store.retrieve_hybrid.call_count == 4
    mock_generate_answer.assert_not_called()
    mock_generate.assert_called_once_with("Can I apply off-campus while registered with CCDC?")
