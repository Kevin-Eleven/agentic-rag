from unittest.mock import patch

import pytest

from workflow import rag_workflow

CHUNK = {"text": "relevant chunk", "source": "doc.pdf", "page": 1}


@pytest.fixture(autouse=True)
def clear_cache():
    """run_workflow is LRU-cached; without this, tests that reuse a query
    would silently get another test's cached answer."""
    rag_workflow.clear_workflow_cache()
    yield
    rag_workflow.clear_workflow_cache()


@patch("workflow.rag_workflow.route_query", return_value=False)
@patch("workflow.rag_workflow.generate", return_value="4")
def test_general_knowledge_query_skips_retrieval(mock_generate, mock_route_query):
    answer = rag_workflow.run_workflow("What is 2+2?")
    assert answer == "4"
    mock_generate.assert_called_once_with("What is 2+2?")


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=(True, "chunks cover the query"))
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten query")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_relevant_retrieval_generates_answer_without_retry(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.return_value = [CHUNK]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "Final answer"
    mock_store.retrieve_hybrid.assert_called_once_with("rewritten query")
    mock_generate_answer.assert_called_once_with("rewritten query", [CHUNK])


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch(
    "workflow.rag_workflow.evaluate_retrieval",
    side_effect=[(False, "chunks are off-topic"), (True, "chunks cover the query")],
)
@patch("workflow.rag_workflow.rewrite_query", side_effect=["rewrite 1", "rewrite 2"])
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_self_corrects_after_one_bad_retrieval(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.side_effect = [[{"text": "irrelevant chunk"}], [CHUNK]]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "Final answer"
    assert mock_store.retrieve_hybrid.call_count == 2
    mock_generate_answer.assert_called_once_with("rewrite 2", [CHUNK])
    # The retry must tell the rewriter what already failed and why.
    mock_rewrite_query.assert_called_with(
        "Can I apply off-campus while registered with CCDC?",
        failed_rewrites=["rewrite 1"],
        feedback="chunks are off-topic",
    )


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate", return_value="answered without context")
@patch("workflow.rag_workflow.generate_answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=(False, "chunks are off-topic"))
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_falls_back_to_no_context_generation_after_three_retries(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval,
    mock_generate_answer, mock_generate, mock_store,
):
    mock_store.retrieve_hybrid.return_value = [{"text": "irrelevant chunk"}]

    answer = rag_workflow.run_workflow("Can I apply off-campus while registered with CCDC?")

    assert answer == "answered without context"
    assert mock_store.retrieve_hybrid.call_count == 4
    mock_generate_answer.assert_not_called()
    mock_generate.assert_called_once_with("Can I apply off-campus while registered with CCDC?")


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=(True, "chunks cover the query"))
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten query")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_trace_exposes_pipeline_decisions(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.return_value = [CHUNK]

    trace = rag_workflow.run_workflow("What does the policy say?", return_trace=True)

    assert trace["answer"] == "Final answer"
    assert trace["needs_retrieval"] is True
    assert trace["rewritten_query"] == "rewritten query"
    assert trace["retrieved_chunks"] == [CHUNK]
    assert trace["retry_count"] == 0
    assert trace["grader_feedback"] == "chunks cover the query"


@patch("workflow.rag_workflow.store")
@patch("workflow.rag_workflow.generate_answer", return_value="Final answer")
@patch("workflow.rag_workflow.evaluate_retrieval", return_value=(True, "ok"))
@patch("workflow.rag_workflow.rewrite_query", return_value="rewritten query")
@patch("workflow.rag_workflow.route_query", return_value=True)
def test_repeated_query_is_served_from_cache(
    mock_route_query, mock_rewrite_query, mock_evaluate_retrieval, mock_generate_answer, mock_store
):
    mock_store.retrieve_hybrid.return_value = [CHUNK]

    rag_workflow.run_workflow("same question")
    rag_workflow.run_workflow("same question")

    mock_route_query.assert_called_once()
    mock_generate_answer.assert_called_once()
