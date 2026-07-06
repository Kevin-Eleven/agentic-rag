from unittest.mock import patch

from agents.router import route_query


@patch("agents.router.generate", return_value='{"needs_retrieval": true}')
def test_route_query_true_for_json_decision(mock_generate):
    assert route_query("What does the placement policy say about NOCs?") is True


@patch("agents.router.generate", return_value='{"needs_retrieval": false}')
def test_route_query_false_for_json_decision(mock_generate):
    assert route_query("What is 2+2?") is False


@patch("agents.router.generate", return_value='{"needs_retrieval": true}')
def test_route_query_requests_deterministic_json_output(mock_generate):
    route_query("anything")
    assert mock_generate.call_args.kwargs["json_mode"] is True
    assert mock_generate.call_args.kwargs["temperature"] == 0.0


@patch("agents.router.generate", return_value="  yes \n")
def test_route_query_falls_back_to_plain_text_yes(mock_generate):
    assert route_query("anything") is True


@patch("agents.router.generate", return_value="NO")
def test_route_query_falls_back_to_plain_text_no(mock_generate):
    assert route_query("anything") is False
