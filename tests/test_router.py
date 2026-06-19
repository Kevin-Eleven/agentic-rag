from unittest.mock import patch

from agents.router import route_query


@patch("agents.router.generate", return_value="YES")
def test_route_query_true_for_yes(mock_generate):
    assert route_query("What does the placement policy say about NOCs?") is True


@patch("agents.router.generate", return_value="NO")
def test_route_query_false_for_no(mock_generate):
    assert route_query("What is 2+2?") is False


@patch("agents.router.generate", return_value="  yes \n")
def test_route_query_normalizes_whitespace_and_case(mock_generate):
    assert route_query("anything") is True
