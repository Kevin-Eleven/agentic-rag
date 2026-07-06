from unittest.mock import MagicMock, patch

import pytest
import requests

from llm.groq_client import LLMError, generate, parse_json_response


def _response(status_code=200, content="hello", text="error body"):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return response


@patch("llm.groq_client.requests.post", return_value=_response(content="the answer"))
def test_generate_returns_message_content(mock_post):
    assert generate("prompt") == "the answer"
    assert mock_post.call_args.kwargs["timeout"] > 0


@patch("llm.groq_client.requests.post", return_value=_response())
def test_generate_json_mode_sets_response_format(mock_post):
    generate("prompt", json_mode=True)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["response_format"] == {"type": "json_object"}


@patch("llm.groq_client.time.sleep")
@patch(
    "llm.groq_client.requests.post",
    side_effect=[_response(status_code=429), _response(content="recovered")],
)
def test_generate_retries_rate_limit_then_succeeds(mock_post, mock_sleep):
    assert generate("prompt") == "recovered"
    assert mock_post.call_count == 2


@patch("llm.groq_client.time.sleep")
@patch(
    "llm.groq_client.requests.post",
    side_effect=[requests.ConnectionError("boom"), _response(content="recovered")],
)
def test_generate_retries_network_error_then_succeeds(mock_post, mock_sleep):
    assert generate("prompt") == "recovered"
    assert mock_post.call_count == 2


@patch("llm.groq_client.requests.post", return_value=_response(status_code=401))
def test_generate_raises_immediately_on_non_retryable_error(mock_post):
    with pytest.raises(LLMError):
        generate("prompt")
    assert mock_post.call_count == 1


@patch("llm.groq_client.time.sleep")
@patch("llm.groq_client.requests.post", return_value=_response(status_code=503))
def test_generate_raises_after_exhausting_retries(mock_post, mock_sleep):
    with pytest.raises(LLMError):
        generate("prompt")
    assert mock_post.call_count == 3


def test_parse_json_response_valid_object():
    assert parse_json_response('{"relevant": true}') == {"relevant": True}


def test_parse_json_response_rejects_malformed_and_non_objects():
    assert parse_json_response("YES") is None
    assert parse_json_response("[1, 2]") is None
    assert parse_json_response(None) is None
