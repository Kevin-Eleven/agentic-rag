from eval.run_eval import chunk_matches, retrieval_metrics


def test_chunk_matches_on_keyword():
    item = {"expected_keywords": ["off-campus"]}
    assert chunk_matches({"text": "Rules for OFF-CAMPUS applications"}, item)
    assert not chunk_matches({"text": "on-campus drive schedule"}, item)


def test_chunk_matches_on_expected_page():
    item = {"expected_pages": [2]}
    assert chunk_matches({"text": "anything", "page": 2}, item)
    assert not chunk_matches({"text": "anything", "page": 5}, item)


def test_retrieval_metrics_hit_rate_and_mrr():
    items = [
        {"question": "q1", "expected_keywords": ["match"]},
        {"question": "q2", "expected_keywords": ["absent"]},
    ]

    def fake_retrieve(question):
        if question == "q1":
            # relevant chunk at rank 2 -> reciprocal rank 0.5
            return [{"text": "filler"}, {"text": "a match here"}]
        return [{"text": "filler"}, {"text": "more filler"}]

    hit_rate, mrr = retrieval_metrics(fake_retrieve, items)
    assert hit_rate == 0.5
    assert mrr == 0.25
