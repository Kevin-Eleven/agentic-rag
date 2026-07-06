"""Evaluation harness for the RAG pipeline.

Measures two things over a golden question set (eval/golden_set.json):

1. Retrieval quality — hit rate and MRR for dense-only vs hybrid retrieval,
   where a "hit" is a retrieved chunk matching the item's expected keywords
   (or expected pages, if provided).
2. Answer quality — an LLM judge scores each generated answer for correctness
   against the golden answer and faithfulness to the retrieved context (1-5).

Usage:
    python -m eval.run_eval                # full eval (needs GROQ_API_KEY)
    python -m eval.run_eval --skip-judge   # retrieval metrics only (no LLM calls)

Prints a markdown table you can paste into the README. Curate golden_set.json
for whichever document you ingested — keywords/pages must exist in that document.
"""

import argparse
import json
import os
from statistics import mean

from llm.groq_client import generate, parse_json_response
from retrieval.vector_store import ChromaStore
from workflow.rag_workflow import run_workflow

DEFAULT_GOLDEN_SET = os.path.join(os.path.dirname(__file__), "golden_set.json")

JUDGE_PROMPT = """
You are grading an answer produced by a RAG system.

Question:
{question}

Reference (golden) answer:
{golden_answer}

Retrieved context the system used:
{context}

Generated answer:
{answer}

Score the generated answer on two axes, each an integer from 1 (worst) to 5 (best):
- "correctness": does it agree with the reference answer?
- "faithfulness": is every claim supported by the retrieved context?

Reply with a JSON object: {{"correctness": <int>, "faithfulness": <int>}}
"""


def load_golden_set(path):
    with open(path) as f:
        return json.load(f)


def chunk_matches(chunk, item):
    """A chunk is a hit if it contains an expected keyword or comes from an
    expected page."""
    text = chunk["text"].lower()
    if any(keyword.lower() in text for keyword in item.get("expected_keywords", [])):
        return True
    return chunk.get("page") in item.get("expected_pages", [])


def retrieval_metrics(retrieve_fn, items):
    """Hit rate (any relevant chunk in top-k) and MRR over the golden set."""
    hits, reciprocal_ranks = [], []
    for item in items:
        chunks = retrieve_fn(item["question"])
        ranks = [rank for rank, chunk in enumerate(chunks, start=1) if chunk_matches(chunk, item)]
        hits.append(1 if ranks else 0)
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
    return mean(hits), mean(reciprocal_ranks)


def judge_answer(item, trace):
    context = "\n\n".join(
        chunk["text"] if isinstance(chunk, dict) else str(chunk)
        for chunk in trace["retrieved_chunks"]
    )
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        golden_answer=item["golden_answer"],
        context=context or "(no context retrieved)",
        answer=trace["answer"],
    )
    parsed = parse_json_response(generate(prompt, temperature=0.0, json_mode=True)) or {}
    return parsed.get("correctness"), parsed.get("faithfulness")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument("--golden-set", default=DEFAULT_GOLDEN_SET)
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only compute retrieval metrics (no LLM calls needed)",
    )
    args = parser.parse_args()

    items = load_golden_set(args.golden_set)
    store = ChromaStore()

    print(f"Evaluating on {len(items)} golden questions\n")

    dense_hit, dense_mrr = retrieval_metrics(store.retrieve, items)
    hybrid_hit, hybrid_mrr = retrieval_metrics(store.retrieve_hybrid, items)

    print("### Retrieval quality\n")
    print("| Retriever | Hit rate | MRR |")
    print("|---|---|---|")
    print(f"| Dense (embeddings only) | {dense_hit:.2f} | {dense_mrr:.2f} |")
    print(f"| Hybrid (BM25 + embeddings, RRF) | {hybrid_hit:.2f} | {hybrid_mrr:.2f} |")

    if args.skip_judge:
        return

    correctness_scores, faithfulness_scores = [], []
    print("\n### Answer quality (LLM as judge, 1-5)\n")
    print("| Question | Correctness | Faithfulness | Retries |")
    print("|---|---|---|---|")
    for item in items:
        trace = run_workflow(item["question"], return_trace=True)
        correctness, faithfulness = judge_answer(item, trace)
        if correctness is not None:
            correctness_scores.append(correctness)
        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)
        print(
            f"| {item['question'][:60]} | {correctness} | {faithfulness} "
            f"| {trace['retry_count']} |"
        )

    if correctness_scores:
        print(
            f"| **Average** | **{mean(correctness_scores):.2f}** "
            f"| **{mean(faithfulness_scores):.2f}** | |"
        )


if __name__ == "__main__":
    main()
