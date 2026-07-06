QUERY_REWRITE_PROMPT = """
Rewrite this question for semantic search.

Question:
{query}

Return only rewritten query.
"""

QUERY_REWRITE_RETRY_PROMPT = """
Rewrite this question for semantic search over a document collection.

Question:
{query}

These earlier rewrites retrieved irrelevant or insufficient context, so produce
a rewrite that is meaningfully different (use synonyms, expand abbreviations,
or shift the focus of the query):
{failed_rewrites}

Grader feedback on the last attempt:
{feedback}

Return only the rewritten query.
"""

ROUTING_PROMPT = """
Determine if the following query requires retrieval of external information to answer:

Query:
{query}

Reply with a JSON object: {{"needs_retrieval": true}} or {{"needs_retrieval": false}}
"""

RETRIEVAL_EVALUATION_PROMPT = """
Query:
{query}

Context:
{context}

Are the context chunks sufficient to answer the query?

Reply with a JSON object: {{"relevant": true or false, "reason": "<one short sentence>"}}
"""

ANSWER_GENERATION_PROMPT = """
Answer the following question based only on the provided context.
Cite the sources you use with bracketed numbers like [1] or [2] that refer to
the numbered context chunks. If the context does not contain the answer, say so.

Context:
{context}

Question:
{query}

Answer:
"""
