QUERY_REWRITE_PROMPT = """
Rewrite this question for semantic search.

Question:
{query}

Return only rewritten query.
"""

ROUTING_PROMPT = """
Determine if the following query requires retrieval of external information to answer:

Query:
{query}

Reply with only YES or NO.
"""

RETRIEVAL_EVALUATION_PROMPT = """
Query:
{query}

Context:
{context}

Are the context chunks sufficient to answer the query?

Reply with only YES or NO.
"""
