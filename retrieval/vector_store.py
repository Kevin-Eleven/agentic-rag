import re

import chromadb
from rank_bm25 import BM25Okapi

from config.settings import CHROMA_PATH, TOP_K
from retrieval.embeddings import embed_query
from utils.logger import time_stage


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("pdf_chunks")
        self._bm25 = None
        self._bm25_ids = None
        self._id_to_doc = None

    def _build_bm25(self):
        corpus = self.collection.get()
        self._bm25_ids = corpus["ids"]
        self._id_to_doc = dict(zip(corpus["ids"], corpus["documents"]))
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in corpus["documents"]])

    def retrieve(self, query, n_results=TOP_K):
        results = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=n_results
        )
        return results["documents"][0]

    def retrieve_with_scores(self, query, n_results=TOP_K):
        """Like `retrieve`, but pairs each chunk with its distance score (for display/eval)."""
        results = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=n_results
        )
        return list(zip(results["documents"][0], results["distances"][0]))

    @time_stage("retrieve")
    def retrieve_hybrid(self, query, n_results=TOP_K, candidate_pool=20, rrf_k=60):
        """Hybrid retrieval: fuses BM25 keyword ranking with embedding-similarity ranking via
        reciprocal rank fusion, so a chunk only one signal favors can still surface."""
        if self._bm25 is None:
            self._build_bm25()

        pool = min(candidate_pool, len(self._bm25_ids))

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked_ids = [
            doc_id
            for doc_id, _ in sorted(zip(self._bm25_ids, bm25_scores), key=lambda pair: -pair[1])
        ][:pool]

        embedding_results = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=pool
        )
        embedding_ranked_ids = embedding_results["ids"][0]

        fused_scores = {}
        for rank, doc_id in enumerate(bm25_ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(embedding_ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

        top_ids = sorted(fused_scores, key=lambda doc_id: -fused_scores[doc_id])[:n_results]
        return [self._id_to_doc[doc_id] for doc_id in top_ids]
