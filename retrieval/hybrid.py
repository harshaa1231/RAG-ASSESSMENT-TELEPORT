"""Hybrid retrieval: Reciprocal Rank Fusion of dense FAISS and sparse BM25."""
from __future__ import annotations

import numpy as np

from embeddings.base import EmbeddingModel
from embeddings.sparse import BM25Store
from vector_store.base import VectorStore


def _rrf(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d)).
    Documents appearing in only one list still receive their single-list score.

    Args:
        dense_results: (chunk_id, score) list from FAISS, best-first.
        sparse_results: (chunk_id, score) list from BM25, best-first.
        k: RRF smoothing constant (default 60 per the original paper).

    Returns:
        Merged and re-ranked (chunk_id, rrf_score) list, descending.
    """
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _) in enumerate(sparse_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Fuses dense (cosine) and sparse (BM25) retrieval via RRF.

    Strategy: retrieve top_k * 2 candidates from each ranker, fuse scores with
    RRF, then return the top_k fused results.  No query expansion is applied.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        rrf_k: int = 60,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            embedding_model: Dense embedding model.
            vector_store: FAISS or NumPy dense index.
            bm25_store: Initialised BM25 sparse index.
            rrf_k: RRF smoothing constant.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[tuple[str, float]], np.ndarray]:
        """Fuse dense and sparse results and return top_k.

        Args:
            query: Raw query string.
            top_k: Final number of results after fusion.

        Returns:
            Tuple of:
              - Fused (chunk_id, rrf_score) list, descending.
              - Query embedding (for potential cosine_drift computation upstream).
        """
        query_embedding = self.embedding_model.embed_query(query)
        dense = self.vector_store.search(query_embedding, top_k * 2)
        sparse = self.bm25_store.search(query, top_k * 2)
        fused = _rrf(dense, sparse, k=self.rrf_k)
        return fused[:top_k], query_embedding
