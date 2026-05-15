"""Strategy A: raw vector retrieval with no query expansion."""
from __future__ import annotations

import numpy as np

from embeddings.base import EmbeddingModel
from vector_store.base import VectorStore


class RawVectorRetriever:
    """Encodes the query as-is and searches the dense FAISS index.

    This is the baseline strategy.  No expansion, no rewriting — pure
    embedding similarity between the query and each corpus chunk.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        """Initialise Strategy A.

        Args:
            embedding_model: Satisfies EmbeddingModel Protocol.
            vector_store: Satisfies VectorStore Protocol (FAISS or NumPy).
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[tuple[str, float]], np.ndarray]:
        """Retrieve the top_k most similar chunks for the query.

        Args:
            query: Raw query string.
            top_k: Maximum results to return.

        Returns:
            Tuple of:
              - List of (chunk_id, cosine_score) in descending order.
              - Query embedding array of shape (1, dimension) — used for
                cosine_drift comparisons in Strategy B.
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results, query_embedding
