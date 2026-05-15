"""Protocol definition for vector stores — no inheritance required."""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    """Structural interface satisfied by any object backed by a dense index."""

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        """Add pre-computed embeddings with their IDs and raw texts.

        Args:
            embeddings: Float32 array of shape (n, dimension).
            chunk_ids: Unique string identifier per embedding.
            texts: Raw text corresponding to each embedding (for BM25 / display).
        """
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, score) pairs sorted by descending score.

        Args:
            query_embedding: Float32 array of shape (1, dimension) or (dimension,).
            top_k: Number of results to return.
        """
        ...

    def get_chunks(self) -> list[tuple[str, str]]:
        """Return all (chunk_id, text) pairs in index order."""
        ...
