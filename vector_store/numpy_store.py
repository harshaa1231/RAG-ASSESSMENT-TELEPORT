"""Pure NumPy vector store — fallback when FAISS is unavailable."""
from __future__ import annotations

import numpy as np


class NumpyVectorStore:
    """Brute-force cosine search using NumPy matrix multiplication.

    Suitable for small corpora (<10k chunks) or environments where
    faiss-cpu cannot be installed.

    Satisfies the VectorStore Protocol without inheriting from it.
    """

    def __init__(self, dimension: int) -> None:
        """Initialise an empty store.

        Args:
            dimension: Embedding vector dimension.
        """
        self._dimension = dimension
        self._embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        """Store L2-normalised embeddings.

        Args:
            embeddings: Float32 array of shape (n, dimension).
            chunk_ids: String IDs per embedding.
            texts: Raw texts for display.
        """
        normed = self._normalise(embeddings.astype(np.float32))
        if self._embeddings.shape[0] == 0:
            self._embeddings = normed
        else:
            self._embeddings = np.vstack([self._embeddings, normed])
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Cosine search via dot product on pre-normalised vectors.

        Args:
            query_embedding: Shape (1, d) or (d,).
            top_k: Number of results.

        Returns:
            Sorted (chunk_id, score) list, descending.
        """
        if self._embeddings.shape[0] == 0:
            return []
        q = self._normalise(query_embedding.reshape(1, -1).astype(np.float32))
        scores = (self._embeddings @ q.T).squeeze()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        k = min(top_k, len(self._chunk_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]

    def get_chunks(self) -> list[tuple[str, str]]:
        """Return all (chunk_id, text) pairs."""
        return list(zip(self._chunk_ids, self._texts))

    @staticmethod
    def _normalise(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return embeddings / norms
