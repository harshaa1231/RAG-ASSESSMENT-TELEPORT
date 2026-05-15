"""FAISS-backed vector store using IndexFlatIP with L2 normalisation for cosine."""
from __future__ import annotations

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAISS_AVAILABLE = False


def _safe_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Copy and L2-normalise; handles zero vectors to avoid NaN.

    NOTE: When using LocalEmbeddingModel (BGE), normalize_embeddings=True in
    SentenceTransformer.encode() already produces unit vectors, so this function
    is effectively a no-op for non-zero embeddings.  It is kept here as a safety
    net for custom embedding models that do not pre-normalise.
    """
    copy = embeddings.astype(np.float32).copy()
    norms = np.linalg.norm(copy, axis=1, keepdims=True)
    # Replace zero norms with 1 so division is safe for degenerate vectors
    norms = np.where(norms < 1e-9, 1.0, norms)
    copy = copy / norms
    return copy


class FAISSVectorStore:
    """Cosine-similarity search via FAISS IndexFlatIP + L2 normalisation.

    L2-normalised vectors satisfy: dot(a, b) == cosine_similarity(a, b).
    This is preferred over Euclidean distance for sentence embeddings because
    semantic similarity is direction-based, not magnitude-based.

    Satisfies the VectorStore Protocol without inheriting from it.
    """

    def __init__(self, dimension: int) -> None:
        """Initialise an empty FAISS index.

        Args:
            dimension: Embedding vector dimension.
        """
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self._dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        """Normalise and add embeddings to the FAISS index.

        Args:
            embeddings: Raw (un-normalised) float32 array of shape (n, dimension).
            chunk_ids: String IDs for each embedding.
            texts: Raw texts (stored for BM25 / display, not used by FAISS).
        """
        normed = _safe_normalize(embeddings)
        # faiss.normalize_L2 would mutate in-place; we pre-normalise above
        self._index.add(normed)
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Search for top_k most similar chunks.

        Args:
            query_embedding: Shape (1, d) or (d,). Will be L2-normalised internally.
            top_k: Number of results.

        Returns:
            Sorted list of (chunk_id, cosine_score) in descending order.
        """
        q = query_embedding.reshape(1, -1).astype(np.float32)
        q_normed = _safe_normalize(q)
        scores, indices = self._index.search(q_normed, min(top_k, len(self._chunk_ids)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._chunk_ids[idx], float(score)))
        return results

    def get_chunks(self) -> list[tuple[str, str]]:
        """Return all (chunk_id, text) pairs."""
        return list(zip(self._chunk_ids, self._texts))
