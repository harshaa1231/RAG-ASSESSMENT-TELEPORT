"""BM25 sparse retrieval index backed by rank_bm25."""
from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Store:
    """Sparse term-based retrieval using Okapi BM25.

    Satisfies no shared Protocol — used directly by HybridRetriever.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def add(self, texts: list[str], chunk_ids: list[str]) -> None:
        """Tokenise and index a corpus.

        Args:
            texts: Raw text for each chunk.
            chunk_ids: Corresponding unique identifiers.
        """
        tokenised = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenised)
        self._chunk_ids = list(chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) pairs.

        Args:
            query: Raw query string (will be lowercased and split).
            top_k: Maximum results to return.

        Returns:
            List of (chunk_id, score) sorted by descending BM25 score.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Store is empty — call add() before search()")
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked_indices]
