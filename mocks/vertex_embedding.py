"""Mock that mirrors vertexai.language_models.TextEmbeddingModel exactly.

Production target: vertexai.language_models.TextEmbeddingModel (textembedding-gecko@003).
Local simulation: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-dim).

The real SDK interface:
    from vertexai.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    results = model.get_embeddings(["text"])   # -> list[TextEmbedding]
    results[0].values                           # -> list[float]
    results[0].statistics.token_count           # -> int

This mock preserves every attribute and method signature so that swapping
`MockTextEmbeddingModel` for the real `TextEmbeddingModel` is a one-line change.
gecko does not use asymmetric query prefixes; embed_query() and embed_documents()
behave identically and delegate to the underlying LocalEmbeddingModel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embeddings.base import EmbeddingModel


# ---------------------------------------------------------------------------
# Mirror of vertexai.language_models.TextEmbeddingStatistics
# ---------------------------------------------------------------------------


@dataclass
class MockTextEmbeddingStatistics:
    """Mirrors TextEmbeddingStatistics (token_count only used attribute)."""

    token_count: int


# ---------------------------------------------------------------------------
# Mirror of vertexai.language_models.TextEmbedding
# ---------------------------------------------------------------------------


@dataclass
class MockTextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding."""

    values: list[float]
    statistics: MockTextEmbeddingStatistics


# ---------------------------------------------------------------------------
# Mirror of vertexai.language_models.TextEmbeddingModel
# ---------------------------------------------------------------------------


class MockTextEmbeddingModel:
    """Drop-in replacement for vertexai.language_models.TextEmbeddingModel.

    Delegates actual embedding computation to any EmbeddingModel implementation,
    preserving the real SDK's class-method factory and instance API.

    Migration: replace ``MockTextEmbeddingModel`` with the real
    ``TextEmbeddingModel`` in ``RAGEngine.from_config()`` — zero other changes.
    """

    def __init__(
        self,
        model_name: str,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._model_name = model_name
        self._embedding_model = embedding_model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        embedding_model: EmbeddingModel | None = None,
    ) -> "MockTextEmbeddingModel":
        """Factory that mirrors TextEmbeddingModel.from_pretrained(model_name).

        Args:
            model_name: Vertex AI model ID (e.g. "textembedding-gecko@003").
            embedding_model: Optional real EmbeddingModel for delegation.
                             If None, a LocalEmbeddingModel is constructed lazily.
        """
        if embedding_model is None:
            from embeddings.local import LocalEmbeddingModel
            embedding_model = LocalEmbeddingModel()
        return cls(model_name=model_name, embedding_model=embedding_model)

    def get_embeddings(self, texts: list[str]) -> list[MockTextEmbedding]:
        """Compute embeddings — identical signature to the real SDK.

        Delegates to embed_documents() (no query prefix) to mirror Vertex AI's
        symmetric embedding API, which does not distinguish query vs document paths.

        Args:
            texts: List of strings to embed.

        Returns:
            List of MockTextEmbedding objects with .values and .statistics.
        """
        raw: np.ndarray = self.embed_documents(texts)
        return [
            MockTextEmbedding(
                values=row.tolist(),
                statistics=MockTextEmbeddingStatistics(
                    token_count=len(text.split())
                ),
            )
            for row, text in zip(raw, texts)
        ]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode corpus documents without any query prefix.

        Args:
            texts: List of passage/document strings.

        Returns:
            Float32 array of shape (len(texts), dimension).
        """
        return self._embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a query string.

        gecko does not use asymmetric prefixes so this delegates to embed_documents()
        via the underlying LocalEmbeddingModel.embed_query() which uses QUERY_PREFIX="".

        Args:
            query: Raw user query string.

        Returns:
            Float32 array of shape (1, dimension).
        """
        return self._embedding_model.embed_query(query)

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_model.get_dimension()
