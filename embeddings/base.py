"""Protocol definition for embedding models — no inheritance required.

BGE and similar asymmetric models require separate encode paths for queries
and documents.  The Protocol reflects this with embed_documents() and
embed_query() as distinct methods.
"""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural interface satisfied by any asymmetric embedding model.

    Asymmetric encoding: queries and documents are embedded differently.
    For BGE models a prefix is prepended to query text before encoding so
    the query embedding sits in the same neighbourhood as passage embeddings.
    Document embeddings have no prefix.
    """

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of corpus documents (no query prefix).

        Args:
            texts: List of passage/document strings to embed.

        Returns:
            Float32 array of shape (len(texts), dimension).
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string (with prefix for asymmetric models).

        Args:
            query: User query string.

        Returns:
            Float32 array of shape (1, dimension).
        """
        ...

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...
