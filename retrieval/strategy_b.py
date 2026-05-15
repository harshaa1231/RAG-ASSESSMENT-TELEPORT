"""Strategy B: expanded-query retrieval with three mock modes.

The mock generative model handles the actual expansion text; this module
wires the expansion into the retrieve-and-measure pipeline and computes
cosine_drift between the original and expanded query embeddings.
"""
from __future__ import annotations

import numpy as np

from embeddings.base import EmbeddingModel
from expansion.base import QueryExpander
from vector_store.base import VectorStore

_EXPANSION_PROMPT_TEMPLATE = (
    "Rewrite the following search query to improve document retrieval. "
    "Return only the rewritten query, no explanation.\n\nQuery: {query}"
)


class ExpandedQueryRetriever:
    """Rewrites the query before embedding, then retrieves from the dense index.

    cosine_drift is computed between the original and expanded query embeddings
    and returned alongside the retrieval results so callers can log it.

    Mode is encoded in the injected QueryExpander — this class is mode-agnostic.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        expander: QueryExpander,
        expansion_mode: str,
    ) -> None:
        """Initialise Strategy B.

        Args:
            embedding_model: Satisfies EmbeddingModel Protocol.
            vector_store: Satisfies VectorStore Protocol.
            expander: Satisfies QueryExpander Protocol (mock or real LLM).
            expansion_mode: Label stored in results ("good_expansion" etc.).
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.expander = expander
        self.expansion_mode = expansion_mode

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[tuple[str, float]], np.ndarray, np.ndarray, str]:
        """Expand, embed, and retrieve.

        Args:
            query: Original query string.
            top_k: Maximum results to return.

        Returns:
            Tuple of:
              - List of (chunk_id, cosine_score) in descending order.
              - Original query embedding array (shape 1 × d).
              - Expanded query embedding array (shape 1 × d).
              - Expanded query text.
        """
        prompt = _EXPANSION_PROMPT_TEMPLATE.format(query=query)
        expanded_text = self.expander.expand(prompt)

        original_embedding = self.embedding_model.embed_query(query)
        expanded_embedding = self.embedding_model.embed_query(expanded_text)

        results = self.vector_store.search(expanded_embedding, top_k)
        return results, original_embedding, expanded_embedding, expanded_text
