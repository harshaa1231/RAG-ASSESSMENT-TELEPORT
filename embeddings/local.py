"""Local simulation of Vertex AI textembedding-gecko using multilingual mpnet.

Model choice: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - 768-dimensional embeddings, same as textembedding-gecko
  - Multilingual (50+ languages), matching gecko's multilingual support
  - Closest open-source behavioral equivalent to gecko for retrieval tasks

In production this is replaced by a one-line swap to the real Vertex AI SDK:
    from vertexai.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

Gecko does not use asymmetric query prefixes; QUERY_PREFIX is set to the empty
string so embed_query() and embed_documents() encode identically.  The separate
method names are preserved so the EmbeddingModel Protocol is satisfied and a
future BGE-style asymmetric model can be swapped in with no caller changes.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Gecko does not use asymmetric query prefixes — empty string is a no-op.
QUERY_PREFIX = ""


class LocalEmbeddingModel:
    """Local simulation of Vertex AI textembedding-gecko.

    Uses 768-dim multilingual mpnet as closest open-source behavioral equivalent.

    Satisfies the EmbeddingModel Protocol without inheriting from it.
    Both embed_documents() and embed_query() use normalize_embeddings=True so
    returned vectors are L2-unit-length, making FAISS IndexFlatIP mathematically
    equivalent to cosine similarity search without any additional normalisation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ) -> None:
        """Load the sentence-transformer model.

        Args:
            model_name: HuggingFace model identifier.
        """
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dimension: int = get_dim()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode corpus documents.

        Args:
            texts: Batch of passage/document strings.

        Returns:
            L2-normalised float32 array of shape (len(texts), dimension).
        """
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a query string.

        QUERY_PREFIX is empty for gecko-equivalent models.  The method exists
        to satisfy the EmbeddingModel Protocol and allow asymmetric models to be
        dropped in without changing any caller.

        Args:
            query: Raw user query string.

        Returns:
            L2-normalised float32 array of shape (1, dimension).
        """
        prefixed = QUERY_PREFIX + query
        return self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def get_dimension(self) -> int:
        """Return the embedding dimension (768 for mpnet / gecko)."""
        return self._dimension
