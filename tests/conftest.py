"""Shared pytest fixtures.

FakeEmbeddingModel assigns deterministic keyword-based embeddings so that:
  - Tests run without downloading any ML model.
  - Non-adversarial queries retrieve relevant chunks first → MRR >= 1.0.
  - Noisy-expansion embeddings are orthogonal to the corpus → MRR = 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.corpus import CORPUS, Chunk
from benchmark.queries import QUERIES, Query
from core.config import BenchmarkConfig
from embeddings.sparse import BM25Store
from vector_store.numpy_store import NumpyVectorStore


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS: list[tuple[int, list[str]]] = [
    (0, ["cloud run", "cloudrun"]),
    (1, ["autoscal", "traffic", "peak", "scale", "burst"]),
    (2, ["bigquery", "big query"]),
    (3, ["slot reservation", "slot commit", "slot alloc"]),
    (4, ["pub/sub", "pubsub"]),
    (5, ["flow control", "backpressure", "outstanding"]),
    (6, ["vertex ai", "vertexai"]),
    (7, ["serving", "endpoint", "inference", "latency"]),
    (8, ["gke", "kubernetes"]),
    (9, ["cluster autoscal", "node pool", "autoprovis"]),
]

_DIM = 32  # must be >= 10 to hold keyword dims


class FakeEmbeddingModel:
    """Deterministic keyword-based embedding model used in tests.

    Implements the asymmetric EmbeddingModel Protocol (embed_documents /
    embed_query) so tests remain model-agnostic after the BGE migration.
    The query prefix is stripped before matching so query and document
    embeddings align on the same keyword dimensions.
    """

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def _vectorise(self, texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            for dim, keywords in _KEYWORD_GROUPS:
                if any(kw in lower for kw in keywords):
                    embeddings[i, dim] = 1.0
        return embeddings

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._vectorise(texts)

    def embed_query(self, query: str) -> np.ndarray:
        # Strip the BGE prefix if present so keyword matching still works
        clean = query.replace(self._QUERY_PREFIX, "")
        return self._vectorise([clean])

    def get_dimension(self) -> int:
        return _DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def fake_config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.embedding.dimension = _DIM
    config.retrieval.top_k = 5
    config.retrieval.latency_runs = 2
    return config


@pytest.fixture
def numpy_store() -> NumpyVectorStore:
    return NumpyVectorStore(_DIM)


@pytest.fixture
def bm25_store() -> BM25Store:
    return BM25Store()


@pytest.fixture
def ingested_engine(fake_model, fake_config, numpy_store, bm25_store):
    """RAGEngine with fake embeddings and the full corpus already ingested."""
    from core.rag_engine import RAGEngine

    engine = RAGEngine(fake_model, numpy_store, bm25_store, fake_config)
    engine.ingest(CORPUS)
    return engine


@pytest.fixture
def sample_chunk() -> Chunk:
    return CORPUS[0]


@pytest.fixture
def sample_query() -> Query:
    return QUERIES[4]  # q5: "PubSub flow control" — keyword-exact, easy to verify


@pytest.fixture
def all_queries() -> list[Query]:
    return QUERIES


@pytest.fixture
def non_adversarial_queries() -> list[Query]:
    return [q for q in QUERIES if not q.is_adversarial]
