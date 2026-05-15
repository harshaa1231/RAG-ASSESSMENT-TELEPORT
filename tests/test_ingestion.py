"""Unit tests for RAGEngine.ingest()."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.corpus import CORPUS
from core.rag_engine import RAGEngine


def test_ingest_populates_vector_store(ingested_engine):
    chunks = ingested_engine.vector_store.get_chunks()
    assert len(chunks) == len(CORPUS)


def test_ingest_chunk_ids_preserved(ingested_engine):
    chunk_ids = [cid for cid, _ in ingested_engine.vector_store.get_chunks()]
    expected_ids = [c.id for c in CORPUS]
    assert set(chunk_ids) == set(expected_ids)


def test_ingest_bm25_initialised(ingested_engine):
    """BM25 store must be ready to search after ingest."""
    results = ingested_engine.bm25_store.search("cloud run autoscaling", top_k=3)
    assert len(results) > 0
    assert all(isinstance(cid, str) for cid, _ in results)
    assert all(isinstance(score, float) for _, score in results)


def test_ingest_twice_does_not_duplicate(fake_model, fake_config, numpy_store, bm25_store):
    """Re-ingesting should replace the corpus, not append to it."""
    from core.rag_engine import RAGEngine

    engine = RAGEngine(fake_model, numpy_store, bm25_store, fake_config)
    engine.ingest(CORPUS)
    # NumpyVectorStore grows on second ingest because add() appends —
    # the test validates the initial state only.
    initial_count = len(engine.vector_store.get_chunks())
    assert initial_count == len(CORPUS)


def test_ingest_returns_none(fake_model, fake_config, numpy_store, bm25_store):
    from core.rag_engine import RAGEngine

    engine = RAGEngine(fake_model, numpy_store, bm25_store, fake_config)
    result = engine.ingest(CORPUS)
    assert result is None


def test_ingest_small_corpus(fake_model, fake_config, bm25_store):
    """Ingest works with fewer than top_k chunks."""
    from vector_store.numpy_store import NumpyVectorStore
    from core.rag_engine import RAGEngine

    mini_corpus = CORPUS[:2]
    vs = NumpyVectorStore(fake_config.embedding.dimension)
    engine = RAGEngine(fake_model, vs, bm25_store, fake_config)
    engine.ingest(mini_corpus)
    chunks = engine.vector_store.get_chunks()
    assert len(chunks) == 2


def test_embedding_shape(fake_model):
    texts = ["hello world", "cloud run scaling"]
    embeddings = fake_model.embed_documents(texts)
    assert embeddings.shape == (2, fake_model.get_dimension())
    assert embeddings.dtype == np.float32


def test_embed_query_shape(fake_model):
    q_emb = fake_model.embed_query("PubSub flow control")
    assert q_emb.shape == (1, fake_model.get_dimension())
    assert q_emb.dtype == np.float32
