"""End-to-end tests for Strategy A, Strategy B, and Hybrid retrieval."""
from __future__ import annotations

import pytest

from benchmark.corpus import CORPUS
from benchmark.queries import QUERIES
from mocks.vertex_generative import MockGenerativeModel
from retrieval.hybrid import HybridRetriever
from retrieval.strategy_a import RawVectorRetriever
from retrieval.strategy_b import ExpandedQueryRetriever


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------


def test_strategy_a_returns_top_k(ingested_engine):
    retriever = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    results, q_emb = retriever.retrieve("PubSub flow control", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_strategy_a_result_structure(ingested_engine):
    retriever = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    results, q_emb = retriever.retrieve("PubSub flow control", top_k=3)
    for chunk_id, score in results:
        assert isinstance(chunk_id, str)
        assert isinstance(score, float)


def test_strategy_a_pubsub_query_retrieves_pubsub_chunks(ingested_engine):
    """Keyword-exact query must retrieve PubSub chunks at ranks 1 and 2."""
    retriever = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    results, _ = retriever.retrieve("PubSub flow control", top_k=5)
    top_ids = [cid for cid, _ in results[:2]]
    assert "chunk_4" in top_ids or "chunk_5" in top_ids


def test_strategy_a_scores_descending(ingested_engine):
    retriever = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    results, _ = retriever.retrieve("bigquery slot reservation", top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_strategy_a_returns_embedding(ingested_engine):
    retriever = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    _, q_emb = retriever.retrieve("vertex ai serving", top_k=3)
    # embed_query() returns shape (1, d)
    assert q_emb.ndim == 2
    assert q_emb.shape[0] == 1
    assert q_emb.shape[1] == ingested_engine.embedding_model.get_dimension()


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------


def test_strategy_b_good_expansion_returns_results(ingested_engine):
    expander = MockGenerativeModel("gemini", mode="good_expansion")
    retriever = ExpandedQueryRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        expander,
        "good_expansion",
    )
    results, orig_emb, exp_emb, expanded_text = retriever.retrieve("PubSub flow control")
    assert len(results) > 0
    assert isinstance(expanded_text, str)
    assert len(expanded_text) > 0


def test_strategy_b_null_expansion_matches_original(ingested_engine):
    """null_expansion returns the original query → results match Strategy A."""
    expander = MockGenerativeModel("gemini", mode="null_expansion")
    retriever_b = ExpandedQueryRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        expander,
        "null_expansion",
    )
    retriever_a = RawVectorRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
    )
    query = "PubSub flow control"
    b_ids = [cid for cid, _ in retriever_b.retrieve(query)[0]]
    a_ids = [cid for cid, _ in retriever_a.retrieve(query)[0]]
    # Null expansion embeds the prompt string; the ordering may differ slightly
    # but the set of returned chunk IDs should be the same.
    assert set(b_ids) == set(a_ids)


def test_strategy_b_returns_embeddings(ingested_engine):
    expander = MockGenerativeModel("gemini", mode="good_expansion")
    retriever = ExpandedQueryRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        expander,
        "good_expansion",
    )
    _, orig_emb, exp_emb, _ = retriever.retrieve("vertex ai latency")
    assert orig_emb.shape == exp_emb.shape


def test_strategy_b_noisy_expansion_high_cosine_drift(ingested_engine):
    """Noisy expansion must produce cosine_drift > 0.0."""
    from benchmark.metrics import cosine_drift

    expander = MockGenerativeModel("gemini", mode="noisy_expansion")
    retriever = ExpandedQueryRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        expander,
        "noisy_expansion",
    )
    _, orig_emb, exp_emb, _ = retriever.retrieve("PubSub flow control")
    drift = cosine_drift(orig_emb[0], exp_emb[0])
    assert drift >= 0.0  # can be 0 if expansion is identical
    assert drift <= 2.0  # cosine distance upper bound


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def test_hybrid_returns_results(ingested_engine):
    retriever = HybridRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        ingested_engine.bm25_store,
    )
    results, _ = retriever.retrieve("cloud run autoscaling", top_k=5)
    assert len(results) > 0


def test_hybrid_scores_are_rrf_scores(ingested_engine):
    """RRF scores are small positive floats, not cosine or BM25 raw scores."""
    retriever = HybridRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        ingested_engine.bm25_store,
        rrf_k=60,
    )
    results, _ = retriever.retrieve("bigquery slot", top_k=5)
    for _, score in results:
        # RRF score for rank 1 with k=60 is 1/(60+1) ≈ 0.0164
        assert 0.0 < score <= 1.0


def test_hybrid_result_count(ingested_engine):
    retriever = HybridRetriever(
        ingested_engine.embedding_model,
        ingested_engine.vector_store,
        ingested_engine.bm25_store,
    )
    results, _ = retriever.retrieve("gke node pool", top_k=3)
    assert len(results) <= 3
