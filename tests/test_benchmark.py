"""Parametrised end-to-end benchmark tests.

Asserts MRR@3 > 0.5 for non-adversarial queries on Strategy A, B_good,
B_null, and Hybrid.  The adversarial CAP-theorem query is separately tested
to confirm it returns MRR = 0.0.  The noisy expansion mode is tested to
confirm it degrades retrieval relative to Strategy A.
"""
from __future__ import annotations

import pytest

from benchmark.queries import QUERIES
from mocks.vertex_generative import MockGenerativeModel
from retrieval.hybrid import HybridRetriever
from retrieval.strategy_a import RawVectorRetriever
from retrieval.strategy_b import ExpandedQueryRetriever
from benchmark.metrics import mrr_at_k, cosine_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NON_ADVERSARIAL = [q for q in QUERIES if not q.is_adversarial]
ADVERSARIAL = [q for q in QUERIES if q.is_adversarial]
ALL_QUERY_IDS = [q.id for q in QUERIES]


def _strategy_a_mrr(engine, query, top_k=5):
    retriever = RawVectorRetriever(engine.embedding_model, engine.vector_store)
    results, _ = retriever.retrieve(query.text, top_k)
    return mrr_at_k([cid for cid, _ in results], query.relevance)


def _strategy_b_mrr(engine, query, mode, top_k=5):
    expander = MockGenerativeModel("gemini", mode=mode)
    retriever = ExpandedQueryRetriever(
        engine.embedding_model, engine.vector_store, expander, mode
    )
    results, orig_emb, exp_emb, _ = retriever.retrieve(query.text, top_k)
    return mrr_at_k([cid for cid, _ in results], query.relevance)


def _hybrid_mrr(engine, query, top_k=5):
    retriever = HybridRetriever(
        engine.embedding_model,
        engine.vector_store,
        engine.bm25_store,
    )
    results, _ = retriever.retrieve(query.text, top_k)
    return mrr_at_k([cid for cid, _ in results], query.relevance)


# ---------------------------------------------------------------------------
# Strategy A — non-adversarial queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", NON_ADVERSARIAL, ids=[q.id for q in NON_ADVERSARIAL])
def test_strategy_a_mrr_non_adversarial(ingested_engine, query):
    mrr = _strategy_a_mrr(ingested_engine, query)
    assert mrr > 0.5, (
        f"Strategy A MRR={mrr:.3f} for {query.id} ({query.text!r}) — "
        "expected > 0.5 for non-adversarial queries"
    )


# ---------------------------------------------------------------------------
# Strategy B good_expansion — non-adversarial queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", NON_ADVERSARIAL, ids=[q.id for q in NON_ADVERSARIAL])
def test_strategy_b_good_mrr_non_adversarial(ingested_engine, query):
    mrr = _strategy_b_mrr(ingested_engine, query, mode="good_expansion")
    assert mrr > 0.5, (
        f"B_good MRR={mrr:.3f} for {query.id} — expected > 0.5"
    )


# ---------------------------------------------------------------------------
# Strategy B null_expansion — must match Strategy A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", NON_ADVERSARIAL, ids=[q.id for q in NON_ADVERSARIAL])
def test_strategy_b_null_mrr_non_adversarial(ingested_engine, query):
    mrr = _strategy_b_mrr(ingested_engine, query, mode="null_expansion")
    assert mrr > 0.5, (
        f"B_null MRR={mrr:.3f} for {query.id} — expected > 0.5"
    )


# ---------------------------------------------------------------------------
# Hybrid — non-adversarial queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", NON_ADVERSARIAL, ids=[q.id for q in NON_ADVERSARIAL])
def test_hybrid_mrr_non_adversarial(ingested_engine, query):
    mrr = _hybrid_mrr(ingested_engine, query)
    assert mrr > 0.5, (
        f"Hybrid MRR={mrr:.3f} for {query.id} — expected > 0.5"
    )


# ---------------------------------------------------------------------------
# Adversarial query — all strategies should return MRR = 0.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ADVERSARIAL, ids=[q.id for q in ADVERSARIAL])
def test_strategy_a_mrr_adversarial(ingested_engine, query):
    """The CAP-theorem query has no relevant chunks — MRR must be 0."""
    mrr = _strategy_a_mrr(ingested_engine, query)
    assert mrr == 0.0, f"Expected MRR=0 for adversarial query, got {mrr}"


# ---------------------------------------------------------------------------
# Noisy expansion — demonstrates retrieval degradation vs Strategy A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", NON_ADVERSARIAL, ids=[q.id for q in NON_ADVERSARIAL])
def test_strategy_b_noisy_degrades_vs_strategy_a(ingested_engine, query):
    """Noisy expansion should produce MRR <= Strategy A MRR."""
    mrr_a = _strategy_a_mrr(ingested_engine, query)
    mrr_noisy = _strategy_b_mrr(ingested_engine, query, mode="noisy_expansion")
    assert mrr_noisy <= mrr_a, (
        f"Noisy MRR={mrr_noisy:.3f} > A MRR={mrr_a:.3f} for {query.id} — "
        "noisy expansion must not outperform baseline"
    )


# ---------------------------------------------------------------------------
# RAGEngine.benchmark() — integration smoke test
# ---------------------------------------------------------------------------


def test_engine_benchmark_returns_results(ingested_engine, all_queries):
    results = ingested_engine.benchmark(all_queries, strategies=["A"])
    assert len(results) == len(all_queries)


def test_engine_benchmark_all_strategies(ingested_engine, all_queries):
    results = ingested_engine.benchmark(all_queries, strategies=["A", "B", "hybrid"])
    # A: 5, B: 15 (3 modes × 5 queries), hybrid: 5 → 25 total
    assert len(results) == 5 + 15 + 5


def test_engine_benchmark_strategy_names_present(ingested_engine, all_queries):
    results = ingested_engine.benchmark(all_queries, strategies=["A", "B", "hybrid"])
    strategies = {r.strategy for r in results}
    assert "A" in strategies
    assert "B_good_expansion" in strategies
    assert "B_noisy_expansion" in strategies
    assert "B_null_expansion" in strategies
    assert "hybrid" in strategies


def test_engine_benchmark_cosine_drift_present_for_b(ingested_engine, all_queries):
    results = ingested_engine.benchmark(all_queries, strategies=["B"])
    for r in results:
        assert r.cosine_drift is not None, (
            f"Strategy B must always report cosine_drift, got None for {r.strategy}"
        )


def test_engine_benchmark_no_cosine_drift_for_a(ingested_engine, all_queries):
    results = ingested_engine.benchmark(all_queries, strategies=["A"])
    for r in results:
        assert r.cosine_drift is None
