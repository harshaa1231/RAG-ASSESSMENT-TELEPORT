"""Unit tests for MRR, NDCG, Precision@k, cosine_drift, and latency.

All tests use known inputs and pre-computed expected outputs.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from benchmark.metrics import (
    cosine_drift,
    measure_latency_ms,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    build_query_metrics,
)


# ---------------------------------------------------------------------------
# MRR@3
# ---------------------------------------------------------------------------


def test_mrr_first_result_relevant():
    """Relevant at rank 1 → MRR = 1.0."""
    assert mrr_at_k(["a", "b", "c"], {"a": 2, "b": 0, "c": 0}) == 1.0


def test_mrr_second_result_relevant():
    """Relevant at rank 2 → MRR = 0.5."""
    assert mrr_at_k(["a", "b", "c"], {"a": 0, "b": 1, "c": 0}) == 0.5


def test_mrr_third_result_relevant():
    """Relevant at rank 3 → MRR = 1/3 ≈ 0.333."""
    assert mrr_at_k(["a", "b", "c"], {"a": 0, "b": 0, "c": 2}) == pytest.approx(1 / 3)


def test_mrr_no_relevant():
    """No relevant result in top-3 → MRR = 0.0."""
    assert mrr_at_k(["a", "b", "c"], {"a": 0, "b": 0, "c": 0}) == 0.0


def test_mrr_empty_retrieved():
    assert mrr_at_k([], {"a": 2}) == 0.0


def test_mrr_partial_relevance_counts():
    """Relevance > 0 (including partial label 1) should count."""
    assert mrr_at_k(["x"], {"x": 1}) == 1.0


def test_mrr_cutoff_respected():
    """Result at rank 4 must not contribute when k=3."""
    assert mrr_at_k(["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 0, "d": 2}, k=3) == 0.0


# ---------------------------------------------------------------------------
# NDCG@3
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking():
    """Perfect ranking (most relevant first) → NDCG = 1.0."""
    relevance = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], relevance) == pytest.approx(1.0)


def test_ndcg_all_irrelevant():
    """All retrieved results are irrelevant → NDCG = 0.0."""
    assert ndcg_at_k(["a", "b", "c"], {"a": 0, "b": 0, "c": 0}) == 0.0


def test_ndcg_single_relevant_at_rank1():
    relevance = {"a": 2, "b": 0, "c": 0}
    # DCG = 2/log2(2) = 2; IDCG = 2/log2(2) = 2; NDCG = 1.0
    assert ndcg_at_k(["a", "b", "c"], relevance) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank2():
    relevance = {"a": 0, "b": 2, "c": 0}
    # DCG = 2/log2(3); IDCG = 2/log2(2) = 2
    dcg = 2 / math.log2(3)
    idcg = 2 / math.log2(2)
    expected = dcg / idcg
    assert ndcg_at_k(["a", "b", "c"], relevance) == pytest.approx(expected)


def test_ndcg_reversed_is_less_than_perfect():
    relevance = {"a": 2, "b": 1, "c": 0}
    perfect = ndcg_at_k(["a", "b", "c"], relevance)
    reversed_ = ndcg_at_k(["b", "a", "c"], relevance)
    assert reversed_ < perfect


# ---------------------------------------------------------------------------
# Precision@3
# ---------------------------------------------------------------------------


def test_p3_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a": 2, "b": 1, "c": 2}) == pytest.approx(1.0)


def test_p3_none_relevant():
    assert precision_at_k(["a", "b", "c"], {"a": 0, "b": 0, "c": 0}) == 0.0


def test_p3_one_relevant():
    assert precision_at_k(["a", "b", "c"], {"a": 0, "b": 0, "c": 1}) == pytest.approx(1 / 3)


def test_p3_cutoff_at_k():
    """Fourth result must not affect P@3."""
    result = precision_at_k(["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 0, "d": 2}, k=3)
    assert result == 0.0


# ---------------------------------------------------------------------------
# cosine_drift
# ---------------------------------------------------------------------------


def test_cosine_drift_identical_vectors():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_drift(v, v) == pytest.approx(0.0)


def test_cosine_drift_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert cosine_drift(v1, v2) == pytest.approx(1.0)


def test_cosine_drift_opposite_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert cosine_drift(v1, v2) == pytest.approx(2.0)


def test_cosine_drift_zero_vector_returns_one():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 0.0])
    assert cosine_drift(v1, v2) == pytest.approx(1.0)


def test_cosine_drift_unnormalised_vectors():
    """cosine_drift normalises internally — scale should not matter."""
    v1 = np.array([3.0, 4.0])
    v2 = np.array([6.0, 8.0])  # same direction, different magnitude
    assert cosine_drift(v1, v2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def test_latency_returns_positive():
    latency = measure_latency_ms(lambda: None, runs=3)
    assert latency >= 0.0


def test_latency_is_milliseconds():
    """A sleep of 5ms should produce a median close to 5ms."""
    def _sleep_5ms():
        time.sleep(0.005)

    latency = measure_latency_ms(_sleep_5ms, runs=5)
    assert 3.0 <= latency <= 30.0  # generous range for CI jitter


def test_latency_runs_single():
    latency = measure_latency_ms(lambda: None, runs=1)
    assert latency >= 0.0


# ---------------------------------------------------------------------------
# build_query_metrics integration
# ---------------------------------------------------------------------------


def test_build_query_metrics_assembles_correctly():
    relevance = {"c0": 2, "c1": 0, "c2": 1}
    pairs = [("c0", 0.9), ("c1", 0.5), ("c2", 0.3)]
    orig = np.array([1.0, 0.0])
    exp = np.array([0.0, 1.0])

    qm = build_query_metrics(
        query_id="q1",
        query_text="test query",
        strategy="B_good",
        expansion_mode="good_expansion",
        retrieved_pairs=pairs,
        relevance=relevance,
        latency_ms=5.0,
        original_embedding=orig,
        expanded_embedding=exp,
    )

    assert qm.mrr == pytest.approx(1.0)
    assert 0.0 <= qm.ndcg <= 1.0
    assert qm.p3 == pytest.approx(2 / 3)
    assert qm.latency_ms == pytest.approx(5.0)
    assert qm.cosine_drift == pytest.approx(1.0)
    assert len(qm.retrieved) == 3
    assert qm.retrieved[0].rank == 1
    assert qm.retrieved[0].chunk_id == "c0"
